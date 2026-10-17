"""Generate learning paths for every (system, generator model, competency, run).

Generation and judging are decoupled: this script only generates and caches
paths (with programmatic metrics) in `results/gen_v2.jsonl`; `rejudge.py` scores
them with one or more LLM judges.

Usage:
    python src/run.py --seed-from-v1                       # import the original run-0 paths
    python src/run.py --systems single_agent multi_agent --models gpt-4o-mini --runs 3
    python src/run.py --systems s1_laya s1_kev4b           # System One decision models
    python src/run.py --smoke --systems s1_laya            # one competency
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import PlaforEduDB
from systems import LLM_SYSTEMS, FIXED_SYSTEMS
from judge import programmatic_metrics

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
GEN_PATH = RESULTS_DIR / "gen_v2.jsonl"
V1_PATH = RESULTS_DIR / "raw_results.jsonl"


def key_of(r: dict) -> tuple:
    return (r["system"], r["gen_model"], r["competency_id"], r.get("run", 0))


def load_done(path: Path) -> set:
    if not path.exists():
        return set()
    return {key_of(json.loads(l)) for l in path.read_text().splitlines() if l.strip()}


def seed_from_v1():
    """Copy the original run-0 generations (judged by the v1 judge) into gen_v2."""
    v1_gen = {"naive_all": "-", "retrieval_topk": "text-embedding-3-small",
              "single_agent": "gpt-4o-mini", "multi_agent": "gpt-4o-mini"}
    judge_cols = ["relevance", "ordering", "coverage", "overall", "rationale",
                  "judge_prompt_tokens", "judge_completion_tokens"]
    done = load_done(GEN_PATH)
    rows = {}
    for l in V1_PATH.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            r["gen_model"] = v1_gen[r["system"]]
            r["v1_judge"] = {c: r.pop(c, None) for c in judge_cols}
            rows[key_of(r)] = r  # keep last, as analyze.py does
    with GEN_PATH.open("a") as f:
        n = 0
        for k, r in rows.items():
            if k not in done:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n += 1
    print(f"seeded {n} records from {V1_PATH.name}")


def merge_into_gen(paths: list[Path]):
    """Append records from side files (written while another run held gen_v2) into gen_v2."""
    done = load_done(GEN_PATH)
    with GEN_PATH.open("a") as f:
        for p in paths:
            n = 0
            for l in p.read_text().splitlines():
                if l.strip() and key_of(json.loads(l)) not in done:
                    f.write(l + "\n")
                    done.add(key_of(json.loads(l)))
                    n += 1
            print(f"merged {n} records from {p.name}")


def derive_s1_top5():
    """Equal-length System One variant: top-5 courses by relevance probability,
    ordered by predicted level. Derived offline from the stored decisions."""
    db = PlaforEduDB()
    done = load_done(GEN_PATH)
    out = []
    for l in GEN_PATH.read_text().splitlines():
        r = json.loads(l) if l.strip() else None
        if not r or not r["system"].startswith("s1_"):
            continue
        d = dict(r, system=r["system"].replace("s1_", "s1top5_"))
        if key_of(d) in done:
            continue
        top = sorted(r["s1_scores"], key=lambda x: -x["p_relevant"])[:5]
        top.sort(key=lambda x: (x["level"], -x["p_relevant"]))
        d["selected_courses"] = [{"id": x["id"], "name": x["name"],
                                  "justification": f"p_relevante={x['p_relevant']:.2f}; nivel={x['level']:.2f}"}
                                 for x in top]
        d["n_selected"] = len(top)
        d.update(programmatic_metrics(db, d))
        out.append(d)
    with GEN_PATH.open("a") as f:
        for d in out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    print(f"derived {len(out)} top-5 System One records")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-from-v1", action="store_true")
    ap.add_argument("--merge", nargs="+", type=Path, default=[], help="side files to merge into gen_v2")
    ap.add_argument("--derive-s1-top5", action="store_true")
    ap.add_argument("--systems", nargs="+", default=[])
    ap.add_argument("--models", nargs="+", default=["gpt-4o-mini"],
                    help="generator specs for LLM systems, e.g. gpt-4o-mini gpt-5.4-mini:none")
    ap.add_argument("--runs", type=int, default=1, help="run ids 0..runs-1")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=GEN_PATH,
                    help="write to a separate file when another run is appending to the default")
    args = ap.parse_args()

    if args.seed_from_v1:
        seed_from_v1()
    if args.merge:
        merge_into_gen(args.merge)
    if args.derive_s1_top5:
        derive_s1_top5()
    if not args.systems:
        return

    db = PlaforEduDB()
    comps = db.competencies_with_courses(min_courses=1)
    comps.sort(key=lambda c: c["n_candidates"])
    if args.smoke:
        comps = [comps[len(comps) // 2]]

    jobs = []
    for run in range(args.runs):
        for comp in comps:
            for s in args.systems:
                if s in LLM_SYSTEMS:
                    for m in args.models:
                        jobs.append((s, m, comp, run))
                elif s in FIXED_SYSTEMS:
                    jobs.append((s, FIXED_SYSTEMS[s][1], comp, run))
                else:
                    sys.exit(f"unknown system {s}")
    done = load_done(GEN_PATH) | load_done(args.out)
    jobs = [j for j in jobs if (j[0], j[1], j[2]["id"], j[3]) not in done]
    print(f"{len(jobs)} generations to run ({len(done)} cached)")

    def work(job):
        s, m, comp, run = job
        local_db = PlaforEduDB()  # sqlite connections are per-thread
        if s in LLM_SYSTEMS:
            rec = LLM_SYSTEMS[s](local_db, comp["id"], gen_model=m)
        else:
            rec = FIXED_SYSTEMS[s][0](local_db, comp["id"])
        rec.update(gen_model=m, run=run, competency_name=comp["name"])
        rec.update(programmatic_metrics(local_db, rec))
        return rec

    # System One servers run on the local GPU: keep them sequential.
    workers = 1 if any(s.startswith("s1_") for s in args.systems) else args.workers
    with args.out.open("a") as f, ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            s, m, comp, run = futs[fut]
            try:
                rec = fut.result()
            except Exception as e:
                print(f"ERROR {s} {m} {comp['id']} run{run}: {e}")
                traceback.print_exc()
                continue
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{i}/{len(jobs)}] run{run} {s:20} {m:22} {comp['name'][:28]:28} "
                  f"sel={rec['n_selected']:>3}/{rec['n_candidates']:<3} "
                  f"halluc={rec['n_hallucinated']} iters={rec['iterations']}")


if __name__ == "__main__":
    main()
