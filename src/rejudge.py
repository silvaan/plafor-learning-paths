"""Judge v2: reason-first, anchored, pool-aware rubric scored by several LLM judges.

Changes from the v1 judge (judge.py), motivated by its diagnostics in the paper:
  * each dimension is argued before it is scored (reason-first), with anchors per level;
  * the judge sees the full candidate pool, so coverage is judged against what was
    actually available rather than against an imagined ideal catalog;
  * length is separated from coverage through an explicit `concision` dimension;
  * judges from two model families (OpenAI, Anthropic) score every path, so
    agreement and self-preference can be measured.

Usage:
    python src/rejudge.py --judges gpt-4o claude-opus-5-5
    python src/rejudge.py --judges claude-opus-5-5 --filter-run 0
    python src/rejudge.py --judges gpt-4o --stratified --batch submit   # OpenAI Batch API, 50% off
    python src/rejudge.py --judges gpt-4o --batch collect
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import PlaforEduDB

RESULTS = Path(__file__).resolve().parent.parent / "results"
GEN_PATH = RESULTS / "gen_v2.jsonl"
JUDGE_PATH = RESULTS / "judgments_v2.jsonl"
BATCH_DIR = RESULTS / "batch"
# Batch API prices in USD per 1M tokens (input, output), and the per-judgment token
# means measured on the synchronous gpt-4o run, used to cap spending before submitting.
BATCH_PRICE = {"gpt-4o": (1.25, 5.00)}
MEAN_TOKENS = (1511, 396)

JUDGE_V2_PROMPT = """Você é um avaliador especialista em design instrucional e curadoria de trilhas formativas para servidores públicos. Avalie a trilha proposta (sequência ordenada de cursos) para o objetivo abaixo. A trilha só pode usar cursos do conjunto de candidatos do catálogo.

# Objetivo da trilha (competência) #
{objective}

# Cursos candidatos disponíveis no catálogo para esta competência ({n_pool}) #
{pool}

# Trilha proposta, na ordem apresentada ({n_path} cursos) #
{path}

# Como avaliar #
Para cada dimensão, primeiro escreva uma análise curta e concreta (cite os cursos), depois dê a nota inteira de 1 a 5 usando as âncoras. Avalie cada dimensão de forma independente; não copie a mesma nota entre dimensões sem justificativa. Um curso marcado como [NÃO EXISTE NO CATÁLOGO] não pode ser cursado e conta como erro grave em relevância.

relevance (precisão): os cursos da trilha são pertinentes ao objetivo?
  5 = todos claramente centrais ao objetivo; 4 = quase todos centrais, no máximo um periférico; 3 = mistura de centrais e periféricos; 2 = maioria periférica ou algum curso inexistente; 1 = majoritariamente irrelevante ou inexistente.
coverage (cobertura): considerando APENAS os candidatos disponíveis, a trilha inclui os cursos necessários para desenvolver a competência?
  5 = nenhum candidato essencial ficou de fora; 4 = falta um candidato útil, mas não essencial; 3 = falta um candidato essencial; 2 = faltam vários essenciais; 1 = a trilha cobre uma parte mínima da competência. Se o catálogo tem poucos candidatos, julgue em relação ao que existe.
ordering (progressão): a ordem forma uma progressão pedagógica coerente?
  5 = fundamentos antes de aprofundamentos, sem inversões; 4 = uma inversão leve; 3 = ordem em parte arbitrária; 2 = várias inversões claras; 1 = ordem contrária à progressão. Trilha com um único curso recebe 3.
concision (concisão): a trilha evita cursos redundantes, sobrepostos ou desnecessários?
  5 = nenhum curso supérfluo; 4 = um curso dispensável; 3 = alguns redundantes ou dispensáveis; 2 = muitos supérfluos; 1 = lista quase sem curadoria.
overall: qualidade geral como percurso formativo a ser oferecido a um servidor, ponderando as dimensões acima."""


class VerdictV2(BaseModel):
    analysis_relevance: str = Field(description="Análise da relevância, citando cursos.")
    relevance: int = Field(description="Nota 1-5.")
    analysis_coverage: str = Field(description="Análise da cobertura frente aos candidatos.")
    coverage: int = Field(description="Nota 1-5.")
    analysis_ordering: str = Field(description="Análise da ordem.")
    ordering: int = Field(description="Nota 1-5.")
    analysis_concision: str = Field(description="Análise de redundância/excesso.")
    concision: int = Field(description="Nota 1-5.")
    overall: int = Field(description="Nota geral 1-5.")


DIMS = ["relevance", "coverage", "ordering", "concision", "overall"]


def build_prompt(db: PlaforEduDB, rec: dict) -> str:
    pool = db.get_courses_for_competency(rec["competency_id"])
    pool_txt = "\n".join(f"- {c['name']}" for c in pool)
    lines = []
    for i, c in enumerate(rec["selected_courses"], 1):
        course = db.get_course(c["id"])
        in_pool = course is not None and any(p["id"] == c["id"] for p in pool)
        if in_pool:
            lines.append(f"{i}. {course['name']}\n   {course['description']}")
        else:
            lines.append(f"{i}. {c.get('name', '?')} [NÃO EXISTE NO CATÁLOGO]")
    return JUDGE_V2_PROMPT.format(objective=rec["objective"], n_pool=len(pool), pool=pool_txt,
                                  n_path=len(lines), path="\n".join(lines))


# --------------------------------------------------------------------------- #
# Judge backends
# --------------------------------------------------------------------------- #
def judge_openai(model: str, prompt: str) -> tuple[VerdictV2, int, int]:
    from openai import OpenAI
    resp = OpenAI().beta.chat.completions.parse(
        model=model, temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
        response_format=VerdictV2,
    )
    return resp.choices[0].message.parsed, resp.usage.prompt_tokens, resp.usage.completion_tokens


def judge_anthropic(model: str, prompt: str) -> tuple[VerdictV2, int, int]:
    import anthropic
    resp = anthropic.Anthropic().messages.parse(
        model=model, max_tokens=16000,
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": prompt}],
        output_format=VerdictV2,
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"refusal: {resp.stop_details}")
    return resp.parsed_output, resp.usage.input_tokens, resp.usage.output_tokens


def run_judge(judge: str, prompt: str):
    return (judge_anthropic if judge.startswith("claude") else judge_openai)(judge, prompt)


def key_of(r: dict) -> tuple:
    return (r["system"], r["gen_model"], r["competency_id"], r.get("run", 0))


def openai_body(model: str, prompt: str) -> dict:
    from openai.lib._parsing._completions import type_to_response_format_param
    return {"model": model, "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": type_to_response_format_param(VerdictV2)}


def batch_submit(judge: str, jobs: list, recs: dict, max_usd: float):
    from openai import OpenAI
    pin, pout = BATCH_PRICE[judge]
    est = len(jobs) * (MEAN_TOKENS[0] * pin + MEAN_TOKENS[1] * pout) / 1e6
    print(f"{len(jobs)} requests, estimated ${est:.2f} (cap ${max_usd:.2f})")
    if est > max_usd:
        sys.exit("estimated cost above cap, not submitting")
    BATCH_DIR.mkdir(exist_ok=True)
    db = PlaforEduDB()
    req_path = BATCH_DIR / f"{judge}_requests.jsonl"
    with req_path.open("w") as f:
        for _, k in jobs:
            f.write(json.dumps({"custom_id": json.dumps(list(k)), "method": "POST",
                                "url": "/v1/chat/completions",
                                "body": openai_body(judge, build_prompt(db, recs[k]))},
                               ensure_ascii=False) + "\n")
    client = OpenAI()
    fid = client.files.create(file=req_path.open("rb"), purpose="batch").id
    b = client.batches.create(input_file_id=fid, endpoint="/v1/chat/completions",
                              completion_window="24h")
    (BATCH_DIR / f"{judge}_batch_id.txt").write_text(b.id)
    print(f"submitted batch {b.id}")


def batch_collect(judge: str):
    from openai import OpenAI
    client = OpenAI()
    b = client.batches.retrieve((BATCH_DIR / f"{judge}_batch_id.txt").read_text().strip())
    print(f"batch {b.id}: {b.status} {b.request_counts}")
    if b.status != "completed":
        return
    pin, pout = BATCH_PRICE[judge]
    n = cost = 0
    with JUDGE_PATH.open("a") as f:
        for l in client.files.content(b.output_file_id).text.splitlines():
            r = json.loads(l)
            body = r["response"]["body"] if r.get("response") else None
            if not body or r["response"]["status_code"] != 200:
                print(f"ERROR {r['custom_id']}: {r.get('error') or body}")
                continue
            d = VerdictV2.model_validate_json(body["choices"][0]["message"]["content"]).model_dump()
            for dim in DIMS:
                d[dim] = max(1, min(5, int(d[dim])))
            u = body["usage"]
            cost += (u["prompt_tokens"] * pin + u["completion_tokens"] * pout) / 1e6
            f.write(json.dumps({"judge": judge, "key": json.loads(r["custom_id"]), **d,
                                "prompt_tokens": u["prompt_tokens"],
                                "completion_tokens": u["completion_tokens"], "batch": True},
                               ensure_ascii=False) + "\n")
            n += 1
    print(f"wrote {n} judgments, actual cost ${cost:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+", default=["gpt-4o"])
    ap.add_argument("--filter-run", type=int, default=None, help="only judge this run id")
    ap.add_argument("--stratified", action="store_true",
                    help="run 0 of every configuration, plus all runs of configurations "
                         "the judge has already largely scored on runs > 0")
    ap.add_argument("--batch", choices=["submit", "collect"], default=None)
    ap.add_argument("--max-usd", type=float, default=4.0, help="spending cap for --batch submit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    recs = {}
    for l in GEN_PATH.read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            recs[key_of(r)] = r
    done = set()
    if JUDGE_PATH.exists():
        for l in JUDGE_PATH.read_text().splitlines():
            if l.strip():
                j = json.loads(l)
                done.add((j["judge"], *j["key"]))
    jobs = [(jd, k) for jd in args.judges for k, r in recs.items()
            if (jd, *k) not in done
            and (args.filter_run is None or r.get("run", 0) == args.filter_run)]
    if args.stratified:
        from collections import Counter
        later = Counter((jd, k[0], k[1]) for jd, *k in done if k[3] > 0)
        full = {c for c, n in later.items() if n >= 100}
        jobs = [(jd, k) for jd, k in jobs if k[3] == 0 or (jd, k[0], k[1]) in full]
    if args.limit:
        jobs = jobs[: args.limit]
    if args.batch == "collect":
        for jd in args.judges:
            batch_collect(jd)
        return
    if args.batch == "submit":
        empty = [(jd, k) for jd, k in jobs if not recs[k]["selected_courses"]]
        with JUDGE_PATH.open("a") as f:
            for jd, k in empty:
                f.write(json.dumps({"judge": jd, "key": list(k), **{d: 1 for d in DIMS},
                                    "empty": True, "prompt_tokens": 0, "completion_tokens": 0}) + "\n")
        jobs = [j for j in jobs if j not in empty]
        for jd in args.judges:
            batch_submit(jd, [j for j in jobs if j[0] == jd], recs, args.max_usd)
        return
    print(f"{len(jobs)} judgments to run ({len(done)} cached)")

    def work(job):
        jd, k = job
        rec = recs[k]
        out = {"judge": jd, "key": list(k)}
        if not rec["selected_courses"]:
            out.update({d: 1 for d in DIMS}, empty=True, prompt_tokens=0, completion_tokens=0)
            return out
        prompt = build_prompt(PlaforEduDB(), rec)
        v, pt, ct = run_judge(jd, prompt)
        d = v.model_dump()
        for dim in DIMS:
            d[dim] = max(1, min(5, int(d[dim])))
        out.update(d, prompt_tokens=pt, completion_tokens=ct)
        return out

    with JUDGE_PATH.open("a") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(work, j): j for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            jd, k = futs[fut]
            try:
                out = fut.result()
            except Exception as e:
                print(f"ERROR {jd} {k}: {e}")
                traceback.print_exc()
                continue
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {jd} {k[0]} {k[1]} -> "
                      + " ".join(f"{d[:3]}={out[d]}" for d in DIMS))


if __name__ == "__main__":
    main()
