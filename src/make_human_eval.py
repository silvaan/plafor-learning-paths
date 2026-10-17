"""Build the blind human-evaluation workbook and score it against the LLM judges.

    python src/make_human_eval.py build      # -> results/human_eval/avaliacao_trilhas.xlsx + key
    python src/make_human_eval.py score      # after curators fill the workbook(s)

Curators see each path with the competency objective and the candidate pool, never
the system that produced it, and score it on the same anchored rubric as judge v2.
Several filled copies (one per curator) can be dropped into results/human_eval/filled/.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from db import PlaforEduDB
from rejudge import DIMS, JUDGE_V2_PROMPT

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "human_eval"
SYSTEMS = [("retrieval_topk", "text-embedding-3-small"), ("multi_agent", "gpt-4o-mini"),
           ("hybrid", "gpt-4o-mini"), ("single_agent", "gpt-5.4-mini:high"),
           ("s1top5_kev4b", "kev-4b")]
N_COMP = 16
SEED = 7

RUBRIC = JUDGE_V2_PROMPT.split("# Como avaliar #", 1)[1].strip()


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    db = PlaforEduDB()
    gen = {}
    for l in (ROOT / "results" / "gen_v2.jsonl").read_text().splitlines():
        r = json.loads(l)
        if r.get("run", 0) == 0:
            gen[(r["system"], r["gen_model"], r["competency_id"])] = r
    # competencies where curation is non-trivial (pool >= 6) and every system has a path
    comps = sorted({k[2] for k in gen})
    comps = [c for c in comps
             if len(db.get_courses_for_competency(c)) >= 6
             and all((s, m, c) in gen for s, m in SYSTEMS)]
    rnd = random.Random(SEED)
    comps = rnd.sample(comps, min(N_COMP, len(comps)))

    items = []
    for c in comps:
        for s, m in SYSTEMS:
            items.append(gen[(s, m, c)])
    rnd.shuffle(items)
    rows, key = [], []
    for i, r in enumerate(items, 1):
        pool = db.get_courses_for_competency(r["competency_id"])
        pool_ids = {p["id"] for p in pool}
        path = []
        for j, c in enumerate(r["selected_courses"], 1):
            course = db.get_course(c["id"]) if c["id"] in pool_ids else None
            path.append(f"{j}. {course['name']} — {course['description']}" if course
                        else f"{j}. {c.get('name', '?')} [NÃO EXISTE NO CATÁLOGO]")
        rows.append({"item": f"T{i:03d}", "competencia_objetivo": r["objective"],
                     "trilha_proposta": "\n".join(path),
                     "candidatos_disponiveis": "\n".join(f"- {p['name']}" for p in pool),
                     **{d: None for d in DIMS}, "comentario": None})
        key.append({"item": f"T{i:03d}", "system": r["system"], "gen_model": r["gen_model"],
                    "competency_id": r["competency_id"], "run": 0})
    instr = pd.DataFrame({"Instruções": [
        "Avalie cada trilha (aba 'avaliacao') de forma independente, sem consultar outras pessoas.",
        "Cada linha traz o objetivo da competência, a trilha proposta (em ordem) e os cursos candidatos do catálogo.",
        "Atribua uma nota inteira de 1 a 5 em cada coluna de dimensão, seguindo as âncoras abaixo.",
        "Use a coluna 'comentario' para justificar notas extremas (1 ou 5) ou registrar problemas.",
        "", *RUBRIC.splitlines()]})
    xlsx = OUT / "avaliacao_trilhas.xlsx"
    with pd.ExcelWriter(xlsx) as w:
        instr.to_excel(w, sheet_name="instrucoes", index=False)
        pd.DataFrame(rows).to_excel(w, sheet_name="avaliacao", index=False)
    pd.DataFrame(key).to_csv(OUT / "chave_NAO_ENVIAR.csv", index=False)
    print(f"{len(rows)} items over {len(comps)} competencies -> {xlsx}")


def score():
    from analyze_v2 import kripp_alpha_ordinal, load
    from scipy import stats
    key = pd.read_csv(OUT / "chave_NAO_ENVIAR.csv")
    filled = sorted((OUT / "filled").glob("*.xlsx"))
    if not filled:
        sys.exit(f"no filled workbooks in {OUT / 'filled'}")
    hs = []
    for i, f in enumerate(filled):
        h = pd.read_excel(f, sheet_name="avaliacao")[["item", *DIMS]].dropna()
        h["rater"] = i
        hs.append(h)
    h = pd.concat(hs).merge(key, on="item")
    df = load()
    m = h.merge(df, on=["system", "gen_model", "competency_id", "run"], suffixes=("_h", ""))
    for d in DIMS:
        hm = m.groupby("item")[[f"{d}_h", d, f"{d}_G", f"{d}_C"]].mean()
        print(f"{d:10} rho(human, judges)={stats.spearmanr(hm[f'{d}_h'], hm[d])[0]:.2f}  "
              f"rho(human, G)={stats.spearmanr(hm[f'{d}_h'], hm[f'{d}_G'])[0]:.2f}  "
              f"rho(human, C)={stats.spearmanr(hm[f'{d}_h'], hm[f'{d}_C'])[0]:.2f}")
    if len(filled) >= 2:
        p = h.pivot_table(index="item", columns="rater", values="overall")
        print("inter-curator alpha (overall, raters 0-1):",
              round(kripp_alpha_ordinal(p[0].round(), p[1].round()), 2))
    sysh = m.groupby("system")[["overall_h", "overall"]].mean()
    print(sysh.round(2))


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
