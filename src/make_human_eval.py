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
from rejudge import DIMS

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "human_eval"
SYSTEMS = [("retrieval_topk", "text-embedding-3-small"), ("multi_agent", "gpt-4o-mini"),
           ("hybrid", "gpt-4o-mini"), ("single_agent", "gpt-5.4-mini:high"),
           ("s1top5_kev4b", "kev-4b")]
N_COMP = 16
SEED = 7
# one evaluation sheet per curator, same items in a different random order each
RATERS = {"raymundo": "Raymundo", "fabiane": "Fabiane"}

DIM_PT = {"relevance": "Relevância", "coverage": "Cobertura", "ordering": "Ordem",
          "concision": "Concisão", "overall": "Nota geral"}
ANCHORS = {
    "relevance": ("Os cursos da trilha são pertinentes ao objetivo da competência?",
                  ["5 = todos claramente centrais ao objetivo",
                   "4 = quase todos centrais, no máximo um periférico",
                   "3 = mistura de centrais e periféricos",
                   "2 = maioria periférica, ou algum curso inexistente",
                   "1 = majoritariamente irrelevante ou inexistente"]),
    "coverage": ("Considerando APENAS os cursos candidatos listados, a trilha inclui os necessários?",
                 ["5 = nenhum candidato essencial ficou de fora",
                  "4 = falta um candidato útil, mas não essencial",
                  "3 = falta um candidato essencial",
                  "2 = faltam vários candidatos essenciais",
                  "1 = cobre uma parte mínima da competência"]),
    "ordering": ("A ordem forma uma progressão pedagógica coerente?",
                 ["5 = fundamentos antes de aprofundamentos, sem inversões",
                  "4 = uma inversão leve",
                  "3 = ordem em parte arbitrária (ou trilha com um único curso)",
                  "2 = várias inversões claras",
                  "1 = ordem contrária à progressão"]),
    "concision": ("A trilha evita cursos redundantes, sobrepostos ou desnecessários?",
                  ["5 = nenhum curso supérfluo",
                   "4 = um curso dispensável",
                   "3 = alguns redundantes ou dispensáveis",
                   "2 = muitos supérfluos",
                   "1 = lista quase sem curadoria"]),
    "overall": ("Qualidade geral: você ofereceria esta trilha a um servidor?",
                ["5 = ofereceria como está",
                 "4 = ofereceria com um ajuste pequeno",
                 "3 = precisaria de ajustes moderados",
                 "2 = precisaria ser refeita em grande parte",
                 "1 = inaproveitável"]),
}
DESC_CHARS = 350


def clean(text: str) -> str:
    # the catalog dump renamed the SQL schema "public" everywhere, including inside words
    return (text or "").replace("plaforedu_schema_dev", "public")


def short(text: str, n: int = DESC_CHARS) -> str:
    text = " ".join(clean(text).split())
    return text if len(text) <= n else text[: n].rsplit(" ", 1)[0] + "…"


def build():
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.worksheet.datavalidation import DataValidation

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
    sample = rnd.sample(comps, min(N_COMP, len(comps)))
    items = [gen[(s, m, c)] for c in sample for s, m in SYSTEMS]
    rnd.shuffle(items)
    # example for the instructions: a competency outside the sample
    example = gen[("retrieval_topk", "text-embedding-3-small", sorted(set(comps) - set(sample))[0])]

    F = "Arial"
    H = Font(name=F, bold=True, color="FFFFFF")
    HFILL = PatternFill("solid", fgColor="1F4E78")
    FILL = PatternFill("solid", fgColor="FFF2CC")  # cells the curator fills in
    EXFILL = PatternFill("solid", fgColor="E2EFDA")
    thin = Side(style="thin", color="BFBFBF")
    BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
    WRAP = Alignment(wrap_text=True, vertical="top")
    CENTER = Alignment(horizontal="center", vertical="top")

    def path_text(r):
        pool_ids = {p["id"] for p in db.get_courses_for_competency(r["competency_id"])}
        out = []
        for j, c in enumerate(r["selected_courses"], 1):
            course = db.get_course(c["id"]) if c["id"] in pool_ids else None
            out.append(f"{j}. {course['name']}: {short(course['description'])}" if course
                       else f"{j}. {c.get('name', '?')} [NÃO EXISTE NO CATÁLOGO]")
        return "\n\n".join(out) or "(trilha vazia)"

    def pool_text(r):
        return "\n".join(f"• {p['name']}" for p in db.get_courses_for_competency(r["competency_id"]))

    wb = Workbook()
    ws = wb.active
    ws.title = "instrucoes"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 26
    ws.column_dimensions["B"].width = 110
    row = 1

    def put(a, b="", bold=False, size=10, fill=None, height=None):
        nonlocal row
        ws.cell(row, 1, a).font = Font(name=F, bold=True, size=size)
        c = ws.cell(row, 2, b)
        c.font = Font(name=F, bold=bold, size=size)
        c.alignment = WRAP
        ws.cell(row, 1).alignment = WRAP
        if fill:
            c.fill = fill
            c.border = BOX
        if height:
            ws.row_dimensions[row].height = height
        row += 1

    put("Avaliação de trilhas formativas", "", size=14)
    row += 1
    for tag, name in RATERS.items():
        put(f"Progresso {name}",
            f"=COUNT(aval_{tag}!E2:I{len(items) + 1})&\" de {5 * len(items)} notas preenchidas\"")
    row += 1
    put("Objetivo",
        "Você vai avaliar 80 trilhas formativas, sequências ordenadas de cursos do PlaforEdu propostas "
        "para desenvolver uma competência. As trilhas foram geradas por métodos diferentes, que não são "
        "identificados. Sua avaliação será comparada às notas de avaliadores automáticos.", height=42)
    put("Como fazer",
        "1. Use SOMENTE a aba com o seu nome (" + ", ".join(f"'aval_{t}'" for t in RATERS)
        + "). As abas têm as mesmas trilhas em ordens diferentes.\n"
        "2. Em cada linha, leia em cada linha o objetivo da competência, a trilha proposta (na ordem) "
        "e a lista de cursos candidatos que o catálogo oferecia para essa competência.\n"
        "3. Dê uma nota inteira de 1 a 5 em cada uma das cinco colunas amarelas, seguindo as âncoras abaixo. "
        "Use a lista de opções da célula.\n"
        "4. Use 'Comentário' para justificar notas 1 ou 5 ou registrar qualquer problema (opcional).", height=86)
    put("Regras",
        "• Avalie sozinho(a), sem conversar com o outro avaliador sobre as trilhas e sem abrir a aba dele(a).\n"
        "• Avalie cada trilha independentemente das outras; trilhas da mesma competência aparecem em pontos diferentes.\n"
        "• Julgue cada dimensão separadamente; não repita a mesma nota em todas sem motivo.\n"
        "• A trilha só pode usar os cursos candidatos listados. Um curso marcado [NÃO EXISTE NO CATÁLOGO] é erro grave.\n"
        "• As descrições dos cursos estão resumidas; o nome e o início da descrição bastam para julgar.\n"
        "• Não é preciso fazer tudo de uma vez: as notas ficam salvas e você pode continuar depois. "
        "Tempo estimado: 3 a 5 min por trilha.",
        height=110)
    row += 1
    put("Critérios de avaliação", "", size=12)
    for d in DIMS:
        q, anchors = ANCHORS[d]
        put(DIM_PT[d], q + "\n" + "\n".join(anchors), height=88)
    row += 1
    put("Exemplo (formato)", "Linha ilustrativa de uma competência que NÃO está na avaliação, só para mostrar "
        "como preencher. As notas são um exemplo de formato, não uma referência.", size=10)
    heads = ["Objetivo da competência", "Trilha proposta", *[DIM_PT[d] for d in DIMS], "Comentário"]
    vals = [clean(example["objective"]), path_text(example), 4, 3, 4, 4, 4,
            "Falta o curso introdutório da lista de candidatos."]
    for h, v in zip(heads, vals):
        ws.cell(row, 1, h).font = Font(name=F, bold=True)
        c = ws.cell(row, 2, v)
        c.font = Font(name=F)
        c.fill = EXFILL
        c.alignment = WRAP
        if h == "Trilha proposta":
            ws.row_dimensions[row].height = min(300, 14 * (str(v).count("\n") + 1) + 14 * len(str(v)) // 130)
        row += 1

    def eval_sheet(ev, order):
        cols = [("Item", 7), ("Objetivo da competência", 34), ("Trilha proposta (na ordem)", 70),
                ("Cursos candidatos no catálogo", 38), *[(DIM_PT[d], 11) for d in DIMS], ("Comentário", 30)]
        for j, (h, w) in enumerate(cols, 1):
            c = ev.cell(1, j, h)
            c.font, c.fill, c.alignment, c.border = H, HFILL, Alignment(wrap_text=True, vertical="center",
                                                                        horizontal="center"), BOX
            ev.column_dimensions[c.column_letter].width = w
        ev.row_dimensions[1].height = 32
        dv = DataValidation(type="whole", operator="between", formula1="1", formula2="5", allow_blank=True,
                            showErrorMessage=True, errorTitle="Nota inválida",
                            error="Use uma nota inteira de 1 a 5.")
        dv.promptTitle, dv.prompt, dv.showInputMessage = "Nota", "Inteiro de 1 a 5", True
        ev.add_data_validation(dv)
        dv.add(f"E2:I{len(items) + 1}")
        for rr, (i, r) in enumerate(order, 2):
            path, pool = path_text(r), pool_text(r)
            vals = [f"T{i:03d}", clean(r["objective"]), path, pool, *[None] * 5, None]
            for j, v in enumerate(vals, 1):
                c = ev.cell(rr, j, v)
                c.font = Font(name=F, size=9, bold=(j == 1))
                c.border = BOX
                c.alignment = CENTER if j == 1 or 5 <= j <= 9 else WRAP
                if 5 <= j <= 10:
                    c.fill = FILL
            lines = max(path.count("\n") + len(path) // 95, pool.count("\n") + 1, len(r["objective"]) // 45 + 1)
            ev.row_dimensions[rr].height = min(409, 12 * lines + 8)
        ev.freeze_panes = "B2"
        ev.auto_filter.ref = f"A1:J{len(order) + 1}"

    key = [{"item": f"T{i:03d}", "system": r["system"], "gen_model": r["gen_model"],
            "competency_id": r["competency_id"], "run": 0} for i, r in enumerate(items, 1)]
    for tag in RATERS:
        order = list(enumerate(items, 1))
        random.Random(f"{SEED}-{tag}").shuffle(order)
        eval_sheet(wb.create_sheet(f"aval_{tag}"), order)
    wb.calculation.fullCalcOnLoad = True  # progress formulas have no cached value
    xlsx = OUT / "avaliacao_trilhas.xlsx"
    wb.save(xlsx)
    pd.DataFrame(key).to_csv(OUT / "chave_NAO_ENVIAR.csv", index=False)
    print(f"{len(items)} items over {len(sample)} competencies, sheets for {', '.join(RATERS)} -> {xlsx}")


def score():
    from analyze_v2 import kripp_alpha_ordinal, load
    from scipy import stats
    key = pd.read_csv(OUT / "chave_NAO_ENVIAR.csv")
    filled = sorted((OUT / "filled").glob("*.xlsx"))
    if not filled:
        sys.exit(f"no filled workbooks in {OUT / 'filled'}")
    hs = []
    for f in filled:  # each curator may return the whole workbook; keep the filled rows
        for tag in RATERS:
            h = pd.read_excel(f, sheet_name=f"aval_{tag}")
            h = h.rename(columns={"Item": "item", **{v: k for k, v in DIM_PT.items()}})[["item", *DIMS]].dropna()
            h["rater"] = tag
            hs.append(h)
    h = pd.concat(hs).drop_duplicates(["item", "rater"], keep="last").merge(key, on="item")
    print(h.groupby("rater").size().rename("rated items"))
    df = load()
    m = h.merge(df, on=["system", "gen_model", "competency_id", "run"], suffixes=("_h", ""))
    for d in DIMS:
        hm = m.groupby("item")[[f"{d}_h", d, f"{d}_G", f"{d}_C"]].mean()
        print(f"{d:10} rho(human, judges)={stats.spearmanr(hm[f'{d}_h'], hm[d])[0]:.2f}  "
              f"rho(human, G)={stats.spearmanr(hm[f'{d}_h'], hm[f'{d}_G'])[0]:.2f}  "
              f"rho(human, C)={stats.spearmanr(hm[f'{d}_h'], hm[f'{d}_C'])[0]:.2f}")
    p = h.pivot_table(index="item", columns="rater", values=DIMS).dropna()
    if len(p):
        a, b = RATERS
        for d in DIMS:
            print(f"inter-curator alpha ({d}):",
                  round(kripp_alpha_ordinal(p[(d, a)].round(), p[(d, b)].round()), 2))
    sysh = m.groupby("system")[["overall_h", "overall"]].mean()
    print(sysh.round(2))


if __name__ == "__main__":
    {"build": build, "score": score}[sys.argv[1]]()
