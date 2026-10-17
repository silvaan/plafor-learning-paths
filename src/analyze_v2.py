"""Analysis of the v2 study: all systems x generator models x runs, two v2 judges.

Reads results/gen_v2.jsonl (paths) and results/judgments_v2.jsonl (v2 judge scores),
prints diagnostics, and writes figures, LaTeX tables (tab2_*.tex) and numeric macros
(macros_v2.tex, commands prefixed \\V) into papers/ieee-access/paper/.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "papers" / "ieee-access" / "paper" / "figures"
TABDIR = ROOT / "papers" / "ieee-access" / "paper" / "tables"
EMB = "text-embedding-3-small"
ALL_JUDGES = {"gpt-4o": "G", "claude-opus-5-5": "C"}
# ANALYZE_JUDGES=C restricts the composite score to a subset (e.g. while a judge is incomplete)
import os
JUDGES = {k: v for k, v in ALL_JUDGES.items()
          if v in os.environ.get("ANALYZE_JUDGES", "".join(ALL_JUDGES.values()))}
DIMS = ["relevance", "coverage", "ordering", "concision", "overall"]
RNG = np.random.default_rng(0)

# (system, gen_model) -> (macro stem, table label, family)
VARIANTS = {
    ("naive_all", "-"): ("Naive", "Naive dump", "Baseline"),
    ("retrieval_k3", EMB): ("RetrThree", "Retrieval, $k{=}3$", "Retrieval"),
    ("retrieval_topk", EMB): ("Retr", "Retrieval, $k{=}5$", "Retrieval"),
    ("retrieval_k10", EMB): ("RetrTen", "Retrieval, $k{=}10$", "Retrieval"),
    ("s1_laya", "laya-multilingual"): ("Laya", "Laya, threshold", "System One"),
    ("s1top5_laya", "laya-multilingual"): ("LayaFive", "Laya, top-5", "System One"),
    ("s1_kev4b", "kev-4b"): ("Kev", "Kev-4B, threshold", "System One"),
    ("s1top5_kev4b", "kev-4b"): ("KevFive", "Kev-4B, top-5", "System One"),
    ("single_agent", "gpt-4o-mini"): ("Single", "Single agent", "LLM, gpt-4o-mini"),
    ("single_agent_max5", "gpt-4o-mini"): ("SingleFive", "Single agent, max.\\ 5", "LLM, gpt-4o-mini"),
    ("multi_agent", "gpt-4o-mini"): ("Multi", "Multi-agent (critic loop)", "LLM, gpt-4o-mini"),
    ("multi_agent_guarded", "gpt-4o-mini"): ("Guard", "Multi-agent, guarded", "LLM, gpt-4o-mini"),
    ("hybrid", "gpt-4o-mini"): ("Hybrid", "Hybrid retrieve-generate-verify", "LLM, gpt-4o-mini"),
    ("single_agent", "gpt-5.4-mini:none"): ("SingleNone", "Single agent", "LLM, gpt-5.4-mini, no reasoning"),
    ("multi_agent", "gpt-5.4-mini:none"): ("MultiNone", "Multi-agent (critic loop)", "LLM, gpt-5.4-mini, no reasoning"),
    ("single_agent", "gpt-5.4-mini:high"): ("SingleHigh", "Single agent", "LLM, gpt-5.4-mini, high reasoning"),
    ("multi_agent", "gpt-5.4-mini:high"): ("MultiHigh", "Multi-agent (critic loop)", "LLM, gpt-5.4-mini, high reasoning"),
}
ORDER = list(VARIANTS)
SHORT = {"Naive": "Naive dump", "RetrThree": "Retrieval $k{=}3$", "Retr": "Retrieval $k{=}5$",
         "RetrTen": "Retrieval $k{=}10$", "Laya": "Laya thr.", "LayaFive": "Laya top-5",
         "Kev": "Kev-4B thr.", "KevFive": "Kev-4B top-5", "Single": "Single (4o-mini)",
         "SingleFive": "Single max.\\,5 (4o-mini)", "Multi": "Multi-agent (4o-mini)",
         "Guard": "Guarded (4o-mini)", "Hybrid": "Hybrid (4o-mini)", "SingleNone": "Single (5.4, none)",
         "MultiNone": "Multi-agent (5.4, none)", "SingleHigh": "Single (5.4, high)",
         "MultiHigh": "Multi-agent (5.4, high)"}
STEM = {k: v[0] for k, v in VARIANTS.items()}
BY_STEM = {v[0]: k for k, v in VARIANTS.items()}

# Planned comparisons (stem A, stem B, question), Holm-adjusted as one family.
PLANNED = [
    ("Retr", "Multi", "retrieval vs. proposed multi-agent"),
    ("Single", "Multi", "critic loop, gpt-4o-mini"),
    ("Multi", "Guard", "guardrails on the loop"),
    ("Hybrid", "Retr", "hybrid vs. retrieval"),
    ("Hybrid", "Multi", "hybrid vs. multi-agent"),
    ("Single", "SingleFive", "length cap on the single agent"),
    ("SingleNone", "SingleHigh", "reasoning, single agent"),
    ("MultiNone", "MultiHigh", "reasoning, multi-agent"),
    ("SingleHigh", "MultiHigh", "critic loop under high reasoning"),
    ("SingleNone", "MultiNone", "critic loop without reasoning"),
    ("Kev", "Retr", "System One (Kev) vs. retrieval"),
    ("Laya", "Retr", "System One (Laya) vs. retrieval"),
    ("KevFive", "Retr", "Kev top-5 vs. retrieval at equal length"),
    ("Kev", "SingleNone", "System One vs. System-1 LLM"),
    ("Naive", "Retr", "selection vs. no selection"),
]

# USD per 1M tokens (input, output) at experiment time.
PRICE = {"gpt-4o-mini": (0.15, 0.60), "gpt-5.4-mini": (0.75, 4.50), EMB: (0.02, 0.0),
         "gpt-4o": (2.50, 10.0), "claude-opus-5-5": (4.00, 20.0)}


# --------------------------------------------------------------------------- #
def load() -> pd.DataFrame:
    recs = {}
    for l in (RESULTS / "gen_v2.jsonl").read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            recs[(r["system"], r["gen_model"], r["competency_id"], r.get("run", 0))] = r
    df = pd.DataFrame(recs.values())
    df["run"] = df["run"].fillna(0).astype(int)
    df["variant"] = list(zip(df.system, df.gen_model))
    df = df[df.variant.isin(VARIANTS)].copy()
    df["stem"] = df.variant.map(STEM)

    js = [json.loads(l) for l in (RESULTS / "judgments_v2.jsonl").read_text().splitlines() if l.strip()]
    jd = pd.DataFrame(js)
    jd[["system", "gen_model", "competency_id", "run"]] = pd.DataFrame(jd.key.tolist(), index=jd.index)
    jd = jd.drop_duplicates(subset=["judge", "system", "gen_model", "competency_id", "run"], keep="last")
    for judge, tag in ALL_JUDGES.items():
        sub = jd[jd.judge == judge][["system", "gen_model", "competency_id", "run", *DIMS,
                                     "prompt_tokens", "completion_tokens"]]
        sub = sub.rename(columns={**{d: f"{d}_{tag}" for d in DIMS},
                                  "prompt_tokens": f"jpt_{tag}", "completion_tokens": f"jct_{tag}"})
        df = df.merge(sub, on=["system", "gen_model", "competency_id", "run"], how="left")
    for d in DIMS:
        df[d] = df[[f"{d}_{t}" for t in JUDGES.values()]].mean(axis=1, skipna=False)
    df["gen_cost"] = df.apply(gen_cost, axis=1)
    return df


def gen_cost(r):
    m = r["gen_model"].split(":")[0]
    if m not in PRICE:
        return 0.0
    pin, pout = PRICE[m]
    return (r["prompt_tokens"] * pin + r["completion_tokens"] * pout) / 1e6


def comp_means(df, col="overall"):
    """Per-competency mean over runs: rows = competency, columns = stem."""
    return df.groupby(["competency_id", "stem"])[col].mean().unstack()


def boot_ci(x, n=5000):
    x = np.asarray(x[~np.isnan(x)])
    bs = RNG.choice(x, (n, len(x))).mean(axis=1)
    return np.percentile(bs, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def kripp_alpha_ordinal(a, b, levels=(1, 2, 3, 4, 5)):
    """Krippendorff's alpha, ordinal metric, two coders, no missing values."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    idx = {v: i for i, v in enumerate(levels)}
    k = len(levels)
    o = np.zeros((k, k))
    for x, y in zip(a, b):
        o[idx[x], idx[y]] += 1
        o[idx[y], idx[x]] += 1
    nc = o.sum(axis=1)
    n = nc.sum()
    delta = np.zeros((k, k))
    for c in range(k):
        for d in range(k):
            lo, hi = min(c, d), max(c, d)
            delta[c, d] = (nc[lo:hi + 1].sum() - (nc[lo] + nc[hi]) / 2) ** 2
    do = (o * delta).sum()
    de = (np.outer(nc, nc) * delta).sum() / (n - 1)
    return 1 - do / de if de else float("nan")


def weighted_kappa(a, b, levels=(1, 2, 3, 4, 5)):
    from sklearn.metrics import cohen_kappa_score
    return cohen_kappa_score(a, b, labels=list(levels), weights="quadratic")


def fnum(x, d=2):
    return "--" if pd.isna(x) else f"{x:.{d}f}"


def fp(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def fpr(p):
    """p-value with its relation, for prose: $p\\Vp...$ renders p<0.001 or p=0.042."""
    return "<0.001" if p < 0.001 else f"={p:.3f}"


# --------------------------------------------------------------------------- #
def main():
    df = load()
    judged = df.dropna(subset=["overall"])
    print(f"{len(df)} paths, {len(judged)} judged by both judges, "
          f"{df.competency_id.nunique()} competencies")
    print(judged.groupby("stem").size().reindex([STEM[k] for k in ORDER]))
    M = {}  # macros

    # ---------------- main table ----------------
    cm = comp_means(judged)
    rows = []
    for k in ORDER:
        s = STEM[k]
        g = judged[judged.stem == s]
        if g.empty:
            continue
        lo, hi = boot_ci(cm[s].values)
        row = dict(stem=s, label=VARIANTS[k][1], family=VARIANTS[k][2], runs=g.run.nunique(),
                   ov=cm[s].mean(), lo=lo, hi=hi,
                   ovG=g.overall_G.mean(), ovC=g.overall_C.mean(),
                   rel=g.relevance.mean(), cov=g.coverage.mean(), ord=g.ordering.mean(),
                   con=g.concision.mean(),
                   nsel=g.n_selected.mean(), hall=100 * g.hallucination_rate.mean(),
                   minlen=100 * g.meets_min_courses.mean(),
                   lat=g.latency_s.mean(), cost=g.gen_cost.mean())
        rows.append(row)
        for c in ["ov", "lo", "hi", "ovG", "ovC", "rel", "cov", "ord", "con", "nsel", "hall", "minlen", "lat"]:
            M[f"{c}{s}"] = fnum(row[c], 1 if c in ("hall", "minlen", "nsel") else 2)
        M[f"cost{s}"] = f"{row['cost']:.5f}"
    main_tab = pd.DataFrame(rows)
    print(main_tab.round(2).to_string())

    # ---------------- planned comparisons ----------------
    comp_rows = []
    for a, b, q in PLANNED:
        if a not in cm or b not in cm:
            continue
        m = cm[a].notna() & cm[b].notna()
        x, y = cm[a][m], cm[b][m]
        diff = x - y
        try:
            p = stats.wilcoxon(x, y, zero_method="pratt").pvalue
        except ValueError:
            p = 1.0
        dlo, dhi = boot_ci(diff.values)
        comp_rows.append(dict(a=a, b=b, q=q, n=int(m.sum()), diff=diff.mean(), dlo=dlo, dhi=dhi,
                              win=int((diff > 0).sum()), tie=int((diff == 0).sum()),
                              loss=int((diff < 0).sum()), p=p))
    comp = pd.DataFrame(comp_rows)
    comp["p_holm"] = holm(comp.p)
    print(comp.round(3).to_string())
    for r in comp.itertuples():
        tag = f"{r.a}{r.b}"
        M[f"d{tag}"] = f"{r.diff:+.2f}"
        M[f"dlo{tag}"] = f"{r.dlo:+.2f}"
        M[f"dhi{tag}"] = f"{r.dhi:+.2f}"
        M[f"p{tag}"] = fpr(r.p_holm)
        M[f"wl{tag}"] = f"{r.win}/{r.tie}/{r.loss}"

    # ---------------- mixed model (run-level, random intercept per competency) ----------------
    mm = judged[["overall", "stem", "competency_id"]].copy()
    mm["stem"] = pd.Categorical(mm.stem, categories=["Retr"] + [s for s in cm.columns if s != "Retr"])
    fit = smf.mixedlm("overall ~ C(stem)", mm, groups=mm["competency_id"]).fit(method="lbfgs")
    coef = fit.params.filter(like="C(stem)")
    print("\nMixed model, contrasts vs. Retrieval k=5:\n", pd.DataFrame(
        {"coef": coef, "p": fit.pvalues[coef.index]}).round(3))
    M["mmVarComp"] = fnum(fit.cov_re.iloc[0, 0], 2)
    M["mmVarRes"] = fnum(fit.scale, 2)
    M["mmICC"] = fnum(fit.cov_re.iloc[0, 0] / (fit.cov_re.iloc[0, 0] + fit.scale), 2)
    n_better = sum(1 for i in coef.index if coef[i] > 0 and fit.pvalues[i] < 0.05)
    M["mmNBetterRetr"] = str(n_better)

    # ---------------- judge agreement & diagnostics ----------------
    both = df.dropna(subset=["overall_G", "overall_C"])
    M["nBoth"] = str(len(both))
    ag_rows = []
    for d in DIMS:
        a, b = both[f"{d}_G"].astype(int), both[f"{d}_C"].astype(int)
        ag_rows.append(dict(dim=d, alpha=kripp_alpha_ordinal(a, b), kappa=weighted_kappa(a, b),
                            rho=stats.spearmanr(a, b)[0], exact=100 * (a == b).mean(),
                            meanG=a.mean(), meanC=b.mean()))
    agree = pd.DataFrame(ag_rows)
    print(agree.round(3).to_string())
    for r in agree.itertuples():
        D = r.dim.capitalize()
        M[f"alpha{D}"] = fnum(r.alpha)
        M[f"kappa{D}"] = fnum(r.kappa)
        M[f"rho{D}"] = fnum(r.rho)
        M[f"exact{D}"] = fnum(r.exact, 0)
    # system-level agreement: do the two judges rank the variants the same way?
    sysG = both.groupby("stem").overall_G.mean()
    sysC = both.groupby("stem").overall_C.mean()
    M["rhoSysJudges"] = fnum(stats.spearmanr(sysG, sysC)[0])
    M["tauSysJudges"] = fnum(stats.kendalltau(sysG, sysC)[0])
    # dimension structure (collapse check) for each judge
    for tag in ALL_JUDGES.values():
        X = df.dropna(subset=[f"overall_{tag}"])[[f"{d}_{tag}" for d in DIMS]]
        M[f"pctOvEqCov{tag}"] = fnum(100 * (X[f"overall_{tag}"] == X[f"coverage_{tag}"]).mean(), 0)
        M[f"pctThree{tag}"] = fnum(100 * (X[f"overall_{tag}"] == 3).mean(), 0)
        M[f"rhoRelOrd{tag}"] = fnum(stats.spearmanr(X[f"relevance_{tag}"], X[f"ordering_{tag}"])[0])
        M[f"rhoCovCon{tag}"] = fnum(stats.spearmanr(X[f"coverage_{tag}"], X[f"concision_{tag}"])[0])
        sel = df.dropna(subset=[f"overall_{tag}"])
        sel = sel[sel.system != "naive_all"]
        M[f"rhoRatio{tag}"] = fnum(stats.spearmanr(sel.selection_ratio, sel[f"overall_{tag}"])[0])
        M[f"rhoRatioCov{tag}"] = fnum(stats.spearmanr(sel.selection_ratio, sel[f"coverage_{tag}"])[0])
        M[f"rhoRatioCon{tag}"] = fnum(stats.spearmanr(sel.selection_ratio, sel[f"concision_{tag}"])[0])
    # v1 vs v2 on the original run-0 paths
    v1 = judged[judged.v1_judge.notna()].copy()
    if len(v1):
        v1["ov_v1"] = v1.v1_judge.map(lambda d: d["overall"])
        M["rhoVOneVTwo"] = fnum(stats.spearmanr(v1.ov_v1, v1.overall)[0])
        M["rhoVOneG"] = fnum(stats.spearmanr(v1.ov_v1, v1.overall_G)[0])
        M["nVOne"] = str(len(v1))
    # self-preference probe: gap between judges for OpenAI-generated vs. other paths
    openai_gen = both.gen_model.str.startswith("gpt")
    gap = both.overall_G - both.overall_C
    M["gapOpenAI"] = fnum(gap[openai_gen].mean())
    M["gapOther"] = fnum(gap[~openai_gen].mean())
    M["pGap"] = fpr(stats.mannwhitneyu(gap[openai_gen], gap[~openai_gen]).pvalue)

    # ---------------- run-to-run variability ----------------
    var_rows = []
    for s in ["Single", "Multi", "Guard", "Hybrid", "SingleNone", "MultiNone", "SingleHigh", "MultiHigh"]:
        g = df[df.stem == s]
        if g.run.nunique() < 2:
            continue
        paths = g.groupby("competency_id").selected_courses.apply(
            lambda xs: len({tuple(c["id"] for c in x) for x in xs}))
        sets = g.groupby("competency_id").selected_courses.apply(
            lambda xs: len({frozenset(c["id"] for c in x) for x in xs}))
        sd = judged[judged.stem == s].groupby("competency_id").overall.std().mean()
        var_rows.append(dict(stem=s, identical=100 * (paths == 1).mean(),
                             sameset=100 * (sets == 1).mean(), sd=sd))
        M[f"ident{s}"] = fnum(100 * (paths == 1).mean(), 0)
        M[f"sameset{s}"] = fnum(100 * (sets == 1).mean(), 0)
        M[f"sd{s}"] = fnum(sd)
    print(pd.DataFrame(var_rows).round(2).to_string())

    # ---------------- critic-loop activity ----------------
    for s in ["Multi", "Guard", "MultiNone", "MultiHigh"]:
        g = df[df.stem == s]
        if g.empty:
            continue
        M[f"fired{s}"] = fnum(100 * (g.iterations > 1).mean(), 1)
        M[f"iters{s}"] = fnum(g.iterations.mean())
        M[f"ruim{s}"] = str(int((g.evaluation == "Ruim").sum()))
        M[f"firedN{s}"] = str(int((g.iterations > 1).sum()))
        M[f"firedSmall{s}"] = str(int(((g.iterations > 1) & (g.n_candidates < 3)).sum()))
        M[f"runsN{s}"] = str(len(g))
    g = df[df.stem == "Guard"]
    M["gateFixesGuard"] = str(int(g.gate_fixes.fillna(0).sum()))
    M["gateFixesHybrid"] = str(int(df[df.stem == "Hybrid"].gate_fixes.fillna(0).sum()))

    # ---------------- hallucination breakdown ----------------
    hal = hallucination_table(df)
    for r in hal.itertuples():
        M[f"halCorr{r.stem}"] = str(r.corrupted)
        M[f"halFab{r.stem}"] = str(r.fabricated)
        M[f"halPaths{r.stem}"] = str(r.paths)

    # ---------------- System One diagnostics ----------------
    s1_diag(df, M)

    # ---------------- length control ----------------
    sel = judged[(judged.system != "naive_all") & (judged.n_selected > 0)]
    ols = smf.ols("overall ~ C(stem) + np.log(n_selected)", sel).fit()
    M["betaLogLen"] = fnum(ols.params["np.log(n_selected)"])
    M["pLogLen"] = fpr(ols.pvalues["np.log(n_selected)"])

    # ---------------- pool-size strata ----------------
    judged = judged.copy()
    judged["bucket"] = pd.cut(judged.n_candidates, [0, 2, 5, 10, 1e9], labels=["1-2", "3-5", "6-10", ">10"])
    strata = judged.groupby(["bucket", "stem"], observed=True).overall.mean().unstack()
    print("\nBy pool size:\n", strata.round(2).to_string())
    for b, tag in zip(["1-2", "3-5", "6-10", ">10"], ["A", "B", "C", "D"]):
        M[f"nBucket{tag}"] = str(judged[judged.bucket == b].competency_id.nunique())
        for s in strata.columns:
            M[f"ov{s}Bucket{tag}"] = fnum(strata.loc[b, s]) if b in strata.index else "--"

    # ---------------- costs ----------------
    jcost = 0.0
    for judge, tag in JUDGES.items():
        pin, pout = PRICE[judge]
        c = (df[f"jpt_{tag}"].fillna(0) * pin + df[f"jct_{tag}"].fillna(0) * pout).sum() / 1e6
        M[f"judgeCost{tag}"] = fnum(c, 2)
        jcost += c
    M["judgeCostTotal"] = fnum(jcost, 2)
    M["genCostTotal"] = fnum(df.gen_cost.sum(), 2)
    M["nPaths"] = str(len(df))
    M["nJudged"] = str(len(judged))
    M["nJudgments"] = str(2 * len(judged))
    M["nVariants"] = str(df.stem.nunique())
    M["nCompVTwo"] = str(df.competency_id.nunique())

    write_tables(main_tab, comp, agree, hal)
    make_figures(judged, main_tab, cm)
    with open(TABDIR / "macros_v2.tex", "w") as f:
        f.write("% Auto-generated by src/analyze_v2.py -- do not edit.\n")
        for k, v in sorted(M.items()):
            f.write(f"\\newcommand{{\\V{k}}}{{{v}}}\n")
    print(f"\n{len(M)} macros -> {TABDIR / 'macros_v2.tex'}")


def hallucination_table(df):
    from db import PlaforEduDB
    db = PlaforEduDB()
    names = {c["name"].strip().lower() for c in db.get_courses()}
    rows = []
    for k in ORDER:
        s = STEM[k]
        g = df[df.stem == s]
        if g.empty:
            continue
        corr = fab = paths = 0
        for r in g.itertuples():
            pool = {c["id"] for c in db.get_courses_for_competency(r.competency_id)}
            bad = [c for c in r.selected_courses if c["id"] not in pool]
            paths += bool(bad)
            for c in bad:
                if c.get("name", "").strip().lower() in names:
                    corr += 1
                else:
                    fab += 1
        rows.append(dict(stem=s, label=VARIANTS[k][1], family=VARIANTS[k][2],
                         n=len(g), paths=paths, corrupted=corr, fabricated=fab))
    hal = pd.DataFrame(rows)
    print(hal.to_string())
    return hal


def s1_diag(df, M):
    """How well do System One relevance probabilities predict what a high-reasoning
    LLM selected? (AUC over every candidate course of every competency)."""
    from sklearn.metrics import roc_auc_score
    if "s1_scores" not in df:
        return
    ref = df[(df.stem == "SingleHigh") & (df.run == 0)].set_index("competency_id")
    for s in ["Laya", "Kev"]:
        g = df[df.stem == s].set_index("competency_id")
        y, p, sel_rate = [], [], []
        for cid, r in g.iterrows():
            if cid not in ref.index or not isinstance(r.s1_scores, list):
                continue
            chosen = {c["id"] for c in ref.loc[cid, "selected_courses"]}
            for x in r.s1_scores:
                y.append(x["id"] in chosen)
                p.append(x["p_relevant"])
        if len(set(y)) == 2:
            M[f"auc{s}"] = fnum(roc_auc_score(y, p))
        allp = [x["p_relevant"] for v in g.s1_scores if isinstance(v, list) for x in v]
        M[f"pAbove{s}"] = fnum(100 * np.mean(np.array(allp) >= 0.5), 0)
        M[f"medP{s}"] = fnum(np.median(allp))


# --------------------------------------------------------------------------- #
def write_tables(main_tab, comp, agree, hal):
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Main results over all competencies. Overall is the mean of the two v2 judges "
         r"(gpt-4o, G; claude-opus-5-5, C), averaged over runs per competency, with a bootstrap 95\% CI. "
         r"Rel., Cov., Ord.\ and Conc.\ are the per-dimension means of both judges. $m$ is the mean path length, "
         r"Inv.\ the invalid-identifier rate over all runs, and Cost the generation cost per path in USD.}",
         r"\label{tab:main2}", r"\setlength{\tabcolsep}{3pt}", r"\footnotesize",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{llccccccccccr}", r"\toprule",
         r"Family & System & Runs & Overall [95\% CI] & G & C & Rel. & Cov. & Ord. & Conc. & $m$ & Inv.\,\% & Cost \\",
         r"\midrule"]
    prev = None
    for r in main_tab.itertuples():
        if prev is not None and r.family != prev:
            L.append(r"\midrule")
        fam = r.family if r.family != prev else ""
        prev = r.family
        L.append(f"{fam} & {r.label} & {r.runs} & {r.ov:.2f} [{r.lo:.2f}, {r.hi:.2f}] & {r.ovG:.2f} & {r.ovC:.2f} & "
                 f"{r.rel:.2f} & {r.cov:.2f} & {r.ord:.2f} & {r.con:.2f} & {r.nsel:.1f} & {r.hall:.1f} & "
                 f"{r.cost:.5f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    (TABDIR / "tab2_main.tex").write_text("\n".join(L) + "\n")

    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Planned paired comparisons on the per-competency mean overall score (both judges, runs averaged). "
         r"$\Delta$ is A minus B with a bootstrap 95\% CI; W/T/L counts competencies where A wins, ties or loses; "
         r"$p_{\mathrm{Holm}}$ is the Wilcoxon signed-rank $p$-value, Holm-adjusted over all comparisons.}",
         r"\label{tab:pairwise2}", r"\footnotesize",
         r"\begin{tabular}{lllcccr}", r"\toprule",
         r"Question & A & B & $\Delta$ [95\% CI] & W/T/L & $n$ & $p_{\mathrm{Holm}}$ \\", r"\midrule"]
    lab = {VARIANTS[k][0]: (VARIANTS[k][1] + ("" if VARIANTS[k][2] in ("Baseline", "Retrieval", "System One")
                                               else f" ({VARIANTS[k][2].replace('LLM, ', '')})"))
           for k in VARIANTS}
    for r in comp.itertuples():
        L.append(f"{r.q} & {SHORT[r.a]} & {SHORT[r.b]} & {r.diff:+.2f} [{r.dlo:+.2f}, {r.dhi:+.2f}] & "
                 f"{r.win}/{r.tie}/{r.loss} & {r.n} & {fp(r.p_holm)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (TABDIR / "tab2_pairwise.tex").write_text("\n".join(L) + "\n")

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Agreement between the two v2 judges over all judged paths: Krippendorff's ordinal $\alpha$, "
         r"quadratic-weighted $\kappa$, Spearman $\rho$, exact agreement, and each judge's mean.}",
         r"\label{tab:agree}", r"\footnotesize", r"\begin{tabular}{lcccccc}", r"\toprule",
         r"Dimension & $\alpha$ & $\kappa_w$ & $\rho$ & Exact\,\% & G & C \\", r"\midrule"]
    for r in agree.itertuples():
        L.append(f"{r.dim.capitalize()} & {r.alpha:.2f} & {r.kappa:.2f} & {r.rho:.2f} & {r.exact:.0f} & "
                 f"{r.meanG:.2f} & {r.meanC:.2f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABDIR / "tab2_agree.tex").write_text("\n".join(L) + "\n")

    L = [r"\begin{table}[t]", r"\centering",
         r"\caption{Invalid identifiers over all generated paths (every run). Corrupted: the name exists in the catalog "
         r"but the identifier is not in $A_k$. Fabricated: the course does not exist. Systems not listed produced none.}",
         r"\label{tab:halluc2}", r"\footnotesize", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"System & Paths & Affected & Corr. & Fabr. \\", r"\midrule"]
    for r in hal.itertuples():
        if r.corrupted + r.fabricated == 0:
            continue
        L.append(f"{SHORT[r.stem]} & {r.n} & {r.paths} & {r.corrupted} & {r.fabricated} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABDIR / "tab2_halluc.tex").write_text("\n".join(L) + "\n")


def plain(label):
    return label.replace("$k{=}", "k=").replace("$", "").replace("max.\\ ", "max ")


def make_figures(judged, main_tab, cm):
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fam_color = {"Baseline": "#8c8c8c", "Retrieval": "#2a6fb0", "System One": "#c45a1a",
                 "LLM, gpt-4o-mini": "#3b8f5a", "LLM, gpt-5.4-mini, no reasoning": "#8a5fb5",
                 "LLM, gpt-5.4-mini, high reasoning": "#b5487a"}

    # Forest plot of overall with CIs
    t = main_tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(3.45, 4.2))
    y = np.arange(len(t))
    for yi, r in zip(y, t.itertuples()):
        ax.errorbar(r.ov, yi, xerr=[[r.ov - r.lo], [r.hi - r.ov]], fmt="o", ms=3.5,
                    color=fam_color[r.family], capsize=2, lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([plain(r.label)
                        + ("" if r.family in ("Baseline", "Retrieval", "System One")
                           else {"LLM, gpt-4o-mini": " [4o-mini]", "LLM, gpt-5.4-mini, no reasoning": " [5.4-mini, none]",
                                 "LLM, gpt-5.4-mini, high reasoning": " [5.4-mini, high]"}[r.family])
                        for r in t.itertuples()], fontsize=6.5)
    retr = main_tab.set_index("stem").loc["Retr", "ov"]
    ax.axvline(retr, color="#2a6fb0", ls=":", lw=0.8)
    ax.set_xlabel("Overall score (1-5)")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_forest.png", dpi=300)
    plt.close(fig)

    # Quality vs latency (log)
    off = {"SingleNone": (-34, 5), "MultiNone": (3, 5), "Single": (3, -8), "Multi": (3, 3),
           "Guard": (-24, 4), "Hybrid": (-30, -9), "Laya": (-20, 5), "LayaFive": (5, -7),
           "Retr": (3, 3), "RetrThree": (-38, 3), "RetrTen": (3, -8)}
    fig, ax = plt.subplots(figsize=(3.45, 2.7))
    for r in main_tab.itertuples():
        x = max(r.lat, 1e-3)
        ax.scatter(x, r.ov, color=fam_color[r.family], s=14, zorder=3)
        ax.annotate(r.stem, (x, r.ov), fontsize=5.2, xytext=off.get(r.stem, (3, 2)),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("Latency per path (s, log scale)")
    ax.set_ylabel("Overall score")
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=f.replace("LLM, ", ""), ms=4)
               for f, c in fam_color.items()]
    ax.legend(handles=handles, fontsize=5.2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_pareto.png", dpi=300)
    plt.close(fig)

    # Length vs score per judge
    fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.0), sharey=True)
    sel = judged[judged.system != "naive_all"]
    for ax, (judge, tag) in zip(axes, JUDGES.items()):
        b = pd.cut(sel.selection_ratio, [0, .2, .4, .6, .8, 1.0001])
        mcov = sel.groupby(b, observed=True)[f"coverage_{tag}"].mean()
        mcon = sel.groupby(b, observed=True)[f"concision_{tag}"].mean()
        mov = sel.groupby(b, observed=True)[f"overall_{tag}"].mean()
        xs = np.arange(len(mov))
        ax.plot(xs, mcov.values, "-o", ms=2.5, label="coverage", color="#2a6fb0")
        ax.plot(xs, mcon.values, "-s", ms=2.5, label="concision", color="#c45a1a")
        ax.plot(xs, mov.values, "-^", ms=2.5, label="overall", color="#333333")
        ax.set_xticks(xs)
        ax.set_xticklabels(["<.2", ".2-.4", ".4-.6", ".6-.8", ">.8"][: len(xs)], fontsize=6)
        ax.set_title(judge, fontsize=7)
        ax.set_xlabel("$m/|A_k|$", fontsize=7)
    axes[0].set_ylabel("Mean score")
    axes[0].legend(fontsize=5.5, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_length.png", dpi=300)
    plt.close(fig)

    # Pool-size strata for selected configurations
    sel_stems = [("Naive", "Naive dump"), ("Retr", "Retrieval k=5"), ("KevFive", "Kev-4B top-5"),
                 ("Multi", "Multi-agent [4o-mini]"), ("SingleHigh", "Single agent [5.4-mini, high]")]
    b = pd.cut(judged.n_candidates, [0, 2, 5, 10, 1e9], labels=["1-2", "3-5", "6-10", ">10"])
    st = judged.assign(bucket=b).groupby(["bucket", "stem"], observed=True).overall.mean().unstack()
    ncomp = judged.assign(bucket=b).groupby("bucket", observed=True).competency_id.nunique()
    fig, ax = plt.subplots(figsize=(3.45, 2.3))
    w = 0.16
    xs = np.arange(len(st.index))
    pal = ["#8c8c8c", "#2a6fb0", "#c45a1a", "#3b8f5a", "#b5487a"]
    for i, ((stem, lab), c) in enumerate(zip(sel_stems, pal)):
        if stem in st:
            ax.bar(xs + (i - 2) * w, st[stem].values, w, label=lab, color=c)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{i}\n(n={ncomp[i]})" for i in st.index], fontsize=6.5)
    ax.set_xlabel("Candidate courses per competency", fontsize=7)
    ax.set_ylabel("Overall score")
    ax.set_ylim(1, 4.2)
    ax.legend(fontsize=5.3, frameon=False, ncol=2, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_bysize.png", dpi=300)
    plt.close(fig)

    # Reasoning x architecture interaction
    fig, ax = plt.subplots(figsize=(3.45, 2.2))
    for arch, stems, mk in [("Single agent", ["Single", "SingleNone", "SingleHigh"], "o"),
                            ("Multi-agent", ["Multi", "MultiNone", "MultiHigh"], "s")]:
        vals = [main_tab.set_index("stem").ov.get(s, np.nan) for s in stems]
        ax.plot([0, 1, 2], vals, marker=mk, label=arch, ms=4)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["gpt-4o-mini", "gpt-5.4-mini\nno reasoning", "gpt-5.4-mini\nhigh reasoning"], fontsize=6.5)
    ax.axhline(main_tab.set_index("stem").ov["Retr"], color="#2a6fb0", ls=":", lw=0.8, label="Retrieval k=5")
    ax.set_ylabel("Overall score")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig2_reasoning.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
