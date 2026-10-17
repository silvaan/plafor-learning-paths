"""Aggregate experiment results, run statistical tests, and emit figures + LaTeX tables."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGDIR = ROOT / "papers" / "ieee-access" / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)
TABDIR = ROOT / "papers" / "ieee-access" / "paper" / "tables"
TABDIR.mkdir(parents=True, exist_ok=True)

SYS_ORDER = ["naive_all", "retrieval_topk", "single_agent", "multi_agent"]
SYS_LABEL = {
    "naive_all": "Naive full-catalog",
    "retrieval_topk": "Embedding retrieval",
    "single_agent": "Single-agent, no critic",
    "multi_agent": "Multi-agent, proposed",
}

# Pricing (USD per 1M tokens), as of experiment date.
PRICE = {
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.0),
}


def load(complete_only: bool = True) -> pd.DataFrame:
    rows = [json.loads(l) for l in (RESULTS / "raw_results.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    # de-duplicate on (system, competency, run) keeping last
    df = df.drop_duplicates(subset=["system", "competency_id", "run"], keep="last")
    if complete_only:
        # keep only competencies evaluated by every system (balanced comparison)
        n_sys = df["system"].nunique()
        counts = df.groupby("competency_id")["system"].nunique()
        keep = counts[counts == n_sys].index
        df = df[df["competency_id"].isin(keep)]
    return df


def gen_cost(row):
    # generation model is gpt-4o-mini for LLM systems, embeddings tiny for retrieval
    pin, pout = PRICE["gpt-4o-mini"]
    return (row["prompt_tokens"] * pin + row["completion_tokens"] * pout) / 1e6


def judge_cost(row):
    pin, pout = PRICE["gpt-4o"]
    return (row["judge_prompt_tokens"] * pin + row["judge_completion_tokens"] * pout) / 1e6


def main():
    df = load()
    df["gen_cost"] = df.apply(gen_cost, axis=1)
    df["judge_cost"] = df.apply(judge_cost, axis=1)
    n_comp = df["competency_id"].nunique()
    print(f"{len(df)} records over {n_comp} competencies, systems: {sorted(df.system.unique())}")

    metrics = ["relevance", "ordering", "coverage", "overall",
               "meets_min_courses", "hallucination_rate", "n_selected",
               "selection_ratio", "iterations", "latency_s", "gen_cost"]

    # ---- Main aggregate table ----
    agg = df.groupby("system")[metrics].agg(["mean", "std"])
    print("\n=== Aggregate (mean) ===")
    print(df.groupby("system")[metrics].mean().reindex(SYS_ORDER).round(3).T)

    # ---- Paired statistical tests vs multi_agent ----
    print("\n=== Wilcoxon signed-rank vs multi_agent (overall score) ===")
    pivot = df.pivot_table(index="competency_id", columns="system", values="overall")
    stat_rows = []
    for s in SYS_ORDER:
        if s == "multi_agent":
            continue
        a = pivot["multi_agent"]
        b = pivot[s]
        mask = a.notna() & b.notna()
        try:
            w, p = stats.wilcoxon(a[mask], b[mask])
        except ValueError:
            w, p = float("nan"), float("nan")
        d = (a[mask] - b[mask]).mean()
        stat_rows.append((s, d, p))
        print(f"  multi_agent vs {s:15}: mean_diff={d:+.3f}  p={p:.4g}")

    # ---- Refinement-loop analysis ----
    ma = df[df.system == "multi_agent"]
    print("\n=== Multi-agent refinement ===")
    print(f"  competencies with >1 iteration: {(ma.iterations > 1).sum()}/{len(ma)}")
    print(f"  mean iterations: {ma.iterations.mean():.2f}")
    print(f"  final 'Ruim' (unresolved): {(ma.evaluation=='Ruim').sum()}")

    # ---- Stratify by catalog size ----
    df["catalog_bucket"] = pd.cut(df["n_candidates"], [0, 2, 5, 10, 1e9],
                                  labels=["1-2", "3-5", "6-10", ">10"])
    print("\n=== Overall score by catalog size ===")
    print(df.pivot_table(index="catalog_bucket", columns="system",
                         values="overall", aggfunc="mean").reindex(columns=SYS_ORDER).round(2))

    # ---- Cost summary ----
    print("\n=== Cost / latency per path ===")
    print(df.groupby("system")[["gen_cost", "latency_s"]].mean().reindex(SYS_ORDER).round(4))
    print(f"\nTotal experiment gen cost: ${df.gen_cost.sum():.3f}, judge cost: ${df.judge_cost.sum():.3f}")

    extra = extra_stats(df, pivot, ma)
    make_figures(df, pivot)
    write_tables(df, stat_rows, ma, n_comp, extra)
    write_macros(df, stat_rows, ma, n_comp, extra)
    print("\nFigures ->", FIGDIR, "\nTables ->", TABDIR)


def write_macros(df, stat_rows, ma, n_comp, extra):
    """Emit \\newcommand macros with key numbers used in prose."""
    g = df.groupby("system")
    mean = lambda s, c: g.get_group(s)[c].mean()
    pmap = {s: p for s, _, p in stat_rows}
    small = df[df.n_candidates < 3]
    ma_small = small[small.system == "multi_agent"]
    ma_large = ma[ma.n_candidates >= 3]

    def fnum(x, d=2):
        return f"{x:.{d}f}"

    ma_big = ma[ma.n_candidates >= 6]
    macros = {
        "nComp": str(n_comp),
        "nCompCand": "68",
        "nCompTotal": "69",
        "nCourses": "442",
        "nLinks": "1002",
        "ovMultiBig": f"{ma_big.overall.mean():.2f}" if len(ma_big) else "--",
        "nLargePool": str(int((ma.n_candidates >= 6).sum())),
        "ovRetrBig": fnum(df[(df.system=='retrieval_topk') & (df.n_candidates>=6)].overall.mean()),
        "diffRetr": fnum(mean("retrieval_topk", "overall") - mean("multi_agent", "overall")),
        "pRetr": f"{pmap.get('retrieval_topk', float('nan')):.3g}",
        "nHallComp": str(int((ma.hallucination_rate > 0).sum())),
        "nSelNaive": fnum(mean("naive_all", "n_selected"), 1),
        "nSelRetr": fnum(mean("retrieval_topk", "n_selected"), 1),
        "nSelMulti": fnum(mean("multi_agent", "n_selected"), 1),
        "relRetr": fnum(mean("retrieval_topk", "relevance")),
        "covRetr": fnum(mean("retrieval_topk", "coverage")),
        "covMulti": fnum(mean("multi_agent", "coverage")),
        "judgeCost": fnum(df["judge_cost"].sum(), 2),
        "genCostTotal": fnum(df["gen_cost"].sum(), 2),
        "medCand": str(int(df[df.system=='multi_agent'].n_candidates.median())),
        "maxCand": str(int(df[df.system=='multi_agent'].n_candidates.max())),
        # overall scores
        "ovNaive": fnum(mean("naive_all", "overall")),
        "ovRetr": fnum(mean("retrieval_topk", "overall")),
        "ovSingle": fnum(mean("single_agent", "overall")),
        "ovMulti": fnum(mean("multi_agent", "overall")),
        "relMulti": fnum(mean("multi_agent", "relevance")),
        "ordMulti": fnum(mean("multi_agent", "ordering")),
        "ordNaive": fnum(mean("naive_all", "ordering")),
        "ordSingle": fnum(mean("single_agent", "ordering")),
        "ordRetr": fnum(mean("retrieval_topk", "ordering")),
        "covNaive": fnum(mean("naive_all", "coverage")),
        # constraint / hallucination
        "minNaive": fnum(100*mean("naive_all", "meets_min_courses"), 0),
        "minSingle": fnum(100*mean("single_agent", "meets_min_courses"), 0),
        "minMulti": fnum(100*mean("multi_agent", "meets_min_courses"), 0),
        "hallMulti": fnum(100*mean("multi_agent", "hallucination_rate"), 1),
        "hallSingle": fnum(100*mean("single_agent", "hallucination_rate"), 1),
        # refinement
        "refinePct": fnum(100*(ma.iterations > 1).mean(), 0),
        "meanIters": fnum(ma.iterations.mean()),
        "nSmall": str(int((df[df.system=='multi_agent'].n_candidates < 3).sum())),
        "hallSmall": fnum(100*ma_small.hallucination_rate.mean(), 1) if len(ma_small) else "0.0",
        "hallLarge": fnum(100*ma_large.hallucination_rate.mean(), 1) if len(ma_large) else "0.0",
        # efficiency
        "keepNaive": fnum(100*mean("naive_all", "selection_ratio"), 0),
        "keepMulti": fnum(100*mean("multi_agent", "selection_ratio"), 0),
        "latMulti": fnum(mean("multi_agent", "latency_s"), 1),
        "costMulti": fnum(mean("multi_agent", "gen_cost"), 5),
        # p-values
        "pSingle": f"{pmap.get('single_agent', float('nan')):.3g}",
        "pNaive": f"{pmap.get('naive_all', float('nan')):.3g}",
        # ordering gain single->multi on large catalogs
        "ordSingleLarge": fnum(df[(df.system=='single_agent')&(df.n_candidates>=3)].ordering.mean()),
        "ordMultiLarge": fnum(df[(df.system=='multi_agent')&(df.n_candidates>=3)].ordering.mean()),
    }
    macros.update(extra["macros"])
    lines = [f"\\newcommand{{\\R{k}}}{{{v}}}" for k, v in macros.items()]
    (TABDIR / "macros.tex").write_text("\n".join(lines) + "\n")


def make_figures(df, pivot):
    # Fig 1: candidate-course distribution over the FULL catalog (all competencies),
    # not just the evaluated subset, since it motivates the task at large scale.
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from db import PlaforEduDB
    dbx = PlaforEduDB()
    ncand = [c["n_candidates"] for c in dbx.competencies_with_courses(min_courses=1)]
    dbx.close()
    ncand = np.array(ncand)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(ncand, bins=25, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Number of candidate courses per competency")
    ax.set_ylabel("Competencies")
    ax.axvline(np.median(ncand), color="#C44E52", ls="--",
               label=f"median {np.median(ncand):.0f}, max {ncand.max()}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_courses_per_comp.png", dpi=200)
    plt.close(fig)

    # Fig 2: mean rubric scores per system, grouped bars with 95% CI
    dims = ["relevance", "ordering", "coverage", "overall"]
    means = df.groupby("system")[dims].mean().reindex(SYS_ORDER)
    sems = df.groupby("system")[dims].sem().reindex(SYS_ORDER) * 1.96
    x = np.arange(len(dims))
    w = 0.2
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    colors = ["#B0B0B0", "#8C8C8C", "#DD8452", "#4C72B0"]
    for i, s in enumerate(SYS_ORDER):
        ax.bar(x + (i - 1.5) * w, means.loc[s], w, yerr=sems.loc[s],
               capsize=2, label=SYS_LABEL[s], color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(["Relevance", "Ordering", "Coverage", "Overall"])
    ax.set_ylabel("LLM-judge score, 1 to 5")
    ax.set_ylim(1, 5)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_scores.png", dpi=200)
    plt.close(fig)

    # Fig 3: overall score by catalog size
    piv = df.pivot_table(index="catalog_bucket", columns="system",
                         values="overall", aggfunc="mean").reindex(columns=SYS_ORDER)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    piv.plot(kind="bar", ax=ax, color=colors, width=0.8)
    nb = df[df.system == "multi_agent"]["catalog_bucket"].value_counts()
    ax.set_xticklabels([f"{b}\n(n={nb.get(b, 0)})" for b in piv.index])
    ax.set_xlabel("Candidate courses per competency")
    ax.set_ylabel("Mean overall score, 1 to 5")
    ax.set_ylim(1, 5)
    ax.legend([SYS_LABEL[s] for s in SYS_ORDER], fontsize=8)
    plt.xticks(rotation=0)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_scores_by_size.png", dpi=200)
    plt.close(fig)


def write_tables(df, stat_rows, ma, n_comp, extra):
    dims = ["relevance", "ordering", "coverage", "overall"]
    m = df.groupby("system")[dims + ["meets_min_courses", "hallucination_rate",
                                     "n_selected", "selection_ratio"]].mean().reindex(SYS_ORDER)
    pmap = {s: p for s, _, p in stat_rows}
    dmap = {s: d for s, d, _ in stat_rows}

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Learning-path quality across systems over " + str(n_comp) +
        r" competencies. Rubric scores are LLM-judge means on a 1 to 5 scale. The columns "
        r"$\Delta$ and $p$ compare each system against the proposed multi-agent system with a "
        r"Wilcoxon signed-rank test on the overall score. Best value per column in bold.}",
        r"\label{tab:main}",
        r"\begin{tabular}{lcccccccc}",
        r"\hline",
        r"System & Relevance & Ordering & Coverage & Overall [95\% CI] & $\Delta$Overall & $p$ & "
        r"Min-3, \% & Halluc., \% \\",
        r"\hline",
    ]
    best = {d: m[d].max() for d in dims}
    for s in SYS_ORDER:
        r = m.loc[s]
        cells = []
        for d in dims:
            v = f"{r[d]:.2f}"
            if abs(r[d] - best[d]) < 1e-9:
                v = r"\textbf{" + v + "}"
            if d == "overall":
                lo, hi = extra["ci"][s]
                v += f" [{lo:.2f}, {hi:.2f}]"
            cells.append(v)
        if s == "multi_agent":
            dcell, pcell = "--", "--"
        else:
            dcell = f"{dmap[s]:+.2f}"
            pcell = f"{pmap[s]:.3g}" if pmap[s] == pmap[s] else "--"
        lines.append(
            f"{SYS_LABEL[s]} & " + " & ".join(cells) +
            f" & {dcell} & {pcell} & {100*r['meets_min_courses']:.1f} & {100*r['hallucination_rate']:.1f} \\\\"
        )
    lines += [r"\hline", r"\end{tabular}", r"\end{table*}"]
    (TABDIR / "tab_main.tex").write_text("\n".join(lines))
    (TABDIR / "tab_pairwise.tex").write_text(extra["pairwise_tex"])
    (TABDIR / "tab_halluc.tex").write_text(extra["halluc_tex"])

    # efficiency table
    eff = df.groupby("system")[["n_selected", "selection_ratio", "iterations",
                                "latency_s", "gen_cost"]].mean().reindex(SYS_ORDER)
    el = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Efficiency across systems. Columns report the mean path length, fraction of "
        r"candidates kept, number of refinement iterations, wall-clock latency in seconds and "
        r"generation cost in US dollars per learning path.}",
        r"\label{tab:efficiency}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{lccccc}", r"\hline",
        r"System & \#Courses & Kept \% & Iters & Latency & Cost \\", r"\hline",
    ]
    for s in SYS_ORDER:
        r = eff.loc[s]
        el.append(f"{SYS_LABEL[s]} & {r['n_selected']:.1f} & {100*r['selection_ratio']:.0f} & "
                  f"{r['iterations']:.2f} & {r['latency_s']:.1f} & {r['gen_cost']:.5f} \\\\")
    el += [r"\hline", r"\end{tabular}}", r"\end{table}"]
    (TABDIR / "tab_efficiency.tex").write_text("\n".join(el))




def extra_stats(df, pivot, ma):
    """Additional analyses on the existing results: all pairwise tests with Holm
    correction, judge-dimension diagnostics, length confound, noise floor between
    the two identical-pipeline systems, and a fabricated-vs-corrupted split of
    hallucinated identifiers."""
    import itertools
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).resolve().parent))
    from db import PlaforEduDB

    out = {}
    # ---- 95% CI of overall per system ----
    out["ci"] = {}
    for s in SYS_ORDER:
        x = df[df.system == s]["overall"]
        h = 1.96 * x.sem()
        out["ci"][s] = (x.mean() - h, x.mean() + h)

    # ---- all pairwise Wilcoxon on overall, Holm-adjusted ----
    pairs = list(itertools.combinations(SYS_ORDER, 2))
    rows = []
    for a, b in pairs:
        d = pivot[b] - pivot[a]  # positive = later system (b) better
        nz = int((d != 0).sum())
        try:
            _, p = stats.wilcoxon(pivot[a], pivot[b])
        except ValueError:
            p = float("nan")
        wins, losses = int((d > 0).sum()), int((d < 0).sum())
        rbc = (wins - losses) / nz if nz else float("nan")  # rank-biserial-like effect
        rows.append(dict(a=a, b=b, diff=d.mean(), p=p, nz=nz, wins=wins, losses=losses, rbc=rbc))
    # Holm
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m = len(rows)
    running = 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, (m - rank) * rows[i]["p"])
        running = max(running, adj)
        rows[i]["p_holm"] = running
    print("\n=== All pairwise Wilcoxon (overall), Holm-adjusted ===")
    for r in rows:
        print(f"  {r['b']:15} - {r['a']:15}: diff={r['diff']:+.3f} p={r['p']:.4f} "
              f"p_holm={r['p_holm']:.4f} W/L={r['wins']}/{r['losses']} nz={r['nz']}")
    tex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{All pairwise comparisons of the overall judge score over " +
        str(pivot.shape[0]) + r" competencies. $\Delta$ is the mean paired difference (second "
        r"minus first system), W/L the number of competencies on which the second system scores "
        r"higher or lower (the rest are ties), $p$ the two-sided Wilcoxon signed-rank $p$-value and "
        r"$p_{\mathrm{Holm}}$ its Holm correction over the six comparisons.}",
        r"\label{tab:pairwise}",
        r"\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llcccc}", r"\hline",
        r"First & Second & $\Delta$ & W/L & $p$ & $p_{\mathrm{Holm}}$ \\", r"\hline",
    ]
    short = {"naive_all": "Naive", "retrieval_topk": "Retrieval",
             "single_agent": "Single-agent", "multi_agent": "Multi-agent"}
    for r in rows:
        fp = lambda v: "$<$0.001" if v < 1e-3 else f"{v:.3g}"
        tex.append(f"{short[r['a']]} & {short[r['b']]} & {r['diff']:+.2f} & "
                   f"{r['wins']}/{r['losses']} & {fp(r['p'])} & {fp(r['p_holm'])} \\\\")
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    out["pairwise_tex"] = "\n".join(tex)
    pm = {(r["a"], r["b"]): r for r in rows}

    # ---- judge-dimension diagnostics ----
    dims = ["relevance", "ordering", "coverage", "overall"]
    eq_cov = float((df.overall == df.coverage).mean())
    corr = df[dims].corr(method="spearman")
    print(f"\n=== Judge diagnostics === overall==coverage in {100*eq_cov:.1f}% of records")
    print(corr.round(2))
    dist = df.overall.value_counts(normalize=True)
    # per-dimension retrieval vs multi and single vs multi
    dimp = {}
    for d in ["relevance", "ordering", "coverage"]:
        pv = df.pivot_table(index="competency_id", columns="system", values=d)
        for s in ["retrieval_topk", "single_agent"]:
            try:
                _, p = stats.wilcoxon(pv[s], pv["multi_agent"])
            except ValueError:
                p = float("nan")
            dimp[(d, s)] = ((pv[s] - pv["multi_agent"]).mean(), p)

    # ---- length confound ----
    sub = df[df.system != "naive_all"]
    rho_ratio, p_ratio = stats.spearmanr(sub.selection_ratio, sub.overall)
    rho_n, p_n = stats.spearmanr(sub.n_selected, sub.overall)
    print(f"Spearman selection_ratio vs overall (non-naive): rho={rho_ratio:.2f} p={p_ratio:.2g}")

    # ---- noise floor: single vs multi where loop did not fire ----
    sa = df[df.system == "single_agent"].set_index("competency_id")
    mm = ma.set_index("competency_id")
    noloop = [c for c in mm.index if mm.loc[c, "iterations"] == 1]
    same_path = sum([x["id"] for x in mm.loc[c, "selected_courses"]] ==
                    [x["id"] for x in sa.loc[c, "selected_courses"]] for c in noloop)
    same_set = sum(set(x["id"] for x in mm.loc[c, "selected_courses"]) ==
                   set(x["id"] for x in sa.loc[c, "selected_courses"]) for c in noloop)
    ov_diff = sum(mm.loc[c, "overall"] != sa.loc[c, "overall"] for c in noloop)
    ov_absdiff = np.mean([abs(mm.loc[c, "overall"] - sa.loc[c, "overall"]) for c in noloop])
    print(f"Noise floor: {len(noloop)} no-loop comps, identical path {same_path}, "
          f"same set {same_set}, judge overall differs {ov_diff}")

    # ---- loop outcomes ----
    fired = mm[mm.iterations > 1]
    n_fired = len(fired)
    n_fired_hall = int((fired.hallucination_rate > 0).sum())
    n_fired_ruim = int((fired.evaluation == "Ruim").sum())

    # ---- fabricated vs corrupted-ID split ----
    dbx = PlaforEduDB()
    cat_names = {r["name"].strip().lower(): r["id"] for r in dbx.get_courses()}
    hall_rows = []
    for s in SYS_ORDER:
        n_fab = n_corr = n_paths = 0
        for _, r in df[df.system == s].iterrows():
            valid = {c["id"] for c in dbx.get_courses_for_competency(r.competency_id)}
            bad = [c for c in r.selected_courses if c["id"] not in valid]
            if bad:
                n_paths += 1
            for c in bad:
                if c.get("name", "").strip().lower() in cat_names:
                    n_corr += 1
                else:
                    n_fab += 1
        hall_rows.append((s, n_paths, n_corr, n_fab))
    dbx.close()
    print("Hallucination split (system, paths affected, corrupted-id, fabricated):", hall_rows)
    htex = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Invalid course identifiers per system, split into \emph{corrupted} "
        r"identifiers, whose course name exists in the catalog, and \emph{fabricated} courses, "
        r"whose name does not exist in the catalog.}",
        r"\label{tab:halluc}",
        r"\begin{tabular}{lccc}", r"\hline",
        r"System & Paths affected & Corrupted ID & Fabricated \\", r"\hline",
    ]
    for s, npth, nc, nf in hall_rows:
        htex.append(f"{SYS_LABEL[s]} & {npth} & {nc} & {nf} \\\\")
    htex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    out["halluc_tex"] = "\n".join(htex)
    hr = {s: (npth, nc, nf) for s, npth, nc, nf in hall_rows}

    # ---- bucket means for prose ----
    bk = df.pivot_table(index="catalog_bucket", columns="system", values="overall", aggfunc="mean")
    nb = df[df.system == "multi_agent"]["catalog_bucket"].value_counts()

    f2 = lambda x: f"{x:.2f}"
    out["macros"] = {
        "eqCovPct": f"{100*eq_cov:.0f}",
        "pctCovEq": f"{100*eq_cov:.0f}",
        "nRecords": str(len(df)),
        "wlRetrNaive": f"{pm[('naive_all','retrieval_topk')]['wins']}",
        "rhoRelOrd": f2(corr.loc["relevance", "ordering"]),
        "rhoRelCov": f2(corr.loc["relevance", "coverage"]),
        "pctThree": f"{100*dist.get(3, 0):.0f}",
        "rhoRatio": f2(rho_ratio),
        "rhoN": f2(rho_n),
        "relDiffRetr": f"{dimp[('relevance','retrieval_topk')][0]:+.2f}",
        "pRelRetr": f"{dimp[('relevance','retrieval_topk')][1]:.2g}",
        "ordDiffRetr": f"{dimp[('ordering','retrieval_topk')][0]:+.2f}",
        "pOrdRetr": f"{dimp[('ordering','retrieval_topk')][1]:.2g}",
        "covDiffRetr": f"{dimp[('coverage','retrieval_topk')][0]:+.2f}",
        "pCovRetr": f"{dimp[('coverage','retrieval_topk')][1]:.2g}",
        "ordDiffSingle": f"{dimp[('ordering','single_agent')][0]:+.2f}",
        "pOrdSingle": f"{dimp[('ordering','single_agent')][1]:.2g}",
        "nNoLoop": str(len(noloop)),
        "nSamePath": str(int(same_path)),
        "nSameSet": str(int(same_set)),
        "nOvDiff": str(int(ov_diff)),
        "nFired": str(n_fired),
        "nFiredHall": str(n_fired_hall),
        "nFiredRuim": str(n_fired_ruim),
        "pHolmNaiveMulti": f"{pm[('naive_all','multi_agent')]['p_holm']:.3g}",
        "pHolmNaiveSingle": f"{pm[('naive_all','single_agent')]['p_holm']:.3g}",
        "pHolmNaiveRetr": f"{pm[('naive_all','retrieval_topk')]['p_holm']:.3g}",
        "pHolmRetrMulti": f"{pm[('retrieval_topk','multi_agent')]['p_holm']:.3g}",
        "pHolmRetrSingle": f"{pm[('retrieval_topk','single_agent')]['p_holm']:.3g}",
        "pHolmSingleMulti": f"{pm[('single_agent','multi_agent')]['p_holm']:.3g}",
        "pRawNaiveSingle": f"{pm[('naive_all','single_agent')]['p']:.3g}",
        "pRawNaiveRetr": f"{pm[('naive_all','retrieval_topk')]['p']:.3g}",
        "pRawRetrSingle": f"{pm[('retrieval_topk','single_agent')]['p']:.3g}",
        "wlRetrMulti": f"{pm[('retrieval_topk','multi_agent')]['losses']}/{pm[('retrieval_topk','multi_agent')]['wins']}",
        "wlSingleMulti": f"{pm[('single_agent','multi_agent')]['wins']}/{pm[('single_agent','multi_agent')]['losses']}",
        "hallMultiPaths": str(hr["multi_agent"][0]),
        "hallMultiCorr": str(hr["multi_agent"][1]),
        "hallMultiFab": str(hr["multi_agent"][2]),
        "hallSinglePaths": str(hr["single_agent"][0]),
        "hallSingleCorr": str(hr["single_agent"][1]),
        "hallSingleFab": str(hr["single_agent"][2]),
        "nBucketA": str(int(nb.get("1-2", 0))), "nBucketB": str(int(nb.get("3-5", 0))),
        "nBucketC": str(int(nb.get("6-10", 0))), "nBucketD": str(int(nb.get(">10", 0))),
        "ovNaiveMid": f2(bk.loc["3-5", "naive_all"]),
        "ovRetrMid": f2(bk.loc["3-5", "retrieval_topk"]),
        "ovMultiMid": f2(bk.loc["3-5", "multi_agent"]),
        "ovRetrTop": f2(bk.loc[">10", "retrieval_topk"]),
        "ovMultiTop": f2(bk.loc[">10", "multi_agent"]),
        "ovNaiveTop": f2(bk.loc[">10", "naive_all"]),
        "ciLoMulti": f2(out["ci"]["multi_agent"][0]), "ciHiMulti": f2(out["ci"]["multi_agent"][1]),
        "ciLoRetr": f2(out["ci"]["retrieval_topk"][0]), "ciHiRetr": f2(out["ci"]["retrieval_topk"][1]),
    }
    return out


if __name__ == "__main__":
    main()
