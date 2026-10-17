"""Generate the two schematic figures: data schema (ER) and agentic workflow graph."""
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIGDIR = Path(__file__).resolve().parent.parent / "papers" / "ieee-access" / "paper" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def box(ax, xy, w, h, text, fc, ec="#333333", fs=9, boldtitle=None):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.3))
    if boldtitle:
        ax.text(x + w / 2, y + h - 0.16, boldtitle, ha="center", va="top",
                fontsize=fs + 1, fontweight="bold")
        ax.text(x + w / 2, y + h - 0.42, text, ha="center", va="top", fontsize=fs - 1)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(ax, a, b, color="#333333", style="-|>", rad=0.0, lw=1.4):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}"))


def er_table(ax, x, y, w, h, title, count, attrs, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec="#333333", lw=1.3))
    ax.text(x + w / 2, y + h - 0.28, f"{title}", ha="center", va="top",
            fontsize=10, fontweight="bold")
    ax.text(x + w / 2, y + h - 0.62, f"({count})", ha="center", va="top", fontsize=8.5)
    ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.82, y + h - 0.82], color="#888", lw=0.8)
    for i, a in enumerate(attrs):
        ax.text(x + w / 2, y + h - 1.05 - i * 0.34, a, ha="center", va="top", fontsize=8.5)


def er_diagram():
    fig, ax = plt.subplots(figsize=(7.8, 3.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    er_table(ax, 0.2, 0.7, 2.7, 2.6, "COMPETENCIES", 69,
             ["id (PK)", "name", "description"], "#DCE6F1")
    er_table(ax, 7.1, 0.7, 2.7, 2.6, "COURSES", 442,
             ["id (PK)", "name", "description", "hours"], "#E2EFDA")
    er_table(ax, 3.6, 0.95, 2.8, 2.1, "COURSES_COMPETENCIES", 1002,
             ["id (PK)", "competencyId (FK)", "courseId (FK)"], "#FCE4D6")
    arrow(ax, (2.9, 2.0), (3.6, 2.0))
    arrow(ax, (6.4, 2.0), (7.1, 2.0))
    ax.text(3.15, 2.22, "1..N", fontsize=8)
    ax.text(6.62, 2.22, "N..1", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_data_schema.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def workflow_graph():
    fig, ax = plt.subplots(figsize=(4.2, 6.2))
    ax.set_xlim(0, 6)
    ax.set_ylim(0, 12)
    ax.axis("off")
    W, H, cx = 3.4, 1.15, 1.3

    def node(y, title, sub, fc):
        box(ax, (cx, y), W, H, sub, fc, boldtitle=title, fs=9)
        return (cx + W / 2, y, cx + W / 2, y + H)

    start = (cx + W / 2, 11.4)
    ax.scatter(*start, s=90, color="#555", zorder=3)
    ax.text(start[0] + 0.15, start[1], "START", va="center", fontsize=8)

    n1 = node(9.7, "Context node", "SQL retrieval:\ncompetency + candidate courses", "#DCE6F1")
    n2 = node(7.3, "Generator node (agent)", "select & order courses,\njustify each choice", "#E2EFDA")
    n3 = node(4.9, "Evaluator node (critic)", "check objective + ≥3 courses\n→ Bom / Ruim + feedback", "#FCE4D6")
    end_y = 3.2
    ax.add_patch(FancyBboxPatch((cx + 1.0, end_y), 1.4, 0.7,
                 boxstyle="round,pad=0.02,rounding_size=0.06", fc="#555", ec="#333"))
    ax.text(cx + 1.7, end_y + 0.35, "END", ha="center", va="center",
            color="white", fontsize=10, fontweight="bold")

    arrow(ax, (start[0], start[1] - 0.15), (n1[0], n1[3]))
    arrow(ax, (n1[0], n1[1]), (n2[0], n2[3]))
    arrow(ax, (n2[0], n2[1]), (n3[0], n3[3]))
    # end edge
    arrow(ax, (n3[0], n3[1]), (cx + 1.7, end_y + 0.7))
    ax.text(n3[0] + 0.15, (n3[1] + end_y + 0.7) / 2, '"Bom"', fontsize=8)
    # feedback loop: evaluator -> generator (Ruim)
    arrow(ax, (cx, 5.3), (cx, 7.6), color="#C00000", rad=-0.55, lw=1.6)
    ax.text(0.15, 6.4, '"Ruim"\n+ feedback', fontsize=8, color="#C00000", ha="left")

    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_langgraph_flow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def system_one_pipeline():
    """System One curation: per-candidate typed decisions, deterministic assembly."""
    fig, ax = plt.subplots(figsize=(7.8, 2.5))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 4)
    ax.axis("off")
    box(ax, (0.1, 1.0), 2.5, 2.0, "objective $o_k$\n+ one candidate\n$c \\in A_k$\n(name, description)",
        "#DCE6F1", boldtitle="State", fs=8)
    box(ax, (3.3, 1.0), 3.0, 2.0, "single forward pass,\nno generated tokens\n(Jev / Kev / Laya)",
        "#FCE4D6", boldtitle="System One model", fs=8)
    box(ax, (7.0, 2.15), 2.6, 1.05, "noul: P(relevant)", "#E2EFDA", fs=8.5)
    box(ax, (7.0, 0.8), 2.6, 1.05, "score: level\nstart / middle / end", "#E2EFDA", fs=8.5)
    box(ax, (10.3, 0.6), 2.6, 2.8, "keep $P\\geq 0.5$\n(or top-5);\npad to $\\min(3,|A_k|)$;\nsort by level", "#EDEDED",
        boldtitle="Assemble path", fs=8)
    arrow(ax, (2.6, 2.0), (3.3, 2.0))
    arrow(ax, (6.3, 2.2), (7.0, 2.65))
    arrow(ax, (6.3, 1.8), (7.0, 1.3))
    arrow(ax, (9.6, 2.65), (10.3, 2.2))
    arrow(ax, (9.6, 1.3), (10.3, 1.8))
    ax.text(4.8, 0.55, "repeated for every $c \\in A_k$", ha="center", fontsize=8, style="italic")
    fig.tight_layout()
    fig.savefig(FIGDIR / "fig_system_one.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    er_diagram()
    workflow_graph()
    system_one_pipeline()
    print("wrote figures to", FIGDIR)
