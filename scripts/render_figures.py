"""Regenerate the data figures of the paper
("A Cross-Layer Quantum Kernel for Misbehaviour Detection in Vehicular Networks").

Self-contained: reads ONLY the committed cache in ``full_paper_data/`` and writes the paper
figures (vector PDF, title-free — the LaTeX ``\\caption`` carries the title) to ``figures/``.

Colour-blind-safe Okabe-Ito palette, one fixed colour per model. constrained_layout keeps
text non-overlapping even at 1/3 text-width (3-up placement is safe).

    python scripts/render_figures.py

Draws Fig. 2 (frontier_roc_pr) and Fig. 4 (frontier_detectability). Not drawn here: Fig. 1
(schematic, no data) and Fig. 3 (scripts/studies/fig_crossdataset.py).
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from cycler import cycler
import pandas as pd

# ============================== CONFIG ==============================
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
GREY = "#7F7F7F"
C_CROSS, C_CLASS, C_CLASS2 = PALETTE[0], PALETTE[2], PALETTE[4]

MODEL_COLORS = {
    "qsvm_crosslayer": PALETTE[0], "qsvm_zz": PALETTE[1], "rbf_svm": PALETTE[2],
    "random_forest": PALETTE[4], "xgboost": PALETTE[5], "mlp": PALETTE[3],
    "lstm_seq": PALETTE[7], "knn_bagging": GREY, "rules": "#9aa0a6",
}
FS_BASE, FS_LAB, FS_TICK, FS_LEG = 8.5, 9.0, 8.0, 7.3
LW, MS = 1.8, 5.5
SIZE_SQ = (2.7, 2.35); SIZE_ROC = (4.3, 4.75)
# ===================================================================

REPO = Path(__file__).resolve().parents[1]          # repository root (scripts/..)
PD = REPO / "full_paper_data"
OUT = REPO / "figures"
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    "figure.dpi": 120, "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS_BASE, "axes.titlesize": FS_BASE, "axes.labelsize": FS_LAB,
    "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "lines.linewidth": LW, "lines.markersize": MS, "lines.markeredgewidth": 0.5,
    "legend.frameon": False, "legend.handlelength": 1.5, "legend.columnspacing": 1.1,
    "legend.labelspacing": 0.3, "legend.handletextpad": 0.5, "legend.borderaxespad": 0.2,
    "axes.prop_cycle": cycler(color=PALETTE),
})

made = []
def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf"); plt.close(fig); made.append(name); print(f"  wrote {name}.pdf")
def csv(rel):
    p = PD / rel; return pd.read_csv(p) if p.exists() else None

PRETTY = {"qsvm_crosslayer": "QSVM cross-layer", "qsvm_zz": "QSVM ZZ", "rbf_svm": "RBF-SVM",
          "random_forest": "RF", "knn_bagging": "kNN", "mlp": "MLP", "xgboost": "XGB",
          "lstm_seq": "LSTM", "rules": "rules"}
mc = lambda m: MODEL_COLORS.get(m, GREY)
pn = lambda m: PRETTY.get(m, m)

# Fig. 2: ROC / precision-recall curves; top row dev_grid (matched N), bottom row VeReMi NextGen
# (scripts/roc_pr_curves.py)
df = csv("roc_pr/roc_pr_curves.csv")
if df is not None:
    fig = plt.figure(figsize=SIZE_ROC)
    order = [m for m in MODEL_COLORS if m in set(df["model"])]
    rows = [("dev_grid", r"(a) dev_grid, matched $N$"), ("nextgen", "(b) VeReMi NextGen")]
    for sf, (ds, title) in zip(fig.subfigures(2, 1), rows):
        axes = sf.subplots(1, 2)
        d = df[df.dataset == ds]
        for curve, ax, (xl, yl) in [("roc", axes[0], ("false-positive rate", "true-positive rate")),
                                    ("pr", axes[1], ("recall", "precision"))]:
            for m in order:
                s = d[(d.model == m) & (d.curve == curve)].sort_values("x")
                if len(s): ax.plot(s["x"], s["y"], color=mc(m), lw=LW)
            if curve == "roc": ax.plot([0, 1], [0, 1], ls=":", color=GREY, lw=1)
            ax.set_xlabel(xl); ax.set_ylabel(yl); ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        sf.suptitle(title, fontsize=FS_LAB)
    handles = [Line2D([0], [0], color=mc(m), lw=LW, label=pn(m)) for m in order]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=4, fontsize=FS_LEG, frameon=False, columnspacing=0.8, handlelength=1.2)
    save(fig, "frontier_roc_pr")

# Fig. 4: separability ordering
df = csv("frontier/frontier__controllability.csv")
if df is not None:
    fig, ax = plt.subplots(figsize=SIZE_SQ)
    for (ds, sub), mk, col in zip(df.groupby("dataset"), ["o", "s", "^"], [C_CROSS, C_CLASS, C_CLASS2]):
        ax.scatter(sub["D_full"], sub["Y_rocauc"], marker=mk, s=42, color=col, edgecolor="k", linewidth=0.5,
                   label={"dev_grid": "dev_grid", "nextgen": "NextGen"}.get(ds, ds))
    ax.set_xscale("log"); ax.set_xlabel(r"model-free divergence $D_{\mathrm{full}}$")
    ax.set_ylabel(r"detectability $Y$ (ROC-AUC)")
    ax.legend(title="dataset", loc="lower right")
    save(fig, "frontier_detectability")

print(f"[done] wrote {len(made)} paper figures -> {OUT}")
for m in sorted(made): print("  ", m)
