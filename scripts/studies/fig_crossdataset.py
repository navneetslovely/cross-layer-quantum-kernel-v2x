#!/usr/bin/env python3
"""can_seeds figure — Fig. 3 (fig:crossdataset) regenerated with three-seed CAN means +/- sd.

Mirrors the style of scripts/render_figures_hq.py (Okabe-Ito palette, SIZE_BARW, legend below,
STIX text) but reads the CAN bars from results/studies/can_seeds_per_dataset.csv (seeds
7/17/27) instead of the single-seed CSVs; the VeReMi bar is unchanged (seed 7, no error bar).

Output: figures/frontier_crossdataset.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
PD = REPO / "full_paper_data"
OUT = REPO / "figures"
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
C_CROSS, C_ZZ, C_CLASS = PALETTE[0], PALETTE[1], PALETTE[2]
FS_BASE, FS_LAB, FS_TICK, FS_LEG = 8.5, 9.0, 8.0, 7.3
SIZE_BARW = (4.6, 2.6)
CLASSICAL = ["rbf_svm", "random_forest", "xgboost", "mlp"]


def main():
    plt.rcParams.update({"font.size": FS_BASE, "axes.labelsize": FS_LAB, "xtick.labelsize": FS_TICK,
                         "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEG, "mathtext.fontset": "stix",
                         "font.family": "STIXGeneral", "figure.constrained_layout.use": True,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    rows = []  # (label, cross, cross_sd, zz, zz_sd, best, best_sd)
    v = pd.read_csv(PD / "frontier" / "veremi__veremi_binary_results.csv").set_index("model")
    vc = v.loc[[m for m in CLASSICAL if m in v.index], "roc_auc"]
    rows.append(("VeReMi\n(V2V, seed 7)", v.loc["qsvm_crosslayer", "roc_auc"], 0.0, v.loc["qsvm_zz", "roc_auc"], 0.0,
                 vc.max(), 0.0))
    a = pd.read_csv(REPO / "results" / "studies" / "can_seeds_per_dataset.csv")
    for tag, label in [("cic", "CIC-IoV\n(CAN)"), ("hcrl", "HCRL\n(CAN)"), ("road", "ROAD\n(CAN)")]:
        d = a[a.dataset == tag].set_index("model")
        cl = d[d.index.isin(CLASSICAL)]
        best = cl.roc_mean.idxmax()
        rows.append((label, d.loc["qsvm_crosslayer", "roc_mean"], d.loc["qsvm_crosslayer", "roc_sd"],
                     d.loc["qsvm_zz", "roc_mean"], d.loc["qsvm_zz", "roc_sd"],
                     d.loc[best, "roc_mean"], d.loc[best, "roc_sd"]))
    x = np.arange(len(rows)); w = 0.26
    fig, ax = plt.subplots(figsize=SIZE_BARW)
    ek = dict(capsize=2.5, lw=0.8, ecolor="k")
    ax.bar(x - w, [r[1] for r in rows], w, yerr=[r[2] for r in rows], label="QSVM cross-layer", color=C_CROSS,
           edgecolor="k", linewidth=0.4, error_kw=ek)
    ax.bar(x, [r[3] for r in rows], w, yerr=[r[4] for r in rows], label="QSVM ZZ", color=C_ZZ,
           edgecolor="k", linewidth=0.4, error_kw=ek)
    ax.bar(x + w, [r[5] for r in rows], w, yerr=[r[6] for r in rows], label="best classical", color=C_CLASS,
           edgecolor="k", linewidth=0.4, error_kw=ek)
    ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows]); ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1.08)
    ax.axhline(0.5, color="0.6", lw=0.6, ls=":")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "frontier_crossdataset.pdf"); fig.savefig(OUT / "frontier_crossdataset.png", dpi=300)
    for r in rows:
        print(f"{r[0].replace(chr(10), ' '):22s} cross {r[1]:.3f}±{r[2]:.3f}  zz {r[3]:.3f}±{r[4]:.3f}  best {r[5]:.3f}±{r[6]:.3f}")
    print("saved ->", OUT / "frontier_crossdataset.pdf")


if __name__ == "__main__":
    main()
