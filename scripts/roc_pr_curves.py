#!/usr/bin/env python
"""ROC and precision-recall curves for Fig. 2.

Top row: dev_grid, matched N (the matched-N detectors of Table III; packet loss 0, seed-7
leakage-safe split, 26,392 test windows). Bottom row: VeReMi NextGen (binary task, full
feature set, seed 7, official test split). Curves are downsampled for plotting;
scripts/render_figures.py draws Fig. 2 from the output.

Reads : results/packetloss_dev/predictions.parquet
        (python -m qkv2x.cli experiment --config configs/experiments/packetloss.yaml)
        results/veremi_nextgen/predictions.parquet
        (python -m qkv2x.cli experiment --config configs/experiments/veremi_nextgen.yaml)
Writes: full_paper_data/roc_pr/roc_pr_curves.csv  (dataset,model,curve,x,y)
        full_paper_data/roc_pr/roc_pr_auc.json    {dataset:{model:{n_test,roc_auc,pr_auc}}}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
)

REPO = Path(__file__).resolve().parents[1]
PRED_DEV = REPO / "results" / "packetloss_dev" / "predictions.parquet"
PRED_NG = REPO / "results" / "veremi_nextgen" / "predictions.parquet"
OUT_DIR = REPO / "full_paper_data" / "roc_pr"
OUT_CSV = OUT_DIR / "roc_pr_curves.csv"
OUT_JSON = OUT_DIR / "roc_pr_auc.json"

# Plotted detectors per dataset (order preserved for plotting).
MODELS = {
    "dev_grid": ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest", "mlp"],
    "nextgen": ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest", "xgboost", "lstm_seq"],
}

MAX_POINTS = 150


def downsample_monotone(x, y, max_points=MAX_POINTS):
    """Keep <= max_points samples while preserving the curve's monotone x axis.

    Always retains the two endpoints. x from sklearn roc/pr curves is monotone
    (roc fpr ascending; pr recall descending). We pick evenly spaced indices,
    which preserves monotonicity and endpoint anchoring.
    """
    n = len(x)
    if n <= max_points:
        idx = np.arange(n)
    else:
        idx = np.linspace(0, n - 1, max_points)
        idx = np.unique(np.round(idx).astype(int))
    return x[idx], y[idx]


def curves(sub, dataset, rows, aucs):
    aucs[dataset] = {}
    for m in MODELS[dataset]:
        d = sub[sub["model"] == m]
        if d.empty:
            raise SystemExit(f"MISSING model rows: {dataset}/{m}")
        y = d["y_true"].to_numpy().astype(int)
        s = d["score"].to_numpy().astype(float)
        if len(np.unique(y)) < 2:
            raise SystemExit(f"single-class y_true for {dataset}/{m}; cannot compute curves")

        # ROC
        fpr, tpr, _ = roc_curve(y, s)
        fpr, tpr = downsample_monotone(fpr, tpr)
        for xv, yv in zip(fpr, tpr):
            rows.append((dataset, m, "roc", float(xv), float(yv)))

        # PR
        prec, rec, _ = precision_recall_curve(y, s)
        # precision_recall_curve returns thresholds-1 length arrays with a
        # trailing (recall=0, precision=1) anchor; recall is descending.
        rec, prec = downsample_monotone(rec, prec)
        for xv, yv in zip(rec, prec):
            rows.append((dataset, m, "pr", float(xv), float(yv)))

        aucs[dataset][m] = {
            "n_test": int(len(d)),
            "roc_auc": float(roc_auc_score(y, s)),
            "pr_auc": float(average_precision_score(y, s)),
        }


def main():
    rows, aucs = [], {}
    dev = pd.read_parquet(PRED_DEV)
    curves(dev[dev["packet_loss"] == 0.0], "dev_grid", rows, aucs)
    ng = pd.read_parquet(PRED_NG)
    curves(ng[(ng["task"] == "binary") & (ng["variant"] == "full") & (ng["seed"] == 7)],
           "nextgen", rows, aucs)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows, columns=["dataset", "model", "curve", "x", "y"])
    out.to_csv(OUT_CSV, index=False)
    with open(OUT_JSON, "w") as fh:
        json.dump(aucs, fh, indent=2)

    print(f"rows={len(out)} -> {OUT_DIR.relative_to(REPO)}")
    print(json.dumps(aucs, indent=2))


if __name__ == "__main__":
    main()
