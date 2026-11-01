"""Attack-INTENSITY sweep.

Measures how detectability changes as a single attack type is made subtler or
stronger — e.g. the falsified-position offset magnitude, the random-offset radius,
the drift rate, the flooding factor, or the Sybil ghost count. The matching co-sim
sweep varies ONE intensity knob with a single-attack mix (see
configs/sim/intensity_offset.yaml), so each run differs only in attack intensity.

For each intensity level we take an independent, group-aware (leakage-safe)
train/test split WITHIN that level and evaluate every model, so each number is the
intrinsic detectability AT that intensity. The output is a metric-vs-intensity curve
(with grouped-bootstrap 95% CIs) plus a per-level significance test of each quantum
model against the RBF-SVM baseline.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_yaml
from ..models.metrics import (detection_report, grouped_bootstrap_ci,
                              paired_bootstrap_auc_diff)
from ..utils.plotting import degradation_curves
from ..utils.provenance import capture_provenance
from .common import (fit_reducer, group_split, load_experiment_data,
                     predictions_long, run_model)


def run(config_path: str) -> dict:
    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")
    cfg = load_yaml(config_path)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_boot = int(cfg.get("bootstrap", 200))

    features_cfg = load_yaml(cfg["features_config"])
    cfgs = {
        "features": features_cfg,
        "quantum": load_yaml(cfg["quantum_config"]),
        "baselines": load_yaml(cfg["baselines_config"]),
    }
    data = load_experiment_data(cfg["runs_root"], features_cfg)

    axis = cfg.get("axis", "const_offset_m")
    if axis not in data.points.columns:
        raise ValueError(
            f"intensity axis '{axis}' not in runs points {list(data.points.columns)}; "
            f"generate data with a sweep over '{axis}' (e.g. configs/sim/intensity_offset.yaml).")
    levels = sorted(pd.to_numeric(data.points[axis], errors="coerce").dropna().unique())
    splits = features_cfg["splits"]

    rows, pred_rows, sig_rows, skipped = [], [], [], []
    for lvl in levels:
        mask = (pd.to_numeric(data.points[axis], errors="coerce") == lvl).to_numpy()
        sub = data.subset(mask)
        if len(np.unique(sub.y)) < 2:
            skipped.append((float(lvl), "only one class present"))
            continue
        try:
            tr, va, te = group_split(sub.groups, sub.y, splits["test_frac"],
                                     splits["val_frac"], splits["seed"])
        except Exception as exc:
            skipped.append((float(lvl), f"group split failed: {exc}"))
            print(f"[intensity] skipped level {lvl}: {exc}", file=sys.stderr)
            continue
        reducer = fit_reducer(features_cfg["reduction"], sub.X, sub.blocks, tr)
        Xr = reducer.transform(sub.X)
        res_by = {}
        for model in cfg["models"]:
            res = run_model(model, sub, Xr, {"tr": tr, "va": va, "te": te}, cfgs)
            res_by[model] = res
            pred_rows.extend(predictions_long(res, **{axis: float(lvl)}))
            rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
            ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"],
                                      n_boot=n_boot, seed=splits["seed"])
            rows.append({
                axis: float(lvl), "model": model,
                **{k: rep.get(k) for k in
                   ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall", "fpr_at_tpr"]},
                "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
                "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
                "n_test": rep["n"], "n_pos": rep["n_pos"],
            })
        # per-level significance: each quantum model vs RBF-SVM on the SAME test windows
        ref = res_by.get("rbf_svm")
        if ref is not None:
            for qm in ("qsvm_crosslayer", "qsvm_zz", "qsvm_projected"):
                r = res_by.get(qm)
                if r is None:
                    continue
                for metric in ("roc_auc", "pr_auc"):
                    d = paired_bootstrap_auc_diff(r["y_true"], r["scores"], ref["scores"],
                                                  r["groups"], n_boot=n_boot,
                                                  seed=splits["seed"], metric=metric)
                    sig_rows.append({axis: float(lvl), "model": qm, "vs": "rbf_svm", **d})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "intensity_results.csv", index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_parquet(out_dir / "predictions.parquet")
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(out_dir / "significance.csv", index=False)
    if not results.empty:
        degradation_curves(results, axis, "pr_auc", "model",
                           out_dir / "pr_auc_vs_intensity.png",
                           f"PR-AUC vs attack intensity ({axis})",
                           ci=("pr_auc_lo", "pr_auc_hi"))
        degradation_curves(results, axis, "roc_auc", "model",
                           out_dir / "roc_auc_vs_intensity.png",
                           f"ROC-AUC vs attack intensity ({axis})",
                           ci=("roc_auc_lo", "roc_auc_hi"))

    elapsed = round(time.time() - t0, 1)
    summary = {"out_dir": str(out_dir), "axis": axis,
               "levels": [float(x) for x in levels], "models": cfg["models"],
               "n_rows": int(len(results)), "n_bootstrap": n_boot,
               "wall_time_s": elapsed,
               "skipped": [{"level": l, "reason": r} for l, r in skipped],
               "artifacts": ["intensity_results.csv", "predictions.parquet",
                             "significance.csv", "pr_auc_vs_intensity.{png,pdf}",
                             "roc_auc_vs_intensity.{png,pdf}", "provenance.json"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    capture_provenance(out_dir, timestamp=started, config_path=config_path,
                       extra={"experiment": "intensity", "axis": axis,
                              "wall_time_s": elapsed})
    return summary
