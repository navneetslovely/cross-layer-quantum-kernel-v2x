"""Sybil power-control ablation.

Sweeps the attacker's Sybil power-control mode (same | offset | random) and measures
how detectability changes. In `same` mode every ghost shares one RSSI signature, so the
F3 cross-ID RSSI-correlation feature (sybil_rssi_corr_max ~ 1) flags it easily; `offset`
and `random` vary per-ghost transmit power to mimic distinct physical sources, eroding
that signature. This isolates whether the cross-layer / quantum-kernel detector holds up
better than the simpler |RSSI-diff| baseline as the Sybil attacker becomes more evasive.

Data is produced by a co-sim sweep over the `sybil_power_mode` axis
(configs/sim/sybil_ablation.yaml). For each mode we do an independent, group-aware
train/test split WITHIN that condition and evaluate every model, so each number reflects
detectability AT that evasion level.
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
from ..models.metrics import detection_report
from ..utils.plotting import grouped_bars
from ..utils.provenance import capture_provenance
from .common import (fit_reducer, group_split, load_experiment_data,
                     predictions_long, run_model)


def run(config_path: str) -> dict:
    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")
    cfg = load_yaml(config_path)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    features_cfg = load_yaml(cfg["features_config"])
    cfgs = {
        "features": features_cfg,
        "quantum": load_yaml(cfg["quantum_config"]),
        "baselines": load_yaml(cfg["baselines_config"]),
    }
    data = load_experiment_data(cfg["runs_root"], features_cfg)

    axis = cfg.get("axis", "sybil_power_mode")
    if axis not in data.points.columns:
        raise ValueError(
            f"ablation axis '{axis}' not in runs points {list(data.points.columns)}; "
            f"generate data with a sweep over '{axis}' (configs/sim/sybil_ablation.yaml).")
    modes = list(pd.unique(data.points[axis]))
    splits = features_cfg["splits"]
    corr_i = (data.feature_names.index("sybil_rssi_corr_max")
              if "sybil_rssi_corr_max" in data.feature_names else None)

    rows, skipped, pred_rows = [], [], []
    for mode in modes:
        mask = (data.points[axis] == mode).to_numpy()
        sub = data.subset(mask)
        if len(np.unique(sub.y)) < 2:
            skipped.append((str(mode), "only one class present"))
            continue
        try:
            tr, va, te = group_split(sub.groups, sub.y, splits["test_frac"],
                                     splits["val_frac"], splits["seed"])
        except Exception as exc:
            # Too few benign senders to form a group-aware split for this mode; skip
            # rather than abort the whole ablation (sybil-only mixes are positive-heavy).
            skipped.append((str(mode), f"group split failed: {exc}"))
            print(f"[ablation] skipped mode '{mode}': {exc}", file=sys.stderr)
            continue
        reducer = fit_reducer(features_cfg["reduction"], sub.X, sub.blocks, tr)
        Xr = reducer.transform(sub.X)
        # Mean correlated-RSSI Sybil signature on attack windows (should fall as the
        # attacker varies per-ghost power: high for 'same', lower for 'offset'/'random').
        corr_mean = (float(np.mean(sub.X[sub.y == 1, corr_i]))
                     if corr_i is not None and (sub.y == 1).any() else float("nan"))
        for model in cfg["models"]:
            idx = {"tr": tr, "va": va, "te": te}
            res = run_model(model, sub, Xr, idx, cfgs)
            pred_rows.extend(predictions_long(res, **{axis: str(mode)}))
            rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
            rows.append({
                axis: str(mode), "model": model,
                **{k: rep.get(k) for k in
                   ["pr_auc", "roc_auc", "f1", "mcc", "recall", "fpr_at_tpr"]},
                "sybil_rssi_corr_max_mean_attack": corr_mean,
                "n_test": rep["n"], "n_pos": rep["n_pos"],
            })

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "ablation_results.csv", index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_parquet(out_dir / "predictions.parquet")
    if not results.empty:
        grouped_bars(results, x=axis, group="model", value="pr_auc",
                     out_path=out_dir / "ablation_pr_auc.png",
                     title="Detection vs Sybil power-control mode", ylabel="PR-AUC")

    elapsed = round(time.time() - t0, 1)
    summary = {"out_dir": str(out_dir), "axis": axis,
               "modes": [str(m) for m in modes], "models": cfg["models"],
               "n_rows": int(len(results)), "wall_time_s": elapsed,
               "skipped": [{"mode": m, "reason": r} for m, r in skipped]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    capture_provenance(out_dir, timestamp=started, config_path=config_path,
                       extra={"experiment": "ablation", "axis": axis,
                              "wall_time_s": elapsed})
    return summary
