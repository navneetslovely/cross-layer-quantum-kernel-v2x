"""H3 — low-label learning curves.

At a fixed sweep point, train each model on increasing numbers of labelled windows
(group-aware subsampling, several repeats) and evaluate on a fixed held-out test
set. Validates whether the quantum kernel reaches a target F1 with fewer labels.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_yaml
from ..models.metrics import detection_report
from ..utils.plotting import learning_curves
from ..utils.provenance import capture_provenance
from .common import fit_reducer, group_split, load_experiment_data, run_model


def _subsample_train(tr, y, groups, n_train, rng):
    """Group-aware stratified subsample of the training indices to ~n_train rows."""
    if n_train >= len(tr):
        return tr
    # keep groups intact: shuffle groups, accumulate until n_train reached,
    # while ensuring both classes are present.
    g = groups[tr]
    uniq = rng.permutation(np.unique(g))
    chosen, total = [], 0
    for gid in uniq:
        idx = tr[g == gid]
        chosen.append(idx)
        total += len(idx)
        if total >= n_train:
            break
    sel = np.concatenate(chosen)
    if len(np.unique(y[sel])) < 2:
        # extend until both classes present
        for gid in uniq:
            idx = tr[g == gid]
            sel = np.unique(np.concatenate([sel, idx]))
            if len(np.unique(y[sel])) == 2:
                break
    return sel


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

    # restrict to the chosen point
    point = cfg["point"]
    mask = np.ones(len(data), dtype=bool)
    for k, v in point.items():
        if k in data.points.columns:
            mask &= (data.points[k].astype(float) == float(v)).to_numpy()
    data = data.subset(mask)

    splits = features_cfg["splits"]
    tr, va, te = group_split(data.groups, data.y, splits["test_frac"],
                             splits["val_frac"], splits["seed"])
    reducer = fit_reducer(features_cfg["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)

    rng = np.random.default_rng(splits["seed"])
    rows = []
    for n_train in cfg["n_train_grid"]:
        for rep in range(int(cfg.get("repeats", 5))):
            sub = _subsample_train(tr, data.y, data.groups, int(n_train), rng)
            va_sub = va  # validation kept full (used only by OC-SVM nu selection)
            for model in cfg["models"]:
                idx = {"tr": sub, "va": va_sub, "te": te}
                try:
                    res = run_model(model, data, Xr, idx, cfgs)
                except Exception as exc:
                    rows.append({"model": model, "n_train": int(n_train),
                                 "n_train_actual": int(len(sub)), "repeat": rep,
                                 "pr_auc": np.nan, "f1": np.nan, "mcc": np.nan,
                                 "recall": np.nan, "error": str(exc)})
                    continue
                rep_m = detection_report(res["y_true"], res["scores"], res["y_pred"])
                # x-axis = the REQUESTED grid value so the `repeats` runs aggregate to
                # one mean+std per grid point. The group-aware subsample overshoots the
                # target by up to one group, differently each repeat; recording the
                # realized size (len(sub)) here would scatter every repeat onto its own
                # x and destroy the averaged learning curve / error bars. The realized
                # size is kept separately in n_train_actual for diagnostics.
                rows.append({"model": model, "n_train": int(n_train),
                             "n_train_actual": int(len(sub)), "repeat": rep,
                             "pr_auc": rep_m["pr_auc"], "f1": rep_m["f1"],
                             "mcc": rep_m["mcc"], "recall": rep_m["recall"],
                             "error": ""})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "lowlabel_results.csv", index=False)
    if not results.empty:
        learning_curves(results, out_dir / "lowlabel_pr_auc.png",
                        "Low-label learning curves (PR-AUC)", y_col="pr_auc")
        learning_curves(results, out_dir / "lowlabel_f1.png",
                        "Low-label learning curves (F1)", y_col="f1")
    elapsed = round(time.time() - t0, 1)
    summary = {"out_dir": str(out_dir), "n_rows": len(results),
               "n_train_grid": cfg["n_train_grid"], "models": cfg["models"],
               "repeats": int(cfg.get("repeats", 5)), "wall_time_s": elapsed}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    capture_provenance(out_dir, timestamp=started, config_path=config_path,
                       extra={"experiment": "lowlabel", "wall_time_s": elapsed})
    return summary
