"""RQ3 — cross-attack generalisation.

For each held-out attack: train on benign + all OTHER attacks, then test on
benign + the held-out attack only. Measures whether the cross-layer encoding
transfers to attack types unseen in training.
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
from ..models.metrics import (detection_latency, detection_report,
                              grouped_bootstrap_ci, paired_bootstrap_auc_diff)
from ..utils.provenance import capture_provenance
from .common import (CLASSICAL_MODELS, _cap_train_groupwise, fit_reducer,
                     group_split, load_experiment_data, matched_train_idx,
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

    benign = data.y == 0
    rows = []
    pred_rows = []
    res_by = {}            # (atk, model) -> result, for paired significance
    skipped = []
    for atk in cfg["holdout_attacks"]:
        is_atk = data.attack_name == atk
        if is_atk.sum() == 0:
            skipped.append((atk, "no windows of this attack in the data"))
            continue
        # train: benign + all attacks EXCEPT the held-out one
        train_keep = benign | ((data.y == 1) & (~is_atk))
        tr_a = tr[train_keep[tr]]
        va_a = va[train_keep[va]] if len(va) else va
        # test: benign + ONLY the held-out attack
        test_keep = benign | is_atk
        te_a = te[test_keep[te]]
        if len(np.unique(data.y[tr_a])) < 2 or len(np.unique(data.y[te_a])) < 2:
            skipped.append((atk, "train or test split was single-class"))
            continue
        # Matched-N: cap train group-wise so quantum + classical compare at the same N
        # (quantum kernel can't use the full split — O(N^2)). classical@full added below.
        tr_a_match = matched_train_idx(tr_a, data, cfgs)
        held_mask = (data.attack_name[te_a] == atk)
        for model in cfg["models"]:
            idx = {"tr": tr_a_match, "va": va_a, "te": te_a}
            res = run_model(model, data, Xr, idx, cfgs)
            res_by[(atk, model)] = res
            pred_rows.extend(predictions_long(res, holdout_attack=atk))
            # recall specifically on the held-out attack windows
            held_recall = float(np.mean(res["y_pred"][held_mask] == 1)) if held_mask.any() else np.nan
            rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
            lat = detection_latency(res["t_end"], res["y_true"], res["y_pred"], res["groups"])
            ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"],
                                      n_boot=n_boot, seed=splits["seed"])
            rows.append({"holdout_attack": atk, "model": model,
                         "n_train": int(len(tr_a_match)),
                         "pr_auc": rep["pr_auc"], "roc_auc": rep["roc_auc"],
                         "f1": rep["f1"], "mcc": rep["mcc"],
                         "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
                         "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
                         "heldout_recall": held_recall,
                         "detection_latency_mean": lat["detection_latency_mean"],
                         "detected_fraction": lat["detected_fraction"],
                         "n_test": rep["n"], "n_heldout": int(held_mask.sum())})
        # classical@full-data context (scaling ceiling the quantum kernel can't reach).
        # Bounded by `full_cap` (default 15k) so SVC-RBF (O(N^2)) stays tractable on CPU;
        # n_train column shows the actual N. Still >> the quantum 3k → shows the scaling gap.
        tr_a_full = _cap_train_groupwise(tr_a, data.groups,
                                         int(cfg.get("full_cap", 15000)), splits["seed"])
        if len(tr_a_match) < len(tr_a_full):
            for model in [m for m in cfg["models"] if m in CLASSICAL_MODELS]:
                res = run_model(model, data, Xr,
                                {"tr": tr_a_full, "va": va_a, "te": te_a}, cfgs)
                hr = float(np.mean(res["y_pred"][held_mask] == 1)) if held_mask.any() else np.nan
                rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
                rows.append({"holdout_attack": atk, "model": f"{model}__full",
                             "n_train": int(len(tr_a)),
                             "pr_auc": rep["pr_auc"], "roc_auc": rep["roc_auc"],
                             "f1": rep["f1"], "mcc": rep["mcc"],
                             "heldout_recall": hr,
                             "n_test": rep["n"], "n_heldout": int(held_mask.sum())})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "crossattack_results.csv", index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_parquet(out_dir / "predictions.parquet")
    # significance: each quantum model vs RBF-SVM on the SAME held-out test windows
    sig_rows = []
    for atk in cfg["holdout_attacks"]:
        ref = res_by.get((atk, "rbf_svm"))
        if ref is None:
            continue
        for qm in ("qsvm_crosslayer", "qsvm_zz", "qsvm_projected"):
            r = res_by.get((atk, qm))
            if r is None:
                continue
            for metric in ("roc_auc", "pr_auc"):
                d = paired_bootstrap_auc_diff(r["y_true"], r["scores"], ref["scores"],
                                              r["groups"], n_boot=n_boot,
                                              seed=splits["seed"], metric=metric)
                sig_rows.append({"holdout_attack": atk, "model": qm, "vs": "rbf_svm", **d})
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(out_dir / "significance.csv", index=False)
    # Always write the held-out-recall pivot (empty if every attack was skipped) so
    # the artifact README.md claims is never silently missing on an exit-0 run.
    if not results.empty:
        pivot = results.pivot_table(index="holdout_attack", columns="model",
                                    values="heldout_recall")
    else:
        pivot = pd.DataFrame()
    pivot.to_csv(out_dir / "crossattack_heldout_recall.csv")
    if skipped:
        for atk, why in skipped:
            print(f"[crossattack] skipped held-out attack '{atk}': {why}",
                  file=sys.stderr)
    elapsed = round(time.time() - t0, 1)
    summary = {"out_dir": str(out_dir), "n_rows": len(results),
               "attacks": cfg["holdout_attacks"], "models": cfg["models"],
               "n_bootstrap": n_boot, "wall_time_s": elapsed,
               "skipped": [{"attack": a, "reason": r} for a, r in skipped]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    capture_provenance(out_dir, timestamp=started, config_path=config_path,
                       extra={"experiment": "crossattack", "wall_time_s": elapsed,
                              "split_seed": splits["seed"], "n_bootstrap": n_boot})
    return summary
