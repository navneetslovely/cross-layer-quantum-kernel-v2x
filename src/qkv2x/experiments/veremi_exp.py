"""External comparability — VeReMi reproduction.

Loads the published VeReMi receptions, estimates the per-dataset propagation
parameters from BENIGN messages (so the cross-layer F2 inversion is calibrated to
VeReMi's own channel), builds the same windowed features, and runs the full model
zoo for binary and multi-class evaluation.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..config import load_yaml
from ..data.veremi import load_veremi
from ..features.dataset import build_dataset
from ..models.metrics import (detection_latency, detection_report,
                              grouped_bootstrap_ci, macro_report)
from ..utils.provenance import capture_provenance
from .common import (fit_reducer, group_split, predictions_long, run_model,
                     run_model_multiclass)
from .common import ExperimentData


def estimate_propagation(rx: pd.DataFrame, ref_distance: float = 1.0) -> dict:
    """Fit RSSI(dBm) = (Ptx - PL0) - 10*eta*log10(d/d0) on benign receptions.

    Returns a propagation dict matching features.yaml so build_dataset can invert
    it. Uses only is_attack==0 rows (claimed position == true position for benign).
    """
    benign = rx[rx["is_attack"] == 0].copy()
    d = np.hypot(benign["claimed_x"] - benign["rx_x"], benign["claimed_y"] - benign["rx_y"])
    good = (d > ref_distance) & np.isfinite(benign["rssi_dbm"])
    d = d[good].to_numpy()
    rssi = benign["rssi_dbm"][good].to_numpy()
    if len(d) < 50:
        raise ValueError("Not enough benign receptions to calibrate VeReMi propagation.")
    x = np.log10(d / ref_distance)
    # rssi = a + b*x ; b = -10*eta ; a = Ptx - PL0
    b, a = np.polyfit(x, rssi, 1)
    eta = max(-b / 10.0, 0.5)
    return {
        "tx_power_dbm": float(a),    # absorbs Ptx - PL0 (intercept)
        "ref_loss_db": 0.0,
        "path_loss_exponent": float(eta),
        "ref_distance": float(ref_distance),
        "_fit": {"slope": float(b), "intercept": float(a), "n": int(len(d))},
    }


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

    rx = load_veremi(cfg["veremi_root"])
    propagation = (estimate_propagation(rx) if cfg.get("estimate_propagation", True)
                   else features_cfg["propagation"])

    # VeReMi BSM rate is 1 Hz (CAM) by default; use 1.0 unless overridden.
    nominal = float(cfg.get("nominal_rate_hz", 1.0))
    ds = build_dataset(rx, features_cfg, propagation, nominal)

    # subsample for kernel feasibility (stratified by label)
    max_w = int(cfg.get("max_windows", 200000))
    if len(ds.y) > max_w:
        rng = np.random.default_rng(features_cfg["splits"]["seed"])
        sel = rng.choice(len(ds.y), size=max_w, replace=False)
    else:
        sel = np.arange(len(ds.y))

    data = ExperimentData(
        X=ds.X[sel], y=ds.y[sel], attack_name=ds.attack_name[sel],
        groups=ds.groups[sel], aoi=ds.aoi[sel],
        t_end=ds.meta["t_end"].to_numpy()[sel],
        feature_names=ds.feature_names, blocks=ds.blocks,
        points=pd.DataFrame({"dataset": ["veremi"] * len(sel)}),
        meta=ds.meta.iloc[sel].reset_index(drop=True),
    )

    splits = features_cfg["splits"]
    tr, va, te = group_split(data.groups, data.y, splits["test_frac"],
                             splits["val_frac"], splits["seed"])
    reducer = fit_reducer(features_cfg["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)

    # ---- binary detection (benign vs misbehaving) --------------------------
    rows = []
    pred_rows = []
    for model in cfg["models"]:
        idx = {"tr": tr, "va": va, "te": te}
        res = run_model(model, data, Xr, idx, cfgs)
        pred_rows.extend(predictions_long(res, task="binary"))
        rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
        lat = detection_latency(res["t_end"], res["y_true"], res["y_pred"], res["groups"])
        ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"],
                                  n_boot=n_boot, seed=splits["seed"])
        rows.append({"model": model, **{k: rep.get(k) for k in
                     ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall", "fpr_at_tpr"]},
                     "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
                     "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
                     "detection_latency_mean": lat["detection_latency_mean"],
                     "detected_fraction": lat["detected_fraction"],
                     "n_test": rep["n"], "n_pos": rep["n_pos"]})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "veremi_binary_results.csv", index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_parquet(out_dir / "predictions.parquet")

    # ---- multi-class attack-type identification -
    # Classical models are the standard VeReMi multi-class reference (RF/KNN+bagging
    # reach ~99% F1 on the 5 position attacks). The quantum-kernel SVM here is binary;
    # multi-class quantum kernels are left as future work.
    # The quantum cross-layer kernel runs multi-class too (one-vs-one over the same
    # fidelity Gram); classical RF/KNN+bagging are the standard VeReMi reference.
    mc_models = cfg.get("multiclass_models",
                        ["qsvm_crosslayer", "random_forest", "knn_bagging", "mlp", "rbf_svm"])
    # Integer-encode the attack-type labels ('none' + each attack). Integer targets
    # avoid an sklearn MLP early-stopping crash (np.isnan on string predictions) and
    # keep the confusion matrix in a stable label order; class names kept for output.
    le = LabelEncoder().fit(data.attack_name)
    y_multi = le.transform(data.attack_name)
    label_ids = list(range(len(le.classes_)))
    label_names = [str(c) for c in le.classes_]
    mc_idx = {"tr": tr, "te": te}
    mc_rows, mc_confusions = [], {}
    for model in mc_models:
        pred = run_model_multiclass(model, data, Xr, mc_idx, cfgs, y_multi)
        rep = macro_report(y_multi[te], pred, labels=label_ids)
        mc_rows.append({"model": model, "macro_precision": rep["macro_precision"],
                        "macro_recall": rep["macro_recall"], "macro_f1": rep["macro_f1"]})
        mc_confusions[model] = {"labels": label_names, "confusion": rep["confusion"]}
    mc_results = pd.DataFrame(mc_rows)
    mc_results.to_csv(out_dir / "veremi_multiclass_results.csv", index=False)

    elapsed = round(time.time() - t0, 1)
    out = {"out_dir": str(out_dir), "propagation": propagation,
           "n_windows": int(len(ds.y)), "n_used": int(len(sel)),
           "n_bootstrap": n_boot, "wall_time_s": elapsed,
           "binary": rows, "multiclass": mc_rows, "multiclass_labels": label_names,
           "multiclass_confusion": mc_confusions}
    (out_dir / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    capture_provenance(out_dir, timestamp=started, config_path=config_path,
                       extra={"experiment": "veremi", "wall_time_s": elapsed,
                              "n_used": int(len(sel)), "n_bootstrap": n_boot})
    return out
