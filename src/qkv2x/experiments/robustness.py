"""H2 — detection vs channel degradation.

Trains every model at a reference sweep point and evaluates it at every sweep
point, producing degradation curves (metric vs measured packet loss). Also fits
the  signal-detection model AUC(p_L) to the single discriminative cross-layer
feature and computes the geometric difference g_CQ per attack class as the
explanatory diagnostic for any quantum advantage.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from ..config import load_yaml
from ..models.metrics import (detection_latency, detection_report,
                              grouped_bootstrap_ci, paired_bootstrap_auc_diff)
from ..utils.plotting import bar_per_attack, degradation_curves
from ..utils.provenance import capture_provenance
from .common import (
    CLASSICAL_MODELS,
    _cap_train_groupwise,
    fit_reducer,
    geometric_difference_per_attack,
    group_split,
    load_experiment_data,
    matched_train_idx,
    noise_sensitivity,
    predictions_long,
    run_model,
)


def _point_mask(points: pd.DataFrame, spec: dict) -> np.ndarray:
    mask = np.ones(len(points), dtype=bool)
    for k, v in spec.items():
        if k in points.columns:
            mask &= (points[k].astype(float) == float(v)).to_numpy()
    return mask


def _measured_packet_loss(data, mask, feature="pdr_proxy") -> float:
    if feature in data.feature_names:
        col = data.feature_names.index(feature)
        pdr = np.clip(data.X[mask, col], 0.0, 1.0)
        return float(1.0 - np.mean(pdr))
    return float("nan")


def _single_feature_auc(data, mask, feature="e_xl_mean") -> float:
    col = data.feature_names.index(feature)
    y = data.y[mask]
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, data.X[mask, col]))


def run(config_path: str, cfgs_override: dict | None = None) -> dict:
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

    splits = features_cfg["splits"]
    tr, va, te = group_split(data.groups, data.y, splits["test_frac"],
                             splits["val_frac"], splits["seed"])
    reducer = fit_reducer(features_cfg["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)

    train_point = cfg["train_point"]
    tr_tp = tr[_point_mask(data.points.iloc[tr], train_point)]
    va_tp = va[_point_mask(data.points.iloc[va], train_point)] if len(va) else va
    # Matched-N: cap the train set so quantum + classical compare at the same N (the
    # quantum kernel can't use the full split — O(N^2) Gram). The significance test below
    # is then apples-to-apples; classical@full-data is added afterwards as context (TD-8).
    tr_match = matched_train_idx(tr_tp, data, cfgs)

    # sweep axis = the column that varies (prefer density_scale)
    axis = "density_scale" if "density_scale" in data.points.columns else None
    if axis is None:
        raise ValueError("robustness expects a 'density_scale' sweep axis in the runs.")
    point_values = sorted(data.points[axis].dropna().astype(float).unique())

    rows = []
    pred_rows = []                 # per-window scores for post-hoc curves/tests
    res_by = {}                    # (pv, model) -> result, for paired significance
    for model in cfg["models"]:
        for pv in point_values:
            pmask = (data.points[axis].astype(float) == pv).to_numpy()
            te_p = te[pmask[te]]
            if len(te_p) == 0 or len(np.unique(data.y[te_p])) < 2:
                continue
            idx = {"tr": tr_match, "va": va_tp, "te": te_p}
            res = run_model(model, data, Xr, idx, cfgs)
            res_by[(pv, model)] = res
            pred_rows.extend(predictions_long(res, **{axis: float(pv)}))
            rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
            lat = detection_latency(res["t_end"], res["y_true"], res["y_pred"], res["groups"])
            # grouped-bootstrap 95% CIs on the AUCs (error bars on the single split)
            ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"],
                                      n_boot=n_boot, seed=splits["seed"])
            rows.append({
                "model": model, axis: pv, "n_train": int(len(tr_match)),
                "packet_loss": _measured_packet_loss(data, te_p),
                **{k: rep.get(k) for k in
                   ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall", "fpr_at_tpr"]},
                "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
                "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
                "detection_latency_mean": lat["detection_latency_mean"],
                "detected_fraction": lat["detected_fraction"],
                "n_test": rep["n"], "n_pos": rep["n_pos"],
            })

    # classical@full-data context (the scaling ceiling the O(N^2) quantum kernel can't
    # reach). Tagged '<model>__full'; not part of the matched significance test above.
    # Bounded by `full_cap` (default 15k) so SVC-RBF stays tractable on CPU (n_train shows N).
    tr_full = _cap_train_groupwise(tr_tp, data.groups, int(cfg.get("full_cap", 15000)),
                                   splits["seed"])
    if len(tr_match) < len(tr_full):
        for model in [m for m in cfg["models"] if m in CLASSICAL_MODELS]:
            for pv in point_values:
                pmask = (data.points[axis].astype(float) == pv).to_numpy()
                te_p = te[pmask[te]]
                if len(te_p) == 0 or len(np.unique(data.y[te_p])) < 2:
                    continue
                res = run_model(model, data, Xr,
                                {"tr": tr_full, "va": va_tp, "te": te_p}, cfgs)
                rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
                rows.append({
                    "model": f"{model}__full", axis: pv, "n_train": int(len(tr_full)),
                    "packet_loss": _measured_packet_loss(data, te_p),
                    **{k: rep.get(k) for k in
                       ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall", "fpr_at_tpr"]},
                    "n_test": rep["n"], "n_pos": rep["n_pos"]})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "robustness_results.csv", index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_parquet(out_dir / "predictions.parquet")

    # --- significance: each quantum model vs the RBF-SVM baseline, per point -------
    # grouped paired bootstrap on the SAME test windows (te_p identical across models).
    sig_rows = []
    for pv in point_values:
        ref = res_by.get((pv, "rbf_svm"))
        if ref is None:
            continue
        for qm in ("qsvm_crosslayer", "qsvm_zz", "qsvm_projected"):
            r = res_by.get((pv, qm))
            if r is None:
                continue
            for metric in ("roc_auc", "pr_auc"):
                d = paired_bootstrap_auc_diff(r["y_true"], r["scores"], ref["scores"],
                                              r["groups"], n_boot=n_boot,
                                              seed=splits["seed"], metric=metric)
                sig_rows.append({axis: pv, "model": qm, "vs": "rbf_svm", **d})
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(out_dir / "significance.csv", index=False)

    # ---  signal-detection model fit on the cross-layer feature ----------
    deg = {}
    if cfg.get("fit_degradation_model", True):
        W = float(features_cfg["window"]["size"])
        pls, aucs = [], []
        for pv in point_values:
            pmask = (data.points[axis].astype(float) == pv).to_numpy()
            te_p = te[pmask[te]]
            if len(te_p) == 0:
                continue
            pls.append(_measured_packet_loss(data, te_p))
            aucs.append(_single_feature_auc(data, te_p))
        pls = np.asarray(pls)
        aucs = np.asarray(aucs)
        ok = np.isfinite(pls) & np.isfinite(aucs)

        def model_auc(pL, dmu, sigma0_2, beta):
            dprime = np.abs(dmu) / np.sqrt(sigma0_2 + beta / np.maximum((1 - pL) * W, 1e-6))
            return norm.cdf(dprime / np.sqrt(2.0))

        try:
            popt, _ = curve_fit(model_auc, pls[ok], aucs[ok],
                                p0=[2.0, 0.5, 1.0], maxfev=10000,
                                bounds=([0, 1e-6, 0], [np.inf, np.inf, np.inf]))
            grid = np.linspace(float(np.min(pls[ok])), float(np.max(pls[ok])), 50)
            deg = {"status": "fit", "params": {"d_mu": popt[0], "sigma0_2": popt[1],
                                               "beta": popt[2]},
                   "packet_loss": pls[ok].tolist(), "empirical_auc": aucs[ok].tolist(),
                   "predicted_grid_pL": grid.tolist(),
                   "predicted_auc": model_auc(grid, *popt).tolist()}
        except Exception as exc:  # report non-convergence rather than fabricate
            deg = {"status": f"fit_failed: {exc}",
                   "packet_loss": pls[ok].tolist(), "empirical_auc": aucs[ok].tolist()}

    # --- g_CQ per attack class (H2 explanation) -----------------------------
    gcq = {}
    if cfg.get("per_point_diagnostics", True):
        gcq = geometric_difference_per_attack(
            data, Xr, {"tr": tr_tp, "va": va_tp, "te": te}, cfgs, seed=splits["seed"])

    # ---  hardware-noise sensitivity (finite shots + depolarising) ------
    # Runs when the experiment sets `noise_sensitivity: true` OR quantum.yaml enables
    # diagnostics.depolarizing. Cheap (analytic contraction of a subsampled Gram).
    noise = {}
    dep = (cfgs["quantum"].get("diagnostics", {}) or {}).get("depolarizing", {}) or {}
    if cfg.get("noise_sensitivity", dep.get("enabled", False)):
        noise = noise_sensitivity(
            data, Xr, {"tr": tr_tp, "te": te}, cfgs,
            max_samples=int(dep.get("max_samples", 300)), seed=splits["seed"])

    diagnostics = {"degradation_model": deg,
                   "geometric_difference_per_attack": gcq,
                   "noise_sensitivity": noise}
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    # --- figures ------------------------------------------------------------
    if not results.empty:
        overlay = None
        if deg.get("status") == "fit":
            overlay = (deg["predicted_grid_pL"], deg["predicted_auc"],
                       "single-feature model AUC(p_L)")
        degradation_curves(results, "packet_loss", "pr_auc", "model",
                           out_dir / "pr_auc_vs_loss.png",
                           "PR-AUC vs packet loss", ci=("pr_auc_lo", "pr_auc_hi"))
        # The  single-feature model predicts an ROC-AUC (Gaussian SDT,
        # norm.cdf(d'/sqrt2)), so overlay it on the ROC-AUC degradation curve.
        degradation_curves(results, "packet_loss", "roc_auc", "model",
                           out_dir / "roc_auc_vs_loss.png",
                           "ROC-AUC vs packet loss", overlay=overlay,
                           ci=("roc_auc_lo", "roc_auc_hi"))
    # Always emit the artifact README.md claims; an empty dict (no attack class
    # cleared the per-class window floor at the reference point) yields an annotated
    # empty chart rather than a silently missing file.
    bar_per_attack(gcq, "g_cq", out_dir / "g_cq_per_attack.png",
                   "Geometric difference per attack", "g_CQ")
    # : ROC-AUC vs simulated global-depolarising noise p (kernel concentration).
    if noise.get("status") == "ok" and noise.get("depolarizing"):
        dn = pd.DataFrame(noise["depolarizing"])
        dn["model"] = "qsvm_crosslayer"
        degradation_curves(dn, "p", "roc_auc", "model",
                           out_dir / "roc_auc_vs_depolarizing.png",
                           "ROC-AUC vs depolarising noise p")

    elapsed = round(time.time() - t0, 1)
    summary = {"out_dir": str(out_dir), "n_rows": len(results),
               "models": cfg["models"], "points": point_values,
               "n_bootstrap": n_boot, "wall_time_s": elapsed,
               "noise_sensitivity": noise.get("status", "off"),
               "artifacts": ["robustness_results.csv", "predictions.parquet",
                             "significance.csv", "diagnostics.json",
                             "pr_auc_vs_loss.{png,pdf}", "roc_auc_vs_loss.{png,pdf}",
                             "g_cq_per_attack.{png,pdf}",
                             "roc_auc_vs_depolarizing.{png,pdf}", "provenance.json"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    capture_provenance(out_dir, timestamp=started, config_path=config_path,
                       extra={"experiment": "robustness", "wall_time_s": elapsed,
                              "split_seed": splits["seed"], "n_bootstrap": n_boot})
    return summary
