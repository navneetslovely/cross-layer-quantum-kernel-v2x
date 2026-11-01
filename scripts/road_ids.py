"""ROAD CAN-IDS DETECTION STUDY — benign-vs-attack evaluation of the full model zoo.

Detection study on the ROAD CAN-bus corpus; the loader / feature / window / split code is in
scripts/road_data.py.
It mirrors src/qkv2x/experiments/veremi_exp.py::run: a standard binary intrusion-detection
evaluation (benign window vs. attack-injected window) on a leakage-safe grouped split,
with grouped (by capture) bootstrap CIs.

Models (experiments.common.run_model, CPU-only statevector quantum kernels):
  qsvm_crosslayer  qsvm_zz  rbf_svm  random_forest  xgboost  mlp

Headline checks (reported honestly):
  (a) cross-layer vs ZZ : does the trained cross-layer quantum kernel beat the untrained
      ZZ ablation? (cross-layer >> ZZ expected, as on the 3 V2X datasets.)
  (b) no-advantage check: quantum (best of qsvm_*) vs best classical (RF / XGBoost).
      Expectation, consistent with the whole project: NO quantum advantage.

ROAD has NO GPS/RSSI/V2X PHY. We use the PRINCIPLED CAN two-view (defensible information
split, see road_data.py docstring): register A = WHEN (timing/arrival/rate) x register B
= WHAT (payload/signal-content/ID-diversity) — the in-vehicle analog of mobility x network.
This is the same trained machine on a motivated CAN split, not a literal cross-layer deploy.

Windows are capped in the low thousands (quantum Gram is O(N^2)).

Outputs (results/road_ids/):
  road_detection.csv  per-model PR-AUC / ROC-AUC / F1 / MCC / precision / recall + grouped
                      bootstrap CIs (pr_auc_lo/hi, roc_auc_lo/hi), n_test / n_pos
  metadata.json  seeds, caps, models, env, headline-check verdicts, honest framing

Run (CPU-only, no QPU/GPU):
  python scripts/road_ids.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from _study_lib import RESULTS, now_iso, save_table, split_meta, write_metadata  # noqa: E402
import road_data as R  # noqa: E402  (loader / features / splits)

MODELS = ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest", "xgboost", "mlp"]


def _balance_for_quantum(data, max_windows: int, seed: int):
    """Subsample to <= max_windows, group-disjoint and class-stratified, so the O(N^2)
    quantum Gram is affordable. Keeps WHOLE captures (groups) intact (leakage-safe) and
    targets a manageable positive/negative balance. Returns a boolean mask over rows.
    """
    rng = np.random.default_rng(seed)
    n = len(data.y)
    if n <= max_windows:
        return np.ones(n, dtype=bool)
    # keep all attack (positive) windows up to half the budget, then fill with benign
    pos = np.where(data.y == 1)[0]
    neg = np.where(data.y == 0)[0]
    n_pos = min(len(pos), max_windows // 2)
    n_neg = min(len(neg), max_windows - n_pos)
    sel_pos = rng.choice(pos, n_pos, replace=False)
    sel_neg = rng.choice(neg, n_neg, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[sel_pos] = True
    mask[sel_neg] = True
    return mask


def run(window_s: float = 0.25, min_msgs: int = 15, seed: int = 7,
        max_windows: int = 2400, n_boot: int = 300,
        max_attack_win: int = 700, max_benign_win: int = 600) -> dict:
    from qkv2x.experiments.common import fit_reducer, run_model
    from qkv2x.models.metrics import detection_report, grouped_bootstrap_ci

    t0 = time.time()
    started = now_iso()
    out_dir = RESULTS / "road_ids"
    out_dir.mkdir(parents=True, exist_ok=True)

    cfgs = R._build_cfgs(seed)
    data, cap_info = R.build_road_dataset(
        window_s=window_s, min_msgs=min_msgs,
        max_attack_win_per_capture=max_attack_win,
        max_benign_win_per_capture=max_benign_win, seed=seed)

    # cap windows for the O(N^2) quantum Gram (leakage-safe, class-stratified)
    mask = _balance_for_quantum(data, max_windows, seed)
    data = data.subset(mask)
    fams = sorted(set(data.attack_name) - {"none"})

    # leakage-safe family-aware split (every family in train+test; benign by capture)
    tr, va, te = R.family_aware_split(data, test_frac=0.30, val_frac=0.15, seed=seed)
    reducer = fit_reducer(cfgs["features"]["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)
    idx = {"tr": tr, "va": va, "te": te}

    print("[road-ids] windows=%d pos=%d (%.1f%%) captures=%d families=%s" % (
        len(data), int(data.y.sum()), 100 * data.y.mean(),
        len(set(data.groups)), fams))
    print("[road-ids] split tr/va/te = %d/%d/%d  pos = %d/%d/%d" % (
        len(tr), len(va), len(te),
        int(data.y[tr].sum()), int(data.y[va].sum()), int(data.y[te].sum())))

    rows = []
    for model in MODELS:
        try:
            res = run_model(model, data, Xr, idx, cfgs)
        except Exception as exc:
            print("  %s FAILED: %s" % (model, exc))
            rows.append({"model": model, "status": f"failed: {type(exc).__name__}: {exc}"})
            continue
        rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
        ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"],
                                  n_boot=n_boot, seed=seed)
        rows.append({
            "model": model,
            "pr_auc": rep.get("pr_auc"), "roc_auc": rep.get("roc_auc"),
            "f1": rep.get("f1"), "mcc": rep.get("mcc"),
            "precision": rep.get("precision"), "recall": rep.get("recall"),
            "fpr_at_tpr95": rep.get("fpr_at_tpr"),
            "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
            "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
            "n_test": rep["n"], "n_pos": rep["n_pos"]})
        print("  %-16s PR=%.3f ROC=%.3f F1=%.3f MCC=%.3f  (ROC CI %.3f-%.3f)" % (
            model, rep.get("pr_auc", float("nan")), rep.get("roc_auc", float("nan")),
            rep.get("f1", float("nan")), rep.get("mcc", float("nan")),
            ci["roc_auc"]["lo"], ci["roc_auc"]["hi"]))

    df = pd.DataFrame(rows)
    save_table(df, out_dir, "road_detection.csv")

    # --- headline-check verdicts (honest, from the numbers) ---
    def _get(m, k):
        r = df[df["model"] == m]
        return float(r[k].iloc[0]) if len(r) and pd.notna(r[k].iloc[0]) else None

    cl_roc, zz_roc = _get("qsvm_crosslayer", "roc_auc"), _get("qsvm_zz", "roc_auc")
    cl_pr, zz_pr = _get("qsvm_crosslayer", "pr_auc"), _get("qsvm_zz", "pr_auc")
    best_classical = None
    best_cl_val = -1.0
    for m in ("random_forest", "xgboost", "rbf_svm", "mlp"):
        v = _get(m, "roc_auc")
        if v is not None and v > best_cl_val:
            best_cl_val, best_classical = v, m
    best_quantum = None
    best_q_val = -1.0
    for m in ("qsvm_crosslayer", "qsvm_zz"):
        v = _get(m, "roc_auc")
        if v is not None and v > best_q_val:
            best_q_val, best_quantum = v, m

    cross_beats_zz = (cl_roc is not None and zz_roc is not None and cl_roc > zz_roc)
    q_minus_c = (best_q_val - best_cl_val) if (best_q_val >= 0 and best_cl_val >= 0) else None
    headline = {
        "crosslayer_vs_zz": {
            "qsvm_crosslayer_roc": cl_roc, "qsvm_zz_roc": zz_roc,
            "qsvm_crosslayer_pr": cl_pr, "qsvm_zz_pr": zz_pr,
            "crosslayer_beats_zz": bool(cross_beats_zz),
            "roc_gap_cl_minus_zz": (None if (cl_roc is None or zz_roc is None)
                                    else float(cl_roc - zz_roc)),
            "verdict": ("cross-layer >> ZZ holds (trained cross-layer kernel beats untrained "
                        "ZZ ablation)" if cross_beats_zz else
                        "cross-layer does NOT beat ZZ here — reported verbatim")},
        "no_advantage_check": {
            "best_quantum_model": best_quantum, "best_quantum_roc": best_q_val,
            "best_classical_model": best_classical, "best_classical_roc": best_cl_val,
            "quantum_minus_classical_roc": q_minus_c,
            "verdict": ("NO quantum advantage: best classical (%s) >= best quantum (%s) on ROC"
                        % (best_classical, best_quantum)
                        if (q_minus_c is not None and q_minus_c <= 0) else
                        "best quantum exceeds best classical on ROC by %s — reported verbatim"
                        % (None if q_minus_c is None else round(q_minus_c, 4)))},
    }
    print("\n[road-ids] cross-layer ROC=%s vs ZZ ROC=%s -> %s" % (
        cl_roc, zz_roc, headline["crosslayer_vs_zz"]["verdict"]))
    print("[road-ids] best quantum=%s(%.3f) vs best classical=%s(%.3f) -> %s" % (
        best_quantum, best_q_val, best_classical, best_cl_val,
        headline["no_advantage_check"]["verdict"]))

    write_metadata(
        out_dir, study="road_ids_detection", started_iso=started, t0=t0,
        config_path="configs/quantum.yaml + configs/baselines.yaml (in-script features cfg)",
        hyperparams={"window_s": window_s, "min_msgs": min_msgs, "seed": seed,
                     "max_windows": max_windows, "n_bootstrap": n_boot,
                     "max_attack_win_per_capture": max_attack_win,
                     "max_benign_win_per_capture": max_benign_win,
                     "models": MODELS, "n_qubits": 8, "reduction": "per_block_pca 4+4",
                     "split": "family_aware_split (leakage-safe; benign by capture)"},
        splits=split_meta(data.y, idx),
        results_summary={"detection_rows": rows, "headline_checks": headline},
        notes=("ROAD CAN-IDS detection study (the repurpose). Benign-vs-attack window "
               "detection with the full model zoo on a leakage-safe family-aware split; "
               "grouped-by-capture bootstrap CIs. PRINCIPLED CAN two-view (WHEN x WHAT; the "
               "in-vehicle analog of V2X mobility x network) — same trained machine, NOT a "
               "literal cross-layer deployment (CAN has no PHY/RSSI to invert). Headline "
               "verdicts (cross-layer-vs-ZZ, quantum-vs-best-classical) computed from the "
               "numbers and reported verbatim. accelerator_attack_* "
               "captures excluded (no labeled injection)."),
        extra={"captures": cap_info, "attack_families": fams,
               "command": "python scripts/road_ids.py"},
    )

    print("\n[road-ids] wrote:")
    for f in ("road_detection.csv", "metadata.json"):
        print("  ", out_dir / f)
    return {"rows": rows, "headline": headline}


if __name__ == "__main__":
    run()
