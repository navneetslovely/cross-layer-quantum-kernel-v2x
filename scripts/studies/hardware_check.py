#!/usr/bin/env python3
"""hardware_check — QPU stage 1 of the (): analysis of the 50-window
hardware execution check on ibm_marrakesh (pinned layout, 1,024 shots).

Inputs: results/dev_grid_multiseed/ibm_hardware/devgrid_hw_ibm_marrakesh_s7_n50_trgram_*.json
(Run A: 50x12 test block + 12x12 training Gram; Run B: repeat on the same windows, train Gram
optional). Per run and pooled: kernel agreement (Pearson r, contraction slope) on the 600 test
entries; paired hardware-vs-statevector ROC-AUC on identical windows (DeLong paired test and a
grouped bootstrap over transmitters); run-to-run agreement; the hardware-only scoring chain
(SVM refit on the measured training Gram). Also folds in the three earlier 10-window runs and
the NextGen correlations for the pooled statements used in the paper.

Outputs: results/studies/hardware_check_summary.json, hardware_check_runs.csv.
"""
from __future__ import annotations

import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
HW = REPO / "results" / "dev_grid_multiseed" / "ibm_hardware"
OUT = REPO / "results" / "studies"


# ---------- DeLong paired test (Sun & Xu 2014 fast implementation) ----------
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N); T2[J] = T + 1
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: [k, n] scores with the m positives first."""
    n = preds_sorted.shape[1] - m; k = preds_sorted.shape[0]
    tx = np.array([_compute_midrank(preds_sorted[r, :m]) for r in range(k)])
    ty = np.array([_compute_midrank(preds_sorted[r, m:]) for r in range(k)])
    tz = np.array([_compute_midrank(preds_sorted[r, :]) for r in range(k)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    return aucs, sx / m + sy / n


def delong_paired(y, s_a, s_b):
    y = np.asarray(y).astype(int); order = np.argsort(-y); m = int(y.sum())
    preds = np.vstack([np.asarray(s_a)[order], np.asarray(s_b)[order]])
    aucs, cov = _fast_delong(preds, m)
    d = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = d / np.sqrt(var) if var > 0 else np.nan
    from scipy.stats import norm
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "delta": float(d),
            "se": float(np.sqrt(var)) if var > 0 else None, "z": float(z) if np.isfinite(z) else None, "p": p}


def grouped_paired_bootstrap(y, s_a, s_b, groups, n_boot=2000, seed=7):
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed); y = np.asarray(y); g = np.asarray(groups)
    ug = np.unique(g); idx_by_g = {k: np.where(g == k)[0] for k in ug}
    deltas = []
    for _ in range(n_boot):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by_g[k] for k in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        deltas.append(roc_auc_score(y[idx], np.asarray(s_a)[idx]) - roc_auc_score(y[idx], np.asarray(s_b)[idx]))
    d = np.asarray(deltas)
    return {"delta_mean": float(d.mean()), "lo": float(np.percentile(d, 2.5)), "hi": float(np.percentile(d, 97.5)),
            "p_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())), "n_boot": int(len(d))}


def main():
    t0 = time.time()
    files = sorted(glob.glob(str(HW / "devgrid_hw_ibm_marrakesh_s7_n50*_*.json")))
    if not files:
        sys.exit("no stage-1 run files found")
    from scipy.stats import pearsonr
    from sklearn.metrics import roc_auc_score
    from qkv2x.models.qsvm import QuantumKernelSVM
    from qkv2x.quantum.kernels import psd_clip
    df = pd.read_parquet(REPO / "runs" / "dev_grid" / "features.parquet", columns=["group"])
    groups_all = df["group"].to_numpy()
    runs, per = [], {}
    for f in files:
        d = json.load(open(f))
        K_hw, K_sim = np.asarray(d["K_hw"]), np.asarray(d["K_te_sim"])
        y_te = np.asarray(d["detection"]["y_test"]); y_tr = np.asarray(d["y_train"])
        te_idx = np.asarray(d["test_idx"]); g = groups_all[te_idx]
        K_tr_sim = np.asarray(d["K_tr_sim"])
        svm = QuantumKernelSVM(C_grid=[0.1, 1.0, 10.0, 100.0], cv_folds=3); svm.fit(K_tr_sim, y_tr)
        s_hw, s_sim = svm.decision_function(K_hw), svm.decision_function(K_sim)
        hw, sim = K_hw.ravel(), K_sim.ravel()
        r = float(pearsonr(hw, sim)[0]); a, b = np.polyfit(sim, hw, 1)
        dl = delong_paired(y_te, s_hw, s_sim); gb = grouped_paired_bootstrap(y_te, s_hw, s_sim, g)
        row = {"file": Path(f).name, "n_test": int(d["n_test"]), "n_train": int(d["n_train"]),
               "n_entries": int(hw.size), "shots": d["shots"], "qpu_usage": d.get("qpu_usage_s"),
               "pearson_r": r, "slope": float(a), "intercept": float(b),
               "roc_hw": float(roc_auc_score(y_te, s_hw)), "roc_sim": float(roc_auc_score(y_te, s_sim)),
               "delong_delta": dl["delta"], "delong_p": dl["p"], "boot_lo": gb["lo"], "boot_hi": gb["hi"], "boot_p": gb["p_two_sided"],
               "n_groups": int(len(np.unique(g))), "train_gram_measured": d.get("K_tr_hw") is not None}
        if d.get("K_tr_hw") is not None:
            K_tr_hw = np.asarray(d["K_tr_hw"])
            iu = np.triu_indices(len(K_tr_hw), 1)
            row["train_gram_r"] = float(pearsonr(K_tr_hw[iu], K_tr_sim[iu])[0])
            svm_hw = QuantumKernelSVM(C_grid=[0.1, 1.0, 10.0, 100.0], cv_folds=3); svm_hw.fit(psd_clip(K_tr_hw), y_tr)
            row["roc_hw_chain"] = float(roc_auc_score(y_te, svm_hw.decision_function(K_hw)))
            row["hw_chain_C"] = svm_hw.best_C_
        runs.append(row); per[Path(f).name] = {"s_hw": s_hw, "s_sim": s_sim, "y": y_te, "g": g, "K_hw": K_hw, "K_sim": K_sim, "te_idx": te_idx}
    rdf = pd.DataFrame(runs); rdf.to_csv(OUT / "hardware_check_runs.csv", index=False)
    summ = {"runs": runs, "n_runs": len(runs)}
    # pooled kernel agreement and run-to-run agreement (same windows)
    if len(per) >= 1:
        allhw = np.concatenate([v["K_hw"].ravel() for v in per.values()]); allsim = np.concatenate([v["K_sim"].ravel() for v in per.values()])
        summ["pooled_r"] = float(pearsonr(allhw, allsim)[0]); summ["pooled_slope"] = float(np.polyfit(allsim, allhw, 1)[0])
        summ["r_mean"] = float(rdf.pearson_r.mean()); summ["r_sd"] = float(rdf.pearson_r.std(ddof=1)) if len(rdf) > 1 else None
        summ["roc_hw_mean"] = float(rdf.roc_hw.mean()); summ["roc_hw_sd"] = float(rdf.roc_hw.std(ddof=1)) if len(rdf) > 1 else None
        summ["roc_sim"] = float(rdf.roc_sim.iloc[0])
    keys = list(per)
    if len(keys) >= 2:
        same = all(np.array_equal(per[keys[0]]["te_idx"], per[k]["te_idx"]) for k in keys[1:])
        summ["same_windows_across_runs"] = bool(same)
        if same:
            summ["run_to_run_kernel_r"] = float(pearsonr(per[keys[0]]["K_hw"].ravel(), per[keys[1]]["K_hw"].ravel())[0])
            summ["run_to_run_score_r"] = float(pearsonr(per[keys[0]]["s_hw"], per[keys[1]]["s_hw"])[0])
            # mean-of-runs hardware scores vs simulation (paired)
            s_mean = np.mean([per[k]["s_hw"] for k in keys], axis=0)
            summ["mean_of_runs_vs_sim"] = {**delong_paired(per[keys[0]]["y"], s_mean, per[keys[0]]["s_sim"]),
                                           "grouped_bootstrap": grouped_paired_bootstrap(per[keys[0]]["y"], s_mean, per[keys[0]]["s_sim"], per[keys[0]]["g"])}
    # fold in the earlier 10-window dev_grid runs and the NextGen correlations
    old = sorted(glob.glob(str(HW / "devgrid_hw_ibm_marrakesh_s7_1*.json")))
    old_r, old_roc = [], []
    for f in old:
        d = json.load(open(f))
        if d.get("n_test") == 10:
            old_r.append(d["kernel_agreement"]["pearson_r"]); old_roc.append(d["detection"]["roc_auc_hw"])
    summ["earlier_10_window_runs"] = {"n": len(old_r), "r": old_r, "roc_hw": old_roc}
    hardware_intervals = OUT / "hardware_intervals_summary.json"
    if hardware_intervals.exists():
        f22 = json.load(open(hardware_intervals)); col = f22.get("default_layout_collapse_reproduction", {})
        pinned = col.get("pinned_r", []); default = col.get("repeat_default_r", [])
        allr = list(pinned) + list(default)
        summ["nextgen_r"] = {"nine_runs_mean": float(np.mean(allr)) if allr else None, "nine_runs_sd": float(np.std(allr, ddof=1)) if len(allr) > 1 else None,
                             "pooled_with_outlier_mean": float(np.mean(allr + [col.get("published_default_r", 0.481)])) if allr else None,
                             "pooled_with_outlier_sd": float(np.std(allr + [col.get("published_default_r", 0.481)], ddof=1)) if allr else None}
    summ["protocol"] = {"backend": "ibm_marrakesh", "initial_layout": [33, 34, 35, 39, 51, 52, 53, 58], "shots": 1024,
                        "account": "redacted", "n_test": 50, "n_train": 12, "train_gram": "upper triangle, 66 circuits (Run A)",
                        "surrogate_note": "12-window support set refit; the main detector (888 SVs) is not scored on hardware",
                        "paired_tests": "DeLong (Sun & Xu 2014) and grouped bootstrap over transmitter groups (B=2000)",
                        "interpreter": sys.executable, "runtime_s": time.time() - t0}
    (OUT / "hardware_check_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(rdf.round(4).to_string(index=False)); print(json.dumps({k: v for k, v in summ.items() if k not in ("runs",)}, indent=1, default=float)[:3000])


if __name__ == "__main__":
    main()
