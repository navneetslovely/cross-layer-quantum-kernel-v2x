#!/usr/bin/env python3
"""baseline_fairness -- baseline-fairness controlled comparisons + hyperparameter inventory.

Establishes whether the quantum and classical models use the same 3,023 windows, the
same eight PCA coordinates, the same original 19 features, the same train-validation-test
groups, and comparable tuning budgets. Reports three controlled comparisons (same sample
count with the same eight-dimensional input; same sample count with the full
19-dimensional input where possible; full-data classical scaling) and the full
hyperparameter table of the classical baselines.

WHERE TABLE I LIVES
-------------------
The paper main-detection table (tab:rq1)
is the packet_loss=0.0 slice of results/packetloss_dev/packetloss_results.csv
(src/qkv2x/experiments/packetloss.py + configs/experiments/packetloss.yaml):
matched rows n_train=3,023 / n_test=26,392 / n_pos=5,752; "(full N)" rows n_train=15,048.
The 3-seed replication of the matched comparison is results/dev_grid_multiseed/
(scripts/dev_multiseed.py).

WHAT THIS SCRIPT DOES (read-only w.r.t. src/, configs/, existing results)
-------------------------------------------------------------------------
1. FAIRNESS FACTS, answered from the code with file:line citations AND verified
   empirically by reconstructing the exact Table-I split from
   runs/dev_grid/features.parquet (identical loader to scripts/dev_multiseed.py):
   shared split / shared 8-dim PCA input / matched train subset identity (including
   the run_model-internal re-cap that the quantum branch applies) / what the
   "(full N)" rows actually change (BOTH N and the input representation).
2. Three controlled comparisons:
   (i)  same-N (3,023) same-8-dim input  -> pulled from packetloss_results.csv p=0
        (+ 3-seed mean+-std context from results/dev_grid_multiseed/multiseed.csv);
   (ii) same-N (3,023) full-19-raw-feature classical -> DOES NOT EXIST anywhere in
        results/ (searched all CSVs for __full rows with n_train<10k on dev_grid;
        only 15,048/15,008/15,002/15,005/8,959 = the full_cap rows exist), so it is
        RUN HERE: rbf_svm + random_forest + xgboost via the exact run_model classical
        path (build_baseline, seed 17, 5-fold stratified grid-CV, scoring
        average_precision) on the SAME 3,023-window train subset and the SAME
        26,392-window test split, 19 raw features standardised on train only;
   (iii) full-data classical scaling -> pulled from packetloss_results.csv p=0
        (__full rows, n_train=15,048, 19 raw features).
   A reconstruction check re-runs matched rbf_svm on the reconstructed 8-dim input and
   compares to the published row, validating the split/reducer/cap reconstruction.
3. HYPERPARAMETER INVENTORY -> hyperparams.csv: every model in the paper, the
   search grid / fixed values ACTUALLY used, selection method, and source file.

Determinism: split seed 7 (configs/features.yaml::splits.seed), train-cap seed 7,
baseline seed 17 (configs/baselines.yaml::seed), bootstrap n=200 seed 7 -- identical
to the published packetloss run. All choices recorded under "protocol" in the JSON.

Usage:
    python scripts/studies/baseline_fairness.py \
        2>&1 | tee results/studies/baseline_fairness_run.log

Outputs (results/studies/):
    baseline_fairness.csv  code-verified fairness facts (fact, answer, evidence, file:line)
    baseline_comparisons.csv  the three controlled comparisons, one row per (comparison, model)
    hyperparams.csv  full hyperparameter inventory
    baseline_fairness_summary.json  protocol + facts + comparisons + headline
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RESULTS = REPO / "results"
OUT = RESULTS / "studies"
OUT.mkdir(parents=True, exist_ok=True)

from qkv2x.config import load_yaml                                   # noqa: E402
from qkv2x.experiments.common import (ExperimentData,                # noqa: E402
                                      _cap_train_groupwise,
                                      fit_reducer, group_split)
from qkv2x.features.crosslayer import FEATURE_SPEC                   # noqa: E402
from qkv2x.models.baselines import baseline_scores, build_baseline   # noqa: E402
from qkv2x.models.metrics import detection_report, grouped_bootstrap_ci  # noqa: E402

PARQUET = REPO / "runs" / "dev_grid" / "features.parquet"
PUBLISHED = RESULTS / "packetloss_dev" / "packetloss_results.csv"
MULTISEED = RESULTS / "dev_grid_multiseed" / "multiseed.csv"

N_BOOT = 200           # identical to configs/experiments/packetloss.yaml::bootstrap
BOOT_SEED = 7          # identical to packetloss.py (seed = splits.seed)
BASELINE_SEED = 17     # configs/baselines.yaml::seed, used by run_model's build_baseline
NEW_MODELS = ["rbf_svm", "random_forest", "xgboost"]   # comparison (ii), per baseline_fairness spec


# --------------------------------------------------------------------------- #
# Data reconstruction (identical loader to scripts/dev_multiseed.py::_load)
# --------------------------------------------------------------------------- #
def _load() -> ExperimentData:
    df = pd.read_parquet(PARQUET)
    names = [s[0] for s in FEATURE_SPEC]
    blocks = [s[3] for s in FEATURE_SPEC]
    return ExperimentData(
        X=df[names].to_numpy(), y=df["y"].to_numpy(),
        attack_name=df["attack_name"].to_numpy(object),
        groups=df["group"].to_numpy(), aoi=df["aoi"].to_numpy(),
        t_end=df["t_end"].to_numpy(), feature_names=names, blocks=blocks,
        points=pd.DataFrame(index=range(len(df))))


def _report_row(model, tag, y_true, scores, preds, groups, n_train, n_features,
                input_repr, source):
    rep = detection_report(y_true, scores, preds)
    ci = grouped_bootstrap_ci(y_true, scores, groups, n_boot=N_BOOT, seed=BOOT_SEED)
    return {
        "comparison": tag, "model": model, "n_train": int(n_train),
        "n_features": int(n_features), "input_repr": input_repr,
        "pr_auc": rep["pr_auc"], "roc_auc": rep["roc_auc"],
        "f1": rep["f1"], "mcc": rep["mcc"],
        "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
        "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
        "n_test": rep["n"], "n_pos": rep["n_pos"], "source": source,
    }


def main() -> None:
    t0 = time.time()
    feats = load_yaml(str(REPO / "configs/features.yaml"))
    qcfg = load_yaml(str(REPO / "configs/quantum.yaml"))
    bcfg = load_yaml(str(REPO / "configs/baselines.yaml"))
    splits = feats["splits"]
    split_seed = int(splits["seed"])                       # 7
    max_train = int(qcfg.get("max_train", 3000))           # 3000
    full_cap = 15000                                       # configs/experiments/packetloss.yaml

    # ------------------------------------------------------------------ #
    # 1. Reconstruct the exact Table-I split and verify invariants
    # ------------------------------------------------------------------ #
    data = _load()
    n_windows = len(data)
    print(f"[baseline_fairness] dataset: {n_windows} windows, {data.X.shape[1]} raw features, "
          f"{100 * float(np.mean(data.y)):.2f}% positive")
    assert data.X.shape[1] == 19, "expected 19 raw features (FEATURE_SPEC)"

    tr, va, te = group_split(data.groups, data.y, splits["test_frac"],
                             splits["val_frac"], split_seed)
    # leakage sanity: group-disjoint splits
    g_tr, g_va, g_te = (set(data.groups[i]) for i in (tr, va, te))
    assert not (g_tr & g_te) and not (g_tr & g_va) and not (g_va & g_te), \
        "splits share groups -- leakage"

    # exactly what packetloss.py:75 does (matched_train_idx -> _cap_train_groupwise)
    tr_match = _cap_train_groupwise(tr, data.groups, max_train, split_seed)
    # exactly what run_model does INTERNALLY for quantum models (common.py:292-295):
    tr_q = _cap_train_groupwise(tr_match, data.groups, max_train, split_seed)
    identical_train = bool(np.array_equal(np.sort(tr_match), np.sort(tr_q)))
    dropped = sorted(set(tr_match.tolist()) - set(tr_q.tolist()))
    print(f"[baseline_fairness] split: train={len(tr)} val={len(va)} test={len(te)} | "
          f"matched train={len(tr_match)} (groups={len(set(data.groups[tr_match]))}) | "
          f"quantum inner re-cap={len(tr_q)} identical={identical_train} "
          f"(dropped {len(dropped)} windows)")

    invariants = {
        "n_windows": n_windows, "n_windows_expected": 90553,
        "n_test": int(len(te)), "n_test_expected": 26392,
        "n_test_pos": int(data.y[te].sum()), "n_test_pos_expected": 5752,
        "n_train_matched": int(len(tr_match)), "n_train_matched_expected": 3023,
    }
    recon_ok = (invariants["n_windows"] == 90553 and invariants["n_test"] == 26392
                and invariants["n_test_pos"] == 5752
                and invariants["n_train_matched"] == 3023)
    print(f"[baseline_fairness] invariants vs published Table I: {invariants} -> "
          f"{'MATCH' if recon_ok else 'MISMATCH'}")
    if not recon_ok:
        raise SystemExit("reconstructed split does not match the published Table I "
                         "counts -- refusing to continue (would compare unlike things)")

    # shared 8-dim PCA input: reducer fit on the UNCAPPED train split (packetloss.py:73)
    reducer = fit_reducer(feats["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)
    print(f"[baseline_fairness] reduced input: {Xr.shape[1]} dims "
          f"({feats['reduction']['n_mobility']} mobility + "
          f"{feats['reduction']['n_network']} network, per-block PCA, [0,pi]-scaled)")
    assert Xr.shape[1] == 8

    # ------------------------------------------------------------------ #
    # 2. Published comparisons (i) and (iii) from the Table-I source CSV
    # ------------------------------------------------------------------ #
    pub = pd.read_csv(PUBLISHED)
    pub0 = pub[pub["packet_loss"] == 0.0].copy()
    matched_pub = pub0[~pub0["model"].str.endswith("__full")]
    full_pub = pub0[pub0["model"].str.endswith("__full")]
    assert (matched_pub["n_train"] == 3023).all()
    assert (full_pub["n_train"] == 15048).all()
    assert (pub0["n_test"] == 26392).all() and (pub0["n_pos"] == 5752).all(), \
        "published rows do not share one test split"

    comp_rows = []
    for _, r in matched_pub.iterrows():
        comp_rows.append({
            "comparison": "i_sameN_8dim", "model": r["model"], "n_train": 3023,
            "n_features": 8, "input_repr": "per-block PCA 4+4, [0,pi]-scaled",
            "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"], "f1": r["f1"],
            "mcc": r["mcc"], "pr_auc_lo": r["pr_auc_lo"], "pr_auc_hi": r["pr_auc_hi"],
            "roc_auc_lo": r["roc_auc_lo"], "roc_auc_hi": r["roc_auc_hi"],
            "n_test": 26392, "n_pos": 5752,
            "source": "results/packetloss_dev/packetloss_results.csv (p=0.0, published)"})
    for _, r in full_pub.iterrows():
        comp_rows.append({
            "comparison": "iii_fullN_19feat", "model": r["model"].replace("__full", ""),
            "n_train": 15048, "n_features": 19,
            "input_repr": "19 raw window features (StandardScaler inside pipeline "
                          "where the model uses one)",
            "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"], "f1": r["f1"],
            "mcc": r["mcc"], "pr_auc_lo": np.nan, "pr_auc_hi": np.nan,
            "roc_auc_lo": np.nan, "roc_auc_hi": np.nan,
            "n_test": 26392, "n_pos": 5752,
            "source": "results/packetloss_dev/packetloss_results.csv (p=0.0, "
                      "__full rows, published; CSV carries no CIs for these rows)"})

    # ------------------------------------------------------------------ #
    # 3. Reconstruction check: matched rbf_svm on the reconstructed 8-dim input
    # ------------------------------------------------------------------ #
    print("[baseline_fairness] reconstruction check: rbf_svm on reconstructed Xr[tr_match] ...")
    npos = int(np.bincount(data.y[tr_match].astype(int), minlength=2).min())
    folds = max(2, min(int(bcfg.get("cv_folds", 5)), npos))
    est = build_baseline("rbf_svm", bcfg, seed=BASELINE_SEED, folds=folds,
                         tune=(npos >= 2))
    est.fit(Xr[tr_match], data.y[tr_match])
    sc = baseline_scores(est, Xr[te])
    rep = detection_report(data.y[te], sc, est.predict(Xr[te]))
    pub_rbf = matched_pub[matched_pub["model"] == "rbf_svm"].iloc[0]
    recon_check = {
        "model": "rbf_svm", "reconstructed_pr_auc": rep["pr_auc"],
        "published_pr_auc": float(pub_rbf["pr_auc"]),
        "delta_pr_auc": rep["pr_auc"] - float(pub_rbf["pr_auc"]),
        "reconstructed_roc_auc": rep["roc_auc"],
        "published_roc_auc": float(pub_rbf["roc_auc"]),
        "delta_roc_auc": rep["roc_auc"] - float(pub_rbf["roc_auc"]),
    }
    print(f"[baseline_fairness]  reconstructed PR {rep['pr_auc']:.4f} vs published "
          f"{pub_rbf['pr_auc']:.4f} (delta {recon_check['delta_pr_auc']:+.4f}); "
          f"ROC {rep['roc_auc']:.4f} vs {pub_rbf['roc_auc']:.4f} "
          f"(delta {recon_check['delta_roc_auc']:+.4f})")

    # ------------------------------------------------------------------ #
    # 4. Comparison (ii): same 3,023 windows, full 19 raw features (RUN NOW)
    # ------------------------------------------------------------------ #
    # standardise on train only (rbf_svm's pipeline re-standardises internally per
    # CV fold, which is idempotent on already-standardised data; trees are unaffected)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(data.X[tr_match])
    Xtr19 = scaler.transform(data.X[tr_match])
    Xte19 = scaler.transform(data.X[te])
    for model in NEW_MODELS:
        print(f"[baseline_fairness] comparison (ii): {model} on 3,023 x 19 raw std features ...")
        t1 = time.time()
        est = build_baseline(model, bcfg, seed=BASELINE_SEED, folds=folds,
                             tune=(npos >= 2))
        est.fit(Xtr19, data.y[tr_match])
        sc = baseline_scores(est, Xte19)
        row = _report_row(model, "ii_sameN_19feat", data.y[te], sc,
                          est.predict(Xte19), data.groups[te],
                          n_train=len(tr_match), n_features=19,
                          input_repr="19 raw window features, StandardScaler fit on "
                                     "the 3,023 train windows only",
                          source="computed by baseline_fairness (this script)")
        best = getattr(est, "best_params_", None)
        row["best_params"] = json.dumps(best, default=str) if best else ""
        comp_rows.append(row)
        print(f"[baseline_fairness]   {model}: PR {row['pr_auc']:.4f} "
              f"[{row['pr_auc_lo']:.4f},{row['pr_auc_hi']:.4f}] "
              f"ROC {row['roc_auc']:.4f} best={best} ({time.time()-t1:.0f}s)")

    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(OUT / "baseline_comparisons.csv", index=False)

    # multi-seed context for comparison (i)
    ms = pd.read_csv(MULTISEED) if MULTISEED.exists() else pd.DataFrame()
    multiseed_ctx = (ms[["model", "n_seeds", "pr_auc_mean", "pr_auc_std",
                         "roc_auc_mean", "roc_auc_std"]].to_dict("records")
                     if len(ms) else "IRRECOVERABLE")

    # ------------------------------------------------------------------ #
    # 5. Fairness facts (code-verified, file:line cited)
    # ------------------------------------------------------------------ #
    facts = [
        {"fact": "same_train_val_test_groups",
         "answer": "YES",
         "evidence": "One group-aware split (grouped by claimed sender, test_frac 0.30,"
                     " val_frac 0.15, seed 7) is computed once per packet-loss level and"
                     " the SAME idx dict {tr_match, va, te} is passed to run_model for"
                     " every model, matched and __full alike. Verified empirically:"
                     " test=26,392 windows / 5,752 positives for every published row,"
                     " and the reconstructed split reproduces those counts exactly;"
                     " train/val/test groups are pairwise disjoint.",
         "file_line": "src/qkv2x/experiments/packetloss.py:71,81,102; "
                      "src/qkv2x/experiments/common.py:159-177; configs/features.yaml"
                      "::splits"},
        {"fact": "same_3023_train_windows_classical",
         "answer": "YES",
         "evidence": "Matched-N classical models are fitted on exactly Xr[tr_match]"
                     " where tr_match = matched_train_idx(...) = group-wise cap of the"
                     " train split at quantum max_train=3000 with seed 7, giving 3,023"
                     " windows (whole sender groups are kept, so the cap overshoots"
                     " 3000 by part of one group). Reconstructed: 3,023 windows, "
                     f"{len(set(data.groups[tr_match]))} sender groups.",
         "file_line": "src/qkv2x/experiments/packetloss.py:75,81; "
                      "src/qkv2x/experiments/common.py:266-274,346-354"},
        {"fact": "same_3023_train_windows_quantum",
         "answer": "YES (verified: the internal re-cap is a no-op)" if identical_train
                   else f"ALMOST -- run_model re-caps internally to {len(tr_q)} windows",
         "evidence": ("run_model's quantum branch re-applies _cap_train_groupwise to"
                      " the already-capped tr_match (3,023 > max_train=3,000, so the"
                      " cap re-runs with the same seed over the SUBSET's groups)."
                      f" Empirically the re-cap returns {len(tr_q)} of 3,023 windows"
                      f" ({'identical set' if identical_train else str(len(dropped)) + ' windows from dropped groups differ'})."),
         "file_line": "src/qkv2x/experiments/common.py:286-295 (re-cap), 243-263 "
                      "(_cap_train_groupwise)"},
        {"fact": "same_eight_pca_coordinates",
         "answer": "YES",
         "evidence": "A single BlockReducer (per-block PCA: 4 mobility + 4 network"
                     " components, clip [0.5%,99.5%] -> StandardScaler -> PCA ->"
                     " [0,pi] scaling) is fitted once on the UNCAPPED train split and"
                     " the same transformed matrix Xr serves the quantum kernels AND"
                     " the matched classical baselines (quantum: Xr[tr]/Xr[te] at"
                     " common.py:296-343; classical: Xr[tr]/Xr[te] at 346-354)."
                     " Verified: Xr has exactly 8 columns.",
         "file_line": "src/qkv2x/experiments/packetloss.py:73-74,81; "
                      "src/qkv2x/features/reduce.py; src/qkv2x/experiments/"
                      "common.py:296-354"},
        {"fact": "who_sees_the_19_raw_features",
         "answer": "ONLY the '(full N)' classical rows",
         "evidence": "The __full variants fit on data.X (the 19 raw window features,"
                     " scaled inside each model's own pipeline) with the train split"
                     " capped at full_cap=15,000 -> 15,048 windows. The published"
                     " Table-I '(full N)' rows therefore change BOTH the sample count"
                     " (3,023 -> 15,048) AND the input representation (8-dim PCA ->"
                     " 19 raw features) at once; baseline_fairness comparison (ii) disentangles"
                     " this. The quantum models never see the 19 raw features (the"
                     " 8 qubits encode the 8 reduced angles).",
         "file_line": "src/qkv2x/experiments/common.py:356-365; "
                      "src/qkv2x/experiments/packetloss.py:97-109; "
                      "configs/experiments/packetloss.yaml::full_cap"},
        {"fact": "same_test_set_everywhere",
         "answer": "YES",
         "evidence": "Matched rows and __full rows are scored on the identical te"
                     " indices (the same idx['te'] object); every published p=0 row"
                     " has n_test=26,392, n_pos=5,752.",
         "file_line": "src/qkv2x/experiments/packetloss.py:81,102"},
        {"fact": "validation_split_usage",
         "answer": "Shared but unused by Table-I models",
         "evidence": "va (15% of groups) is carved out identically for all models;"
                     " the matched classical and quantum SVMs tune by internal CV on"
                     " the train subset instead (GridSearchCV / group-aware C-grid"
                     " CV). va is consumed only by the one-class detectors"
                     " (nu selection) and the LSTM (early model selection), which are"
                     " not Table-I rows.",
         "file_line": "src/qkv2x/experiments/common.py:303-313,367-389; "
                      "src/qkv2x/models/qsvm.py:36-60"},
        {"fact": "comparable_tuning_budgets",
         "answer": "Classical tuned at least as hard as quantum on the SVM axis; "
                   "quantum has an extra feature-map-training axis, reported openly",
         "evidence": "Classical: GridSearchCV, 5-fold stratified (seed 17), scoring"
                     " average_precision -- rbf_svm 16 configs/80 fits, random_forest"
                     " 4/20, mlp 4/20, xgboost 36/180. Quantum: C-grid {0.1,1,10,100}"
                     " over 5 group-aware folds (20 fits, same AP scoring) PLUS"
                     " KTA feature-map training (Adam lr 0.05, 200 epochs, batch 512,"
                     " 1 restart, best-alignment snapshot). qsvm_zz deliberately gets"
                     " NO map training (fixed-parameter ablation).",
         "file_line": "src/qkv2x/models/baselines.py:36-116; src/qkv2x/models/"
                      "qsvm.py:36-60; src/qkv2x/quantum/train.py:72-121; "
                      "src/qkv2x/experiments/common.py:224-228"},
    ]
    pd.DataFrame(facts).to_csv(OUT / "baseline_fairness.csv", index=False)

    # ------------------------------------------------------------------ #
    # 6. Hyperparameter inventory
    # ------------------------------------------------------------------ #
    hp = []

    def add(model, name, value, sel, src):
        hp.append({"model": model, "hyperparameter": name,
                   "value_or_grid": value if isinstance(value, str) else json.dumps(value),
                   "selection_method": sel, "source_file": src})

    BY, QY = "configs/baselines.yaml", "configs/quantum.yaml"
    BP, QP, TP = ("src/qkv2x/models/baselines.py", "src/qkv2x/models/qsvm.py",
                  "src/qkv2x/quantum/train.py")

    # shared preprocessing (identical for quantum + matched classical)
    red = feats["reduction"]
    add("_shared_input", "reduction.method", red["method"], "fixed", "configs/features.yaml")
    add("_shared_input", "reduction.n_mobility", red["n_mobility"], "fixed", "configs/features.yaml")
    add("_shared_input", "reduction.n_network", red["n_network"], "fixed", "configs/features.yaml")
    add("_shared_input", "reduction.clip_quantiles", red["clip_quantiles"], "fixed", "configs/features.yaml")
    add("_shared_input", "reduction.scale_range", "[0, pi]", "fixed", "configs/features.yaml")
    add("_shared_protocol", "quantum.max_train (matched-N cap)", qcfg["max_train"], "fixed", QY)
    add("_shared_protocol", "packetloss.full_cap (classical @full rows)", 15000, "fixed",
        "configs/experiments/packetloss.yaml")
    add("_shared_protocol", "cv_folds (classical grid-CV)", bcfg["cv_folds"], "fixed", BY)
    add("_shared_protocol", "grid-CV scoring", "average_precision", "fixed", f"{BP}:61-62")
    add("_shared_protocol", "baseline seed (CV shuffle + model random_state)",
        bcfg["seed"], "fixed", BY)
    add("_shared_protocol", "split seed / test_frac / val_frac / group_by",
        [splits["seed"], splits["test_frac"], splits["val_frac"], splits["group_by"]],
        "fixed", "configs/features.yaml::splits")

    rc = bcfg["rbf_svm"]
    add("rbf_svm", "C", rc["C_grid"], "grid-CV", BY)
    add("rbf_svm", "gamma", rc["gamma_grid"], "grid-CV", BY)
    add("rbf_svm", "class_weight", rc["class_weight"], "fixed", BY)
    add("rbf_svm", "scaler", "StandardScaler (in pipeline)", "fixed", f"{BP}:52")
    rc = bcfg["random_forest"]
    add("random_forest", "n_estimators", rc["n_estimators"], "grid-CV", BY)
    add("random_forest", "max_depth", rc["max_depth"], "grid-CV", BY)
    add("random_forest", "class_weight", rc["class_weight"], "fixed", BY)
    rc = bcfg["knn_bagging"]
    add("knn_bagging", "k (n_neighbors)", rc["k"], "grid-CV", BY)
    add("knn_bagging", "n_estimators (bagging)", rc["n_estimators"], "fixed", BY)
    add("knn_bagging", "max_samples", rc["max_samples"], "fixed", BY)
    add("knn_bagging", "scaler", "StandardScaler (in pipeline)", "fixed", f"{BP}:82")
    rc = bcfg["mlp"]
    add("mlp", "hidden_layer_sizes", rc["hidden"], "grid-CV", BY)
    add("mlp", "alpha", rc["alpha"], "grid-CV", BY)
    add("mlp", "max_iter", rc["max_iter"], "fixed", BY)
    add("mlp", "early_stopping", "False in the binary run_model path (build_baseline "
        "hard-codes it; the yaml 'true' feeds only build_multiclass)", "fixed", f"{BP}:92")
    add("mlp", "solver/activation", "adam / relu (sklearn defaults)", "fixed", f"{BP}:91-93")
    add("mlp", "scaler", "StandardScaler (in pipeline)", "fixed", f"{BP}:93")
    add("xgboost", "n_estimators", [200, 400], "grid-CV", f"{BP}:110 (code default; no "
        "xgboost block in baselines.yaml)")
    add("xgboost", "max_depth", [4, 6, 8], "grid-CV", f"{BP}:111")
    add("xgboost", "learning_rate", [0.05, 0.1], "grid-CV", f"{BP}:112")
    add("xgboost", "scale_pos_weight", [1.0, 3.0, 5.0], "grid-CV", f"{BP}:114")
    add("xgboost", "tree_method / eval_metric", "hist / logloss", "fixed", f"{BP}:28-29")
    rc = bcfg["ocsvm"]
    add("ocsvm_classical", "nu", rc["nu_grid"], "validation (ROC-AUC on va)",
        f"{BY}; src/qkv2x/experiments/common.py:374-386")
    add("ocsvm_classical", "gamma", rc["gamma"], "fixed", BY)
    add("ocsvm_classical", "kernel", "rbf", "fixed", "src/qkv2x/experiments/common.py:382")
    rc = bcfg["rules"]
    for k, v in rc.items():
        add("rules", k, v, "fixed", BY)
    add("lstm_seq", "hidden / layers", [64, 1], "fixed (code defaults; yaml sets only "
        "enabled:true)", "src/qkv2x/models/seq_baselines.py:17-18; "
        "src/qkv2x/experiments/veremi_nextgen_exp.py:596-600")
    add("lstm_seq", "epochs / batch / lr", [40, 256, 0.001], "fixed",
        "src/qkv2x/models/seq_baselines.py:17-18")
    add("lstm_seq", "optimiser / loss", "Adam / BCEWithLogits(pos_weight=neg:pos)",
        "fixed", "src/qkv2x/models/seq_baselines.py:76-80")
    add("lstm_seq", "model selection", "best epoch by validation average precision",
        "validation", "src/qkv2x/models/seq_baselines.py (fit loop)")

    fm = qcfg["feature_map"]
    for qm in ("qsvm_crosslayer", "qsvm_zz"):
        ent = "zz_full (fixed ablation)" if qm == "qsvm_zz" else fm["entangler"]
        add(qm, "n_qubits", "8 (= 4 mobility + 4 network reduced coords)", "fixed",
            "configs/features.yaml::reduction")
        add(qm, "feature_map.n_layers", fm["n_layers"], "fixed", QY)
        add(qm, "feature_map.entangler", ent, "fixed",
            f"{QY}; src/qkv2x/experiments/common.py:206-208")
        add(qm, "feature_map.coupling", fm["coupling"], "fixed", QY)
        add(qm, "feature_map.aoi_modulation (kappa)",
            [fm["aoi_modulation"], feats["aoi"]["kappa"]], "fixed",
            f"{QY}; configs/features.yaml::aoi")
        add(qm, "kernel / shots / psd_repair",
            [qcfg["kernel"], "exact statevector (shots=null)", qcfg["psd_repair"]],
            "fixed", QY)
        add(qm, "svm.C", qcfg["svm"]["C_grid"],
            "grid-CV (5-fold group-aware, average_precision)", f"{QY}; {QP}:36-60")
        add(qm, "svm.class_weight", qcfg["svm"].get("class_weight", "balanced"),
            "fixed", QY)
    tc = qcfg["train"]
    add("qsvm_crosslayer", "KTA optimiser", "Adam", "fixed", f"{TP}:87-88")
    add("qsvm_crosslayer", "KTA lr", tc["lr"], "fixed", QY)
    add("qsvm_crosslayer", "KTA epochs", tc["epochs"], "fixed", QY)
    add("qsvm_crosslayer", "KTA batch (class-balanced)", tc["batch"], "fixed", QY)
    add("qsvm_crosslayer", "KTA objective", tc["objective"] +
        " (centered KTA vs yy^T target)", "fixed", f"{QY}; {TP}:22-34")
    add("qsvm_crosslayer", "KTA init (w,u,gamma)",
        [tc["init_w"], tc["init_u"], tc["init_gamma"]], "fixed", QY)
    add("qsvm_crosslayer", "KTA restarts", "1 (single init; best-alignment snapshot "
        "across epochs)", "fixed", f"{TP}:94-121")
    add("qsvm_crosslayer", "KTA seed", tc["seed"], "fixed", QY)
    add("qsvm_zz", "KTA training", "NONE -- fixed_params() ablation", "fixed",
        "src/qkv2x/experiments/common.py:224-228")
    add("qsvm_projected", "projected_gamma", qcfg["projected_gamma"], "fixed", QY)
    add("ocsvm_quantum", "nu", qcfg["ocsvm"]["nu_grid"],
        "validation (ROC-AUC on va)", f"{QY}; src/qkv2x/experiments/common.py:303-313")
    ny = qcfg["nystrom"]
    add("_nystrom_opt_in", "enabled / landmarks / max_train / seed",
        [ny["enabled"], ny["landmarks"], ny["max_train"], ny["seed"]],
        "fixed (OFF in all headline runs)", QY)

    hp_df = pd.DataFrame(hp)
    hp_df.to_csv(OUT / "hyperparams.csv", index=False)
    print(f"[baseline_fairness] hyperparameter inventory: {len(hp_df)} rows, "
          f"{hp_df['model'].nunique()} models/sections")

    # ------------------------------------------------------------------ #
    # 7. Summary JSON
    # ------------------------------------------------------------------ #
    def _rows(tag):
        cols = ["model", "n_train", "n_features", "pr_auc", "roc_auc", "f1", "mcc",
                "pr_auc_lo", "pr_auc_hi", "roc_auc_lo", "roc_auc_hi", "source"]
        d = comp_df[comp_df["comparison"] == tag][cols]
        return json.loads(d.to_json(orient="records"))

    summary = {
        "item": "baseline_fairness",
        "protocol": {
            "dataset": "runs/dev_grid/features.parquet (90,553 windows, 19 raw "
                       "features; identical loader to scripts/dev_multiseed.py)",
            "table_I_source": "results/packetloss_dev/packetloss_results.csv, "
                              "packet_loss==0.0 slice (matched rows n_train=3023; "
                              "__full rows n_train=15048)",
            "split": {"test_frac": splits["test_frac"], "val_frac": splits["val_frac"],
                      "group_by": splits["group_by"], "seed": split_seed},
            "matched_cap": {"max_train": max_train, "cap_seed": split_seed,
                            "group_wise": True},
            "full_cap": full_cap,
            "comparison_ii": {
                "models": NEW_MODELS,
                "train_windows": "the identical 3,023-window tr_match subset",
                "features": "19 raw window features, StandardScaler fit on train only",
                "tuning": "build_baseline grids, GridSearchCV 5-fold stratified "
                          "(seed 17), scoring average_precision -- identical to the "
                          "published run_model classical path",
                "test": "identical 26,392-window test split",
                "bootstrap": {"n_boot": N_BOOT, "seed": BOOT_SEED,
                              "grouped_by": "sender group"},
                "preexisting": "none found anywhere in results/ (searched all CSVs "
                               "for __full/matched rows with n_train<10k)",
            },
            "seeds": {"split_and_cap": split_seed, "baselines": BASELINE_SEED,
                      "bootstrap": BOOT_SEED},
        },
        "invariants_check": {**invariants, "reconstruction_matches_published": recon_ok},
        "reconstruction_check_rbf_matched_8dim": recon_check,
        "fairness_facts": facts,
        "quantum_inner_recap": {
            "identical_to_classical_matched_set": identical_train,
            "n_matched": int(len(tr_match)), "n_after_recap": int(len(tr_q)),
            "n_dropped": len(dropped)},
        "comparisons": {
            "i_sameN_8dim_published": _rows("i_sameN_8dim"),
            "i_multiseed_context_dev_grid_multiseed": multiseed_ctx,
            "ii_sameN_19feat_computed_here": _rows("ii_sameN_19feat"),
            "iii_fullN_19feat_published": _rows("iii_fullN_19feat"),
        },
        "hyperparam_inventory_rows": int(len(hp_df)),
        "wall_time_s": round(time.time() - t0, 1),
        "outputs": ["baseline_fairness.csv", "baseline_comparisons.csv",
                    "hyperparams.csv", "baseline_fairness_summary.json", "baseline_fairness_run.log"],
    }
    (OUT / "baseline_fairness_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"[baseline_fairness] wrote {OUT}/baseline_fairness_summary.json ({summary['wall_time_s']}s)")


if __name__ == "__main__":
    main()
