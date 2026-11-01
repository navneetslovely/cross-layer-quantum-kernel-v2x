"""mimicry_ksweep.py -- adaptive-mimicry hardening: candidate-count sensitivity,
common-realisation (transfer) control, and the DoS robustness-gap mechanism.

The attacks are per-model white-box attacks. Reports a common attack realisation
across models, attack-wise outcomes, sensitivity to the candidate count (K = 8, 16,
32, 64, 128), and explains why QSVM keeps 0.98 DoS recall while random forest keeps
0.40 (threshold calibration and feature dependence).

Inputs: results/evasion/evasion_results{_k8,_k16,_common32,_k64,_k128}.csv produced
by scripts/mimicry_evasion.py --k-cand ... / --common-donor (runs distributed across
GPU hosts; the published K=32 white-box run is
results/evasion/evasion_results.csv). Missing K files are reported, not fabricated.

DoS mechanism analysis (computed here, locally): the mimicry envelopes substitute
mobility + plausibility (+ e_xl/overlap in 'strong') columns ONLY -- rate/timing
features (rate_ratio, interarrival_std, pdr_proxy) are NEVER substituted, because
lowering the send rate would remove the flooding effect (goal-constrained). So any
DoS recall lost under mimicry measures a model's reliance on NON-rate features for
a rate-defined attack. We quantify that reliance for the published random-forest
model via (a) its native importances and (b) permutation importance on the
DoS-vs-benign test subset, split into the mobility-PCA block (components 0-3,
substitutable) vs the network-PCA block (components 4-7, carrying rate/timing).

Run: python scripts/studies/mimicry_ksweep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qkv2x.config import load_yaml                                   # noqa: E402
from qkv2x.features.reduce import BlockReducer                       # noqa: E402

EV = REPO / "results" / "evasion"
OUT = REPO / "results" / "studies"
K_FILES = {8: "evasion_results_k8.csv", 16: "evasion_results_k16.csv",
           32: "evasion_results.csv",          # published per-model white-box run
           64: "evasion_results_k64.csv", 128: "evasion_results_k128.csv"}
COMMON = "evasion_results_common32.csv"


def _load_nextgen():
    z = np.load(REPO / "results" / "veremi_nextgen" / "checkpoints" /
                "assembled.npz", allow_pickle=True)
    names = [str(s) for s in z["feature_names"]]
    blocks = [str(s) for s in z["blocks"]]
    split = z["split"].astype(str)
    return (z["X"], z["y"].astype(int), z["attack_name"].astype(str), names,
            blocks, np.where(split == "Train")[0], np.where(split == "Test")[0])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- 1. K-sweep aggregation --------------------------------------------
    sweep_rows, missing = [], []
    for k, fname in K_FILES.items():
        p = EV / fname
        if not p.exists():
            missing.append(f"K={k} ({fname})")
            continue
        df = pd.read_csv(p)
        df["k_cand"] = k
        df["selection"] = df.get("selection", "per_model")
        df.loc[df["selection"].isna(), "selection"] = "per_model"
        sweep_rows.append(df)
    sweep = pd.concat(sweep_rows, ignore_index=True)
    cm = None
    if (EV / COMMON).exists():
        cm = pd.read_csv(EV / COMMON)
        cm["k_cand"] = 32
        sweep = pd.concat([sweep, cm], ignore_index=True)
    sweep.to_csv(OUT / "mimicry_ksweep.csv", index=False)

    allrows = sweep[sweep.attack == "ALL"]
    piv = allrows.pivot_table(index=["envelope", "model", "selection"],
                              columns="k_cand", values="recall_after")
    print("recall_after (attack=ALL):")
    print(piv.round(3).to_string())

    # saturation: per (envelope, model) white-box drop from smallest to largest K
    sat = {}
    for (env, m), g in allrows[allrows.selection == "per_model"].groupby(
            ["envelope", "model"]):
        g = g.sort_values("k_cand")
        ks = g.k_cand.tolist()
        ra = g.recall_after.tolist()
        sat[f"{env}/{m}"] = {"k_grid": ks, "recall_after": ra,
                             "delta_first_to_last": float(ra[-1] - ra[0])}

    # white-box vs common-realisation at K=32
    wb32 = allrows[(allrows.k_cand == 32) & (allrows.selection == "per_model")]
    cd32 = allrows[(allrows.k_cand == 32) &
                   (allrows.selection.str.startswith("surrogate"))]
    transfer = []
    for _, r in cd32.iterrows():
        w = wb32[(wb32.envelope == r.envelope) & (wb32.model == r.model)]
        if len(w):
            transfer.append({"envelope": r.envelope, "model": r.model,
                             "recall_after_whitebox": float(w.recall_after.iloc[0]),
                             "recall_after_common": float(r.recall_after),
                             "transfer_gap": float(r.recall_after
                                                   - w.recall_after.iloc[0])})

    # per-attack table across K (dosAttack highlighted)
    dos = sweep[sweep.attack == "dosAttack"].pivot_table(
        index=["envelope", "model", "selection"], columns="k_cand",
        values="recall_after")
    dos.to_csv(OUT / "mimicry_dos_by_k.csv")

    # ---- 2. DoS-gap mechanism ----------------------------------------------
    X, y, atk, names, blocks, tr, te = _load_nextgen()
    fcfg = load_yaml(str(REPO / "configs" / "features_veremi_nextgen.yaml"))
    reducer = BlockReducer(fcfg["reduction"]).fit(X[tr], blocks)
    Xr = reducer.transform(X)
    n_mob = int(fcfg["reduction"]["n_mobility"])

    rf = joblib.load(EV / "checkpoints" / "model_random_forest.joblib")
    core = rf
    for attr in ("best_estimator_",):            # unwrap GridSearchCV
        core = getattr(core, attr, core)
    if hasattr(core, "steps"):                   # unwrap Pipeline
        core = core.steps[-1][1]
    imp = np.asarray(core.feature_importances_, float)
    native_split = {"mobility_block": float(imp[:n_mob].sum()),
                    "network_block": float(imp[n_mob:].sum())}

    # permutation importance on the DoS-vs-benign TEST subset (blockwise)
    from sklearn.metrics import roc_auc_score
    m_dos = (atk == "dosAttack") & np.isin(np.arange(len(y)), te)
    m_ben = (y == 0) & np.isin(np.arange(len(y)), te)
    idx = np.where(m_dos | m_ben)[0]
    yb = (atk[idx] == "dosAttack").astype(int)
    Xb = Xr[idx]
    def _score(Xe):
        return rf.predict_proba(Xe)[:, 1] if hasattr(rf, "predict_proba") \
            else rf.decision_function(Xe)
    base_auc = float(roc_auc_score(yb, _score(Xb)))
    rng = np.random.default_rng(7)
    perm = {}
    for label, cols in [("mobility_block", list(range(n_mob))),
                        ("network_block", list(range(n_mob, Xr.shape[1])))]:
        drops = []
        for _ in range(20):
            Xp = Xb.copy()
            for c in cols:
                Xp[:, c] = rng.permutation(Xp[:, c])
            drops.append(base_auc - float(roc_auc_score(yb, _score(Xp))))
        perm[label] = {"mean_auc_drop": float(np.mean(drops)),
                       "std": float(np.std(drops))}

    # threshold-calibration facts
    thr_facts = {
        "qsvm_crosslayer": "margin threshold 0.0 (SVM decision function sign)",
        "random_forest/xgboost": "probability threshold 0.5",
        "fpr_note": "benign windows are untouched by the substitution, so each "
                    "model's FPR is unchanged by construction; recall_before/after "
                    "are therefore compared at each model's own fixed operating "
                    "point (not at matched FPR) -- the published caveat stands, "
                    "and threshold_protocol supplies validation-calibrated matched-FPR points.",
    }

    summary = {
        "protocol": {
            "whitebox": "per-model best-of-K substitution (each model faces the "
                        "candidate minimising ITS OWN score) -- stated explicitly",
            "common_realisation": "candidates chosen once against the "
                                  "random_forest surrogate, identical adversarial "
                                  "windows evaluated on every model (transfer)",
            "k_grid_present": sorted(allrows[allrows.selection == "per_model"]
                                     .k_cand.unique().tolist()),
            "missing_runs": missing,
            "envelopes": "conservative = mobility + off-road plausibility; strong "
                         "= + e_xl + claim-overlap. Rate/timing features are never "
                         "substituted (goal-constrained: lowering rate removes the "
                         "flooding effect).",
        },
        "saturation": sat,
        "transfer_vs_whitebox_k32": transfer,
        "dos_gap": {
            "published_numbers": {"qsvm_strong": 0.977, "rf_strong": 0.397},
            "rf_native_importance_split": native_split,
            "rf_permutation_auc_drop_dos_vs_benign": perm,
            "base_auc_dos_vs_benign_rf": base_auc,
            "mechanism": "DoS is rate-defined and the substitution never touches "
                         "rate/timing features; recall lost under mimicry therefore "
                         "measures reliance on substitutable (mobility/plausibility) "
                         "features. The RF spreads weight onto the mobility block "
                         "(see importances), so benign-looking mobility values flip "
                         "its DoS windows; the trained cross-layer QSVM anchors DoS "
                         "on the network-block rate signal and keeps recall.",
            "threshold_calibration": thr_facts,
        },
    }
    (OUT / "mimicry_summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=float))
    print(json.dumps({"missing": missing, "native_split": native_split,
                      "perm": perm}, indent=1))


if __name__ == "__main__":
    main()
