"""label_rule_sensitivity.py -- LOCAL postprocess of the label_rule label-rule Grams.

The default window labelling rule marks a window malicious when any message is attack
active, which may add label noise. This script runs sensitivity checks using fully
active windows, majority-malicious windows, the attack fraction per window, and removal
of windows crossing an attack onset.

Design (see gram_inputs_prep.py): 'majority' and 'fully_active' RELABEL the identical
window set under the identical published seed-7 split (isolating the labelling
effect); 'drop_onset' removes the 486 onset-straddling windows (0.54% of the
dataset) and refits the reducer. Quantum Grams were KTA-retrained per rule on
a GPU host with the equivalence-proven GPU engine (labels enter KTA training);
classical models are refit here. A score-fixed attack-fraction sweep (published
'any' scores, relabelled evaluation) covers the continuous-threshold axis without
retraining.

Run: python scripts/studies/label_rule_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qkv2x.config import load_yaml                                   # noqa: E402
from qkv2x.models.baselines import baseline_scores, build_baseline   # noqa: E402
from qkv2x.models.metrics import (detection_report,                  # noqa: E402
                                  grouped_bootstrap_ci)
from qkv2x.models.qsvm import QuantumKernelSVM                       # noqa: E402

OUT = REPO / "results" / "studies"
GRAMS = OUT / "grams"
N_BOOT = 200


def _quantum_row(rule, K_tr, K_te, y_tr, y_te, groups_tr, groups_te, qc, extra):
    svm = QuantumKernelSVM(C_grid=qc["svm"]["C_grid"], cv_folds=qc["svm"]["cv_folds"])
    svm.fit(K_tr, y_tr, groups=groups_tr)
    scores, preds = svm.decision_function(K_te), svm.predict(K_te)
    rep = detection_report(y_te, scores, preds)
    ci = grouped_bootstrap_ci(y_te, scores, groups_te, n_boot=N_BOOT, seed=7)
    return {"rule": rule, "model": "qsvm_crosslayer", **extra,
            "n_te": int(len(y_te)), "n_pos_te": int(y_te.sum()),
            "roc_auc": rep["roc_auc"], "pr_auc": rep["pr_auc"],
            "f1": rep["f1"], "mcc": rep["mcc"],
            "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"]}


def _classical_rows(rule, Xr, itr, ite, y_rule, groups, bcfg, extra):
    rows = []
    for m in ("rbf_svm", "random_forest"):
        est = build_baseline(m, bcfg, seed=7)
        est.fit(Xr[itr], y_rule[itr])
        scores = baseline_scores(est, Xr[ite])
        preds = est.predict(Xr[ite])
        rep = detection_report(y_rule[ite], scores, preds)
        ci = grouped_bootstrap_ci(y_rule[ite], scores, groups[ite],
                                  n_boot=N_BOOT, seed=7)
        rows.append({"rule": rule, "model": m, **extra,
                     "n_te": int(len(ite)), "n_pos_te": int(y_rule[ite].sum()),
                     "roc_auc": rep["roc_auc"], "pr_auc": rep["pr_auc"],
                     "f1": rep["f1"], "mcc": rep["mcc"],
                     "pr_auc_lo": ci["pr_auc"]["lo"],
                     "pr_auc_hi": ci["pr_auc"]["hi"]})
    return rows


def main() -> None:
    z = np.load(OUT / "gram_inputs" / "label_rule_inputs.npz")
    qc = load_yaml(str(REPO / "configs" / "quantum.yaml"))
    bcfg = load_yaml(str(REPO / "configs" / "baselines.yaml"))
    groups = z["groups"]
    itr, ite = z["idx_tr"], z["idx_te"]
    frac = z["attack_frac"]

    rows = []
    # published 'any' anchor rows (same split/protocol, from the released run)
    pub = pd.read_csv(REPO / "results" / "dev_grid_multiseed" / "binary_results.csv")
    for m in ("qsvm_crosslayer", "rbf_svm", "random_forest"):
        p = pub[(pub.seed == 7) & (pub.model == m)]
        if len(p):
            r = p.iloc[0]
            rows.append({"rule": "any (published)", "model": m,
                         "window_set": "full", "n_te": int(r["n_test"]),
                         "n_pos_te": int(r["n_pos"]),
                         "roc_auc": float(r["roc_auc"]),
                         "pr_auc": float(r["pr_auc"]),
                         "f1": float(r["f1"]), "mcc": float(r["mcc"]),
                         "pr_auc_lo": float(r["pr_auc_lo"]),
                         "pr_auc_hi": float(r["pr_auc_hi"])})

    # relabel-only rules (identical windows, identical split)
    for rule in ("majority", "fully_active"):
        y_rule = z[f"y_{rule}"].astype(int)
        g = np.load(GRAMS / f"gram_{rule}.npz")
        rows.append(_quantum_row(rule, g["K_tr"].astype(np.float64),
                                 g["K_te"].astype(np.float64),
                                 y_rule[itr], y_rule[ite],
                                 groups[itr], groups[ite], qc,
                                 {"window_set": "full"}))
        rows.extend(_classical_rows(rule, z["Xr"], itr, ite, y_rule, groups,
                                    bcfg, {"window_set": "full"}))

    # drop-onset rule (window subset, refit reducer, labels coincide across rules)
    y_o = z["y_fully_active"].astype(int)
    itr_o, ite_o = z["idx_tr_onset"], z["idx_te_onset"]
    g = np.load(GRAMS / "gram_onset.npz")
    rows.append(_quantum_row("drop_onset", g["K_tr"].astype(np.float64),
                             g["K_te"].astype(np.float64),
                             y_o[itr_o], y_o[ite_o],
                             groups[itr_o], groups[ite_o], qc,
                             {"window_set": "onset-dropped"}))
    rows.extend(_classical_rows("drop_onset", z["Xr_onset"], itr_o, ite_o, y_o,
                                groups, bcfg, {"window_set": "onset-dropped"}))

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "label_rule.csv", index=False)
    print(df[["rule", "model", "roc_auc", "pr_auc", "n_pos_te"]]
          .to_string(index=False))

    # score-fixed attack-fraction sweep: published seed-7 qsvm scores, relabelled te
    res7 = json.loads((REPO / "results" / "dev_grid_multiseed" / "checkpoints" /
                       "res_qsvm_crosslayer_s7.json").read_text())
    s_te = np.asarray(res7["scores"], float)
    y_cached = np.asarray(res7["y"], int)
    assert (y_cached == z["y_any"][ite].astype(int)).all(), \
        "cached te ordering != label_rule idx_te ordering"
    sweep = []
    for tau, lab in [(0.0, "any (frac>0)"), (0.25, "frac>=0.25"),
                     (0.5, "frac>=0.5 (majority)"), (0.75, "frac>=0.75"),
                     (1.0, "frac==1 (fully_active)")]:
        y_tau = ((frac[ite] >= tau) & (frac[ite] > 0)).astype(int) if tau > 0 \
            else (frac[ite] > 0).astype(int)
        if tau == 1.0:
            y_tau = (frac[ite] == 1.0).astype(int)
        sweep.append({"tau": tau, "label": lab, "n_pos": int(y_tau.sum()),
                      "roc_auc": float(roc_auc_score(y_tau, s_te)),
                      "pr_auc": float(average_precision_score(y_tau, s_te))})
    pd.DataFrame(sweep).to_csv(OUT / "label_rule_fraction_sweep.csv", index=False)

    summary = {
        "protocol": json.loads((OUT / "gram_inputs" / "gram_inputs_summary.json")
                               .read_text()),
        "rows": rows, "fraction_sweep": sweep,
        "note": "mixed windows are only 0.54% of the dataset, so label-rule "
                "sensitivity is bounded a priori; the rows quantify the realised "
                "effect. drop_onset test set is 27 windows smaller (5,973).",
    }
    (OUT / "label_rule_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("fraction sweep:", json.dumps(sweep))


if __name__ == "__main__":
    main()
