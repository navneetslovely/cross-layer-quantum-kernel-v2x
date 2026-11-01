#!/usr/bin/env python3
"""Multi-seed replication + Holm-corrected significance for the MAIN-TASK
(dev_grid) binary detection at matched N.

The NextGen external evaluation already has 3-seed mean±std + corrected paired
tests; this brings the main co-simulation task to the same statistical standard:
seeds {7,17,27} vary the leakage-safe group split, the train cap, the KTA
training, and the model seeds. Models: the two quantum kernels + the strongest
classical baselines incl. XGBoost. Matched N: train capped group-wise to
quantum.yaml::max_train for ALL models (the established RQ1 protocol).

Checkpointed under results/dev_grid_multiseed/checkpoints (QKV2X_BUDGET_S aware;
KTA training resumes epoch-exact). Outputs: binary CSV (per seed), multiseed
mean±std CSV, Holm-corrected significance CSV (seed 7).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qkv2x.config import load_yaml                                  # noqa: E402
from qkv2x.experiments.common import (ExperimentData, _cap_train_groupwise,
                                      fit_reducer, group_split,
                                      make_feature_map)             # noqa: E402
from qkv2x.experiments.veremi_nextgen_exp import (BudgetExceeded,
                                                  _check_budget,
                                                  _qtrain_checkpointed,
                                                  _savez_atomic)    # noqa: E402
from qkv2x.features.crosslayer import FEATURE_SPEC                  # noqa: E402
from qkv2x.models.baselines import baseline_scores, build_baseline  # noqa: E402
from qkv2x.models.metrics import (detection_report, grouped_bootstrap_ci,
                                  holm_bonferroni,
                                  paired_bootstrap_auc_diff)        # noqa: E402
from qkv2x.models.qsvm import QuantumKernelSVM                      # noqa: E402
from qkv2x.quantum.kernels import fidelity_gram                     # noqa: E402

OUT = REPO / "results" / "dev_grid_multiseed"
CK = OUT / "checkpoints"
SEEDS = [7, 17, 27]
MODELS = ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest", "xgboost"]
N_BOOT = 200


def _load():
    df = pd.read_parquet(REPO / "runs" / "dev_grid" / "features.parquet")
    names = [s[0] for s in FEATURE_SPEC]
    blocks = [s[3] for s in FEATURE_SPEC]
    return ExperimentData(
        X=df[names].to_numpy(), y=df["y"].to_numpy(),
        attack_name=df["attack_name"].to_numpy(object),
        groups=df["group"].to_numpy(), aoi=df["aoi"].to_numpy(),
        t_end=df["t_end"].to_numpy(), feature_names=names, blocks=blocks,
        points=pd.DataFrame(index=range(len(df))))


def _eval(model, data, Xr, idx, cfgs, seed):
    rpath = CK / f"res_{model}_s{seed}.json"
    if rpath.exists():
        return json.loads(rpath.read_text())
    _check_budget()
    tr, va, te = idx["tr"], idx["va"], idx["te"]
    qc = cfgs["quantum"]
    if model in ("qsvm_crosslayer", "qsvm_zz"):
        fm = make_feature_map(model, qc, cfgs["features"]["reduction"],
                              cfgs["features"])
        if model == "qsvm_zz":
            params = fm.fixed_params()
        else:
            params, _ = _qtrain_checkpointed(
                fm, Xr[tr], data.y[tr], data.aoi[tr], qc["train"],
                CK / f"qtrain_{model}_s{seed}.pt", multiclass=False)
        gpath = CK / f"gram_{model}_s{seed}.npz"
        if gpath.exists():
            z = np.load(gpath)
            K_tr, K_te = z["K_tr"], z["K_te"]
        else:
            _check_budget()
            s_tr = fm.statevectors(Xr[tr], data.aoi[tr], params)
            K_tr = fidelity_gram(s_tr)
            K_te = fidelity_gram(fm.statevectors(Xr[te], data.aoi[te], params),
                                 s_tr)
            _savez_atomic(gpath, K_tr=K_tr, K_te=K_te)
        svm = QuantumKernelSVM(C_grid=qc["svm"]["C_grid"],
                               cv_folds=qc["svm"]["cv_folds"])
        svm.fit(K_tr, data.y[tr], groups=data.groups[tr])
        scores, preds = svm.decision_function(K_te), svm.predict(K_te)
    else:
        est = build_baseline(model, cfgs["baselines"], seed=seed)
        est.fit(Xr[tr], data.y[tr])
        scores, preds = baseline_scores(est, Xr[te]), est.predict(Xr[te])

    rep = detection_report(data.y[te], scores, preds)
    ci = grouped_bootstrap_ci(data.y[te], scores, data.groups[te],
                              n_boot=N_BOOT, seed=seed)
    out = {"row": {"model": model, "seed": seed,
                   **{k: rep.get(k) for k in
                      ["pr_auc", "roc_auc", "f1", "mcc"]},
                   "pr_auc_lo": ci["pr_auc"]["lo"],
                   "pr_auc_hi": ci["pr_auc"]["hi"],
                   "n_train": int(len(tr)), "n_test": rep["n"],
                   "n_pos": rep["n_pos"]},
           "scores": np.asarray(scores, float).tolist(),
           "y": data.y[idx["te"]].tolist(),
           "groups": data.groups[idx["te"]].tolist()}
    rpath.write_text(json.dumps(out, default=float))
    return out


def main():
    CK.mkdir(parents=True, exist_ok=True)
    data = _load()
    cfgs = {"features": load_yaml(str(REPO / "configs/features.yaml")),
            "quantum": load_yaml(str(REPO / "configs/quantum.yaml")),
            "baselines": load_yaml(str(REPO / "configs/baselines.yaml"))}
    rows = []
    try:
        for seed in SEEDS:
            s = cfgs["features"]["splits"]
            tr, va, te = group_split(data.groups, data.y, s["test_frac"],
                                     s["val_frac"], seed)
            tr = _cap_train_groupwise(tr, data.groups,
                                      cfgs["quantum"].get("max_train", 3000),
                                      seed)
            # cap the ~27k-window test split group-wise to keep the quantum
            # test Gram tractable; the SAME capped test set serves every model
            # (matched comparison; n_test recorded in the CSV)
            te = _cap_train_groupwise(te, data.groups, 6000, seed + 1)
            rpath = CK / f"reduced_s{seed}.npz"
            if not rpath.exists():
                _check_budget()
                red = fit_reducer(cfgs["features"]["reduction"], data.X,
                                  data.blocks, tr)
                _savez_atomic(rpath, Xr=red.transform(data.X))
            Xr = np.load(rpath)["Xr"]
            idx = {"tr": tr, "va": va, "te": te}
            for model in MODELS:
                rows.append(_eval(model, data, Xr, idx, cfgs, seed))
    except BudgetExceeded as exc:
        print(json.dumps({"status": "in_progress", "reason": str(exc),
                          "n_checkpoints": len(list(CK.iterdir()))}))
        return

    res = pd.DataFrame([r["row"] for r in rows])
    res.to_csv(OUT / "binary_results.csv", index=False)
    agg = []
    for model, g in res.groupby("model"):
        agg.append({"model": model, "n_seeds": g.seed.nunique(),
                    **{f"{m}_mean": g[m].mean() for m in
                       ["pr_auc", "roc_auc", "f1", "mcc"]},
                    **{f"{m}_std": g[m].std(ddof=1) for m in
                       ["pr_auc", "roc_auc", "f1", "mcc"]}})
    pd.DataFrame(agg).to_csv(OUT / "multiseed.csv", index=False)

    # Holm-corrected paired significance (seed 7, qsvm_crosslayer vs classical)
    s7 = {r["row"]["model"]: r for r in rows if r["row"]["seed"] == 7}
    base = s7["qsvm_crosslayer"]
    sig, raw = [], {"pr_auc": {}, "roc_auc": {}}
    for m in ["rbf_svm", "random_forest", "xgboost"]:
        for metric in ("pr_auc", "roc_auc"):
            cpath = CK / f"sig_{m}_{metric}.json"
            if cpath.exists():
                t = json.loads(cpath.read_text())
            else:
                _check_budget()
                t = paired_bootstrap_auc_diff(
                    np.array(base["y"]), np.array(base["scores"]),
                    np.array(s7[m]["scores"]), np.array(base["groups"]),
                    n_boot=N_BOOT, seed=7, metric=metric)
                cpath.write_text(json.dumps(t, default=float))
            raw[metric][m] = t["p_two_sided"]
            sig.append({"comparison": f"qsvm_crosslayer_vs_{m}",
                        "metric": metric, "delta": t["delta"],
                        "delta_lo": t["lo"], "delta_hi": t["hi"],
                        "p_raw": t["p_two_sided"]})
    for metric in ("pr_auc", "roc_auc"):
        adj = holm_bonferroni(raw[metric])
        for r in sig:
            if r["metric"] == metric:
                r["p_holm"] = adj[r["comparison"].replace(
                    "qsvm_crosslayer_vs_", "")]
    pd.DataFrame(sig).to_csv(OUT / "significance.csv", index=False)
    print(pd.DataFrame(agg).round(3).to_string(index=False))
    print(pd.DataFrame(sig).round(4).to_string(index=False))
    print("status: done")


if __name__ == "__main__":
    main()
