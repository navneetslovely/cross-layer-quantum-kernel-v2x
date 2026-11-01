#!/usr/bin/env python3
"""param_sensitivity — sensitivity of the cross-layer QSVM to its fixed parameter choices.

Protocol: dev_grid, seed-7 leakage-safe split, train group-capped at
quantum.yaml::max_train (-> 3,023), test group-capped at 6,000 (seed 8) — the same
protocol as the published topology ablation (entanglement_ablation), so the default configuration
reproduces the published seed-7 ROC-AUC (0.9254). One parameter is changed at a time;
everything else is the published default (KTA seed 11, constant init, best-batch rule).

Sweeps: layers L in {1,2,3}; register sizes (n_M,n_N) in {(2,2),(3,3),(4,4),(6,6)};
AoI constant kappa in {0.01, 0.1, 0.5 s, off}; batch in {256,512,1024};
learning rate in {0.01,0.05,0.1}; epochs in {50,100,200}.
Also: cumulative PCA explained variance per block for 1..7 components.

Outputs (results/studies/): param_sensitivity.csv, pca_variance.csv,
param_sensitivity_summary.json. Checkpoints: param_sensitivity_ckpt/.
"""
from __future__ import annotations

import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qkv2x.config import load_yaml                                   # noqa: E402
from qkv2x.experiments.common import (ExperimentData, _cap_train_groupwise,
                                      fit_reducer, group_split,
                                      make_feature_map)              # noqa: E402
from qkv2x.features.crosslayer import FEATURE_SPEC                   # noqa: E402
from qkv2x.models.metrics import detection_report, grouped_bootstrap_ci  # noqa: E402
from qkv2x.models.qsvm import QuantumKernelSVM                       # noqa: E402
from qkv2x.quantum.kernels import fidelity_gram                      # noqa: E402
from qkv2x.quantum.train import train_feature_map                    # noqa: E402

OUT = REPO / "results" / "studies"
CK = OUT / "param_sensitivity_ckpt"
SEED = 7
N_BOOT = 200

CONFIGS = [
    ("default", {}),
    ("L=1", {"n_layers": 1}), ("L=3", {"n_layers": 3}),
    ("reg=2+2", {"n_mobility": 2, "n_network": 2}),
    ("reg=3+3", {"n_mobility": 3, "n_network": 3}),
    ("reg=6+6", {"n_mobility": 6, "n_network": 6}),
    ("kappa=0.01", {"kappa": 0.01}), ("kappa=0.1", {"kappa": 0.1}), ("aoi=off", {"aoi_modulation": False}),
    ("batch=256", {"batch": 256}), ("batch=1024", {"batch": 1024}),
    ("lr=0.01", {"lr": 0.01}), ("lr=0.1", {"lr": 0.1}),
    ("epochs=50", {"epochs": 50}), ("epochs=100", {"epochs": 100}),
]


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


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def _pca_variance(data, tr, reduction_cfg):
    """Cumulative explained-variance ratio per block after the same clipping and
    standardisation as BlockReducer (clip quantiles, z-score), for 1..n_block comps."""
    from sklearn.decomposition import PCA
    lo, hi = reduction_cfg.get("clip_quantiles", [0.005, 0.995])
    rows = []
    blocks = np.asarray(data.blocks)
    for b in sorted(set(blocks)):
        cols = np.where(blocks == b)[0]
        Xb = data.X[tr][:, cols].astype(float)
        ql, qh = np.quantile(Xb, lo, axis=0), np.quantile(Xb, hi, axis=0)
        Xb = np.clip(Xb, ql, qh)
        mu, sd = Xb.mean(0), Xb.std(0) + 1e-12
        Xs = (Xb - mu) / sd
        pca = PCA(n_components=min(Xs.shape[1], 12), random_state=0).fit(Xs)
        cum = np.cumsum(pca.explained_variance_ratio_)
        for k, c in enumerate(cum, start=1):
            rows.append({"block": b, "n_components": k, "cum_explained_variance": float(c),
                         "n_raw_features": int(len(cols))})
    return pd.DataFrame(rows)


def main():
    CK.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data = _load()
    base = {"features": load_yaml(str(REPO / "configs/features.yaml")),
            "quantum": load_yaml(str(REPO / "configs/quantum.yaml")),
            "baselines": load_yaml(str(REPO / "configs/baselines.yaml"))}
    s = base["features"]["splits"]
    tr, va, te = group_split(data.groups, data.y, s["test_frac"], s["val_frac"], SEED)
    tr = _cap_train_groupwise(tr, data.groups, base["quantum"].get("max_train", 3000), SEED)
    te = _cap_train_groupwise(te, data.groups, 6000, SEED + 1)

    pv = _pca_variance(data, tr, base["features"]["reduction"])
    pv.to_csv(OUT / "pca_variance.csv", index=False)

    reducers = {}
    rows = []
    for name, ov in CONFIGS:
        ck = CK / f"{name.replace('=', '_').replace('+', 'p')}.json"
        if ck.exists():
            rows.append(json.loads(ck.read_text()))
            print(f"[param_sensitivity] {name:12s} cached ROC={rows[-1]['roc_auc']:.4f}", flush=True)
            continue
        cfgs = copy.deepcopy(base)
        red = cfgs["features"]["reduction"]
        red["n_mobility"] = int(ov.get("n_mobility", red["n_mobility"]))
        red["n_network"] = int(ov.get("n_network", red["n_network"]))
        cfgs["features"]["aoi"]["kappa"] = float(ov.get("kappa", cfgs["features"]["aoi"]["kappa"]))
        fmc = cfgs["quantum"]["feature_map"]
        fmc["n_layers"] = int(ov.get("n_layers", fmc["n_layers"]))
        fmc["aoi_modulation"] = bool(ov.get("aoi_modulation", fmc["aoi_modulation"]))
        tcfg = cfgs["quantum"]["train"]
        for k in ("batch", "lr", "epochs"):
            if k in ov:
                tcfg[k] = ov[k]
        key = (red["n_mobility"], red["n_network"])
        if key not in reducers:
            reducers[key] = fit_reducer(red, data.X, data.blocks, tr).transform(data.X)
        Xr = reducers[key]
        fm = make_feature_map("qsvm_crosslayer", cfgs["quantum"], red, cfgs["features"])
        t0 = time.time()
        params, hist = train_feature_map(fm, Xr[tr], data.y[tr], data.aoi[tr], tcfg)
        t_train = time.time() - t0
        with torch.no_grad():
            s_tr = fm.statevectors(Xr[tr], data.aoi[tr], params)
            K_tr = fidelity_gram(s_tr)
            K_te = fidelity_gram(fm.statevectors(Xr[te], data.aoi[te], params), s_tr)
        qc = cfgs["quantum"]
        svm = QuantumKernelSVM(C_grid=qc["svm"]["C_grid"], cv_folds=qc["svm"]["cv_folds"])
        svm.fit(K_tr, data.y[tr], groups=data.groups[tr])
        scores, preds = svm.decision_function(K_te), svm.predict(K_te)
        rep = detection_report(data.y[te], scores, preds)
        ci = grouped_bootstrap_ci(data.y[te], scores, data.groups[te], n_boot=N_BOOT, seed=SEED)
        row = {"config": name, "overrides": json.dumps(ov), "n_qubits": fm.n,
               "n_layers": fmc["n_layers"], "n_params": int(sum(v.numel() for v in params.values())),
               "roc_auc": rep["roc_auc"], "pr_auc": rep["pr_auc"], "f1": rep.get("f1"), "mcc": rep.get("mcc"),
               "roc_lo": ci["roc_auc"]["lo"], "roc_hi": ci["roc_auc"]["hi"],
               "best_alignment": hist["best_alignment"], "C": svm.best_C_,
               "train_seconds": t_train, "n_train": int(len(tr)), "n_test": int(len(te))}
        ck.write_text(json.dumps(row, default=float))
        rows.append(row)
        print(f"[param_sensitivity] {name:12s} ROC={row['roc_auc']:.4f} PR={row['pr_auc']:.4f} "
              f"align={row['best_alignment']:.3f} ({t_train:.0f}s)", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "param_sensitivity.csv", index=False)
    d0 = float(df[df.config == "default"].roc_auc.iloc[0])
    df["delta_roc_vs_default"] = df.roc_auc - d0
    df.to_csv(OUT / "param_sensitivity.csv", index=False)
    summ = {"protocol": {"dataset": "dev_grid", "split_seed": SEED, "train_cap": base["quantum"].get("max_train", 3000),
                         "test_cap": 6000, "kta_seed": base["quantum"]["train"]["seed"], "rule": "best training batch",
                         "n_boot": N_BOOT, "interpreter": sys.executable, "git_head": _git_head(),
                         "runtime_s": time.time() - t_start},
            "default_roc": d0, "published_seed7_roc": 0.9254389652792645,
            "rows": df.to_dict(orient="records"),
            "pca_cum_variance_at_default": {b: float(pv[(pv.block == b) & (pv.n_components == (4 if True else 4))].cum_explained_variance.iloc[0])
                                            for b in pv.block.unique()}}
    (OUT / "param_sensitivity_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(df[["config", "n_qubits", "roc_auc", "pr_auc", "delta_roc_vs_default", "best_alignment", "train_seconds"]].round(4).to_string(index=False))
    print(json.dumps({"status": "done", "runtime_s": summ["protocol"]["runtime_s"]}))


if __name__ == "__main__":
    main()
