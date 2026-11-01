"""Nyström-at-scale on dev_grid + VeReMi.

The dense quantum-vs-classical comparison is limited by the O(N^2) Gram cap (N~3k), while classical
baselines can use the full N. The Nyström landmark kernel (Phi = features from K~=C W^+ C^T, O(N*l)) lets the
trained quantum kernel scale far past that cap. NextGen already showed no-advantage holds at N=20k. This extends that to dev_grid and VeReMi: run qsvm_crosslayer / qsvm_zz via Nyström AND
random_forest / xgboost at the SAME large N, on the SAME held-out test set, and compare.

Writes results/nystrom_scale/scale.json + scale.csv.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.svm import LinearSVC

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _study_lib import REPO, RESULTS, now_iso  # noqa: E402

ELL = 512          # Nyström landmarks
N_SCALE = 12000    # large train N (>> the ~3k dense cap)


def _nystrom_quantum(model, data, Xr, idx, cfgs, seed=13):
    from qkv2x.experiments.common import make_feature_map, _maybe_train_params
    from qkv2x.quantum.kernels import fidelity_gram, nystrom_features
    tr, va, te = idx["tr"], idx["va"], idx["te"]
    fm = make_feature_map(model, cfgs["quantum"], cfgs["features"]["reduction"], cfgs["features"])
    params, _ = _maybe_train_params(fm, model, cfgs["quantum"], Xr[tr], data.y[tr], data.aoi[tr])
    rng = np.random.default_rng(seed)
    lm = np.sort(rng.choice(len(tr), min(ELL, len(tr)), replace=False))
    s_tr = fm.statevectors(Xr[tr], data.aoi[tr], params)
    s_lm = s_tr[lm]
    W = fidelity_gram(s_lm)
    Phi_tr = nystrom_features(fidelity_gram(s_tr, s_lm), W); del s_tr
    s_va = fm.statevectors(Xr[va], data.aoi[va], params)
    Phi_va = nystrom_features(fidelity_gram(s_va, s_lm), W); del s_va
    s_te = fm.statevectors(Xr[te], data.aoi[te], params)
    Phi_te = nystrom_features(fidelity_gram(s_te, s_lm), W); del s_te
    best = (-np.inf, None)
    for C in (0.01, 0.1, 1.0, 10.0):
        svm = LinearSVC(C=C, class_weight="balanced", max_iter=5000).fit(Phi_tr, data.y[tr])
        ap = average_precision_score(data.y[va], svm.decision_function(Phi_va))
        if ap > best[0]:
            best = (ap, svm)
    sc = best[1].decision_function(Phi_te)
    return float(average_precision_score(data.y[te], sc)), float(roc_auc_score(data.y[te], sc))


def _classical(model, data, Xr, idx):
    tr, te = idx["tr"], idx["te"]
    if model == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=7)
    else:  # xgboost
        from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, n_jobs=-1, eval_metric="logloss")
    clf.fit(Xr[tr], data.y[tr])
    sc = clf.predict_proba(Xr[te])[:, 1]
    return float(average_precision_score(data.y[te], sc)), float(roc_auc_score(data.y[te], sc))


def _cap(idx, n_tr, n_va, n_te, seed):
    rng = np.random.default_rng(seed)
    def c(a, n):
        return np.sort(rng.choice(a, n, replace=False)) if len(a) > n else a
    return {"tr": c(idx["tr"], n_tr), "va": c(idx["va"], n_va), "te": c(idx["te"], n_te)}


def _load_dev_grid(seed=7):
    from qkv2x.config import load_yaml
    from qkv2x.experiments.common import load_experiment_data, group_split, fit_reducer
    cfg = load_yaml(REPO / "configs" / "features.yaml")
    data = load_experiment_data(REPO / "runs" / "dev_grid", cfg)
    sp = cfg["splits"]
    tr, va, te = group_split(data.groups, data.y, sp["test_frac"], sp["val_frac"], sp["seed"])
    reducer = fit_reducer(cfg["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)
    idx = _cap({"tr": tr, "va": va, "te": te}, N_SCALE, 3000, 4000, seed)
    return data, Xr, idx, {"quantum": load_yaml(REPO / "configs" / "quantum.yaml"), "features": cfg}


def _load_veremi(seed=7):
    from qkv2x.config import load_yaml
    from qkv2x.data.veremi import load_veremi
    from qkv2x.features.dataset import build_dataset
    from qkv2x.experiments.common import ExperimentData, fit_reducer, group_split
    from qkv2x.experiments.veremi_exp import estimate_propagation
    cfg = load_yaml(REPO / "configs" / "experiments" / "veremi.yaml")
    feats = load_yaml(REPO / cfg["features_config"])
    rx = load_veremi(str(REPO / cfg["veremi_root"]))
    prop = estimate_propagation(rx) if cfg.get("estimate_propagation", True) else feats["propagation"]
    ds = build_dataset(rx, feats, prop, float(cfg.get("nominal_rate_hz", 1.0)))
    max_w = 18000
    rng = np.random.default_rng(feats["splits"]["seed"])
    sel = rng.choice(len(ds.y), size=min(max_w, len(ds.y)), replace=False)
    data = ExperimentData(X=ds.X[sel], y=ds.y[sel], attack_name=ds.attack_name[sel],
                          groups=ds.groups[sel], aoi=ds.aoi[sel], t_end=ds.meta["t_end"].to_numpy()[sel],
                          feature_names=ds.feature_names, blocks=ds.blocks,
                          points=pd.DataFrame(), meta=pd.DataFrame())
    sp = feats["splits"]
    tr, va, te = group_split(data.groups, data.y, sp["test_frac"], sp["val_frac"], sp["seed"])
    reducer = fit_reducer(feats["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)
    idx = _cap({"tr": tr, "va": va, "te": te}, N_SCALE, 3000, 4000, seed)
    return data, Xr, idx, {"quantum": load_yaml(REPO / "configs" / "quantum.yaml"), "features": feats}


def _run_dataset(name, loader):
    t0 = time.time()
    data, Xr, idx, cfgs = loader()
    n_tr = len(idx["tr"]); n_te = len(idx["te"])
    print(f"[{name}] N_train={n_tr} N_test={n_te} (Nyström l={ELL}; classical at same N)")
    res = {}
    for m in ("qsvm_crosslayer", "qsvm_zz"):
        pr, roc = _nystrom_quantum(m, data, Xr, idx, cfgs)
        res[m] = {"pr_auc": pr, "roc_auc": roc, "kind": "quantum_nystrom"}
        print(f"  {m:16s} (nystrom)  PR={pr:.3f} ROC={roc:.3f}")
    for m in ("random_forest", "xgboost"):
        pr, roc = _classical(m, data, Xr, idx)
        res[m] = {"pr_auc": pr, "roc_auc": roc, "kind": "classical_fullN"}
        print(f"  {m:16s} (classical) PR={pr:.3f} ROC={roc:.3f}")
    cl = res["qsvm_crosslayer"]["roc_auc"]; zz = res["qsvm_zz"]["roc_auc"]
    best_c = max(("random_forest", "xgboost"), key=lambda m: res[m]["roc_auc"])
    res["_summary"] = {
        "n_train": n_tr, "n_test": n_te,
        "crosslayer_beats_zz": bool(cl > zz),
        "best_classical": best_c, "best_classical_roc": res[best_c]["roc_auc"],
        "qsvm_crosslayer_roc": cl,
        "no_quantum_advantage_at_scale": bool(res[best_c]["roc_auc"] >= cl),
        "wall_s": round(time.time() - t0, 1),
    }
    return res


def main() -> None:
    out = RESULTS / "nystrom_scale"; out.mkdir(parents=True, exist_ok=True)
    started = now_iso()
    allres = {"study": "nystrom_at_scale", "started": started,
              "params": {"landmarks": ELL, "n_scale": N_SCALE},
              "note": ("quantum kernels evaluated via Nyström (O(N*l)) at N>>dense-cap; classical at "
                       "same N on same test split. Closes the matched-N artifact on dev_grid+VeReMi "
                       "(NextGen already settled at N=20k, earlier)."),
              "datasets": {}}
    for name, loader in (("dev_grid", _load_dev_grid), ("veremi", _load_veremi)):
        try:
            allres["datasets"][name] = _run_dataset(name, loader)
        except Exception as exc:
            print(f"[{name}] FAILED: {type(exc).__name__}: {exc}")
            allres["datasets"][name] = {"error": f"{type(exc).__name__}: {exc}"}
    (out / "scale.json").write_text(json.dumps(allres, indent=2))
    rows = []
    for ds, r in allres["datasets"].items():
        for m, v in r.items():
            if m.startswith("_") or "pr_auc" not in v:
                continue
            rows.append({"dataset": ds, "model": m, **{k: v[k] for k in ("pr_auc", "roc_auc", "kind")}})
    pd.DataFrame(rows).to_csv(out / "scale.csv", index=False)
    print("\n=== Nyström-at-scale summary ===")
    for ds, r in allres["datasets"].items():
        s = r.get("_summary")
        if s:
            print(f"  {ds}: cross-layer ROC={s['qsvm_crosslayer_roc']:.3f} vs best classical "
                  f"{s['best_classical']}={s['best_classical_roc']:.3f} -> "
                  f"no_advantage_at_scale={s['no_quantum_advantage_at_scale']} (N={s['n_train']})")


if __name__ == "__main__":
    main()
