#!/usr/bin/env python3
"""Adaptive-adversary (mimicry evasion) study on VeReMi NextGen.

Threat model: a detector-aware attacker who controls only what it TRANSMITS.
It can make its claimed kinematics arbitrarily smooth and its claimed positions
map-plausible (the attacker-CONTROLLABLE window features), but it cannot control
receiver-side channel observables (delay statistics, rates, inter-arrivals) or
what OTHER traffic does (Sybil-signature context). Evasion is implemented as a
best-of-K MIMICRY attack: for each attack window, the controllable features are
replaced by those of K=32 randomly drawn BENIGN TRAINING windows (guaranteed
plausibility — the values are real benign behaviour) and the candidate that
minimises the target model's score is kept, per model (white-box selection).

Two envelopes:
  conservative — mobility block + off-road plausibility (the smooth, on-road liar)
  strong       — + e_xl (delay-consistent claims) + claim-overlap avoidance

Reported per (model, envelope, attack): recall at the model's ORIGINAL operating
threshold before/after evasion (FPR unchanged by construction: benign windows
untouched). This is an upper bound on feature-space evasion under the stated
envelope; attack-goal degradation (a fully benign-looking falsification carries
less utility) is discussed, not modelled.

Checkpointed under results/evasion/checkpoints (QKV2X_BUDGET_S aware).

Optional flags — all opt-in, defaults reproduce the published run exactly:
  --k-cand N  candidate count K (default 32 = published)
  --seed S  RNG seed (default 7 = published)
  --common-donor  transfer setting: candidates are selected ONCE against a
                    surrogate model (--surrogate, default random_forest) and the
                    SAME adversarial realisation is evaluated on every model
                    (vs the default per-model white-box best-of-K)
  --out-suffix STR  appended to checkpoint/result filenames so sweeps never
                    collide with the published checkpoints
  --dump-scores  additionally save per-window s_before/s_after (+ attack name,
                    transmitter group, threshold) to results/evasion/
                    scores_<envelope>_<model><suffix>.npz, so recall can be given
                    grouped-bootstrap confidence intervals (scripts/studies/mimicry_ci.py)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from qkv2x.config import load_yaml                                  # noqa: E402
from qkv2x.experiments.common import make_feature_map               # noqa: E402
from qkv2x.experiments.veremi_nextgen_exp import (BudgetExceeded,
                                                  _check_budget)    # noqa: E402
from qkv2x.features.reduce import BlockReducer                      # noqa: E402
from qkv2x.models.baselines import baseline_scores, build_baseline  # noqa: E402
from qkv2x.models.qsvm import QuantumKernelSVM                      # noqa: E402
from qkv2x.quantum.kernels import fidelity_gram                     # noqa: E402

OUT = REPO / "results" / "evasion"
CK = OUT / "checkpoints"
NCK = REPO / "results" / "veremi_nextgen" / "checkpoints"
SEED, K_CAND = 7, 32

MOBILITY = ["e_kin_mean", "e_kin_max", "e_kin_std", "speed_consistency_mean",
            "speed_consistency_max", "heading_consistency_mean",
            "accel_abs_mean"]
ENVELOPES = {
    "conservative": MOBILITY + ["offroad_max", "offroad_mean"],
    "strong": MOBILITY + ["offroad_max", "offroad_mean", "e_xl_mean",
                          "e_xl_max", "e_xl_std", "overlap_3m_max",
                          "overlap_3m_mean"],
}


def _load():
    z = np.load(NCK / "assembled.npz", allow_pickle=True)
    names = [str(s) for s in z["feature_names"]]
    blocks = [str(s) for s in z["blocks"]]
    split = z["split"].astype(str)
    return (z["X"], z["y"], z["aoi"], z["attack_name"].astype(str), names,
            blocks, np.where(split == "Train")[0], np.where(split == "Test")[0])


def _candidates(X, env_cols, atk_idx, benign_tr_idx, rng, k=K_CAND):
    """[n_atk, K, F] mimicry candidates: controllable cols swapped wholesale."""
    n, F = len(atk_idx), X.shape[1]
    donors = rng.choice(benign_tr_idx, size=(n, k))
    cand = np.repeat(X[atk_idx][:, None, :], k, axis=1)
    cand[:, :, env_cols] = X[donors][:, :, env_cols]
    return cand


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k-cand", type=int, default=K_CAND)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--common-donor", action="store_true")
    ap.add_argument("--surrogate", default="random_forest",
                    choices=["random_forest", "xgboost"])
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--dump-scores", action="store_true")
    args = ap.parse_args()
    k_cand, suffix = int(args.k_cand), str(args.out_suffix)

    CK.mkdir(parents=True, exist_ok=True)
    X, y, aoi, atk_name, names, blocks, tr, te = _load()
    cfgs = {"features": load_yaml(str(REPO / "configs/features_veremi_nextgen.yaml")),
            "quantum": load_yaml(str(REPO / "configs/quantum.yaml")),
            "baselines": load_yaml(str(REPO / "configs/baselines.yaml"))}
    rng = np.random.default_rng(int(args.seed))

    reducer = BlockReducer(cfgs["features"]["reduction"]).fit(X[tr], blocks)
    Xr = reducer.transform(X)
    atk_te = te[y[te] == 1]
    benign_tr = tr[y[tr] == 0]

    # --- model handles: score functions + original thresholds -----------------
    import torch
    st = torch.load(NCK / "qtrain_binary_full_qsvm_crosslayer.pt",
                    weights_only=False)
    qparams = {k: v for k, v in st["best_params"].items()}
    fm = make_feature_map("qsvm_crosslayer", cfgs["quantum"],
                          cfgs["features"]["reduction"], cfgs["features"])
    Kz = np.load(NCK / "gram_binary_full_qsvm_crosslayer.npz")
    qsvm = QuantumKernelSVM(C_grid=cfgs["quantum"]["svm"]["C_grid"],
                            cv_folds=cfgs["quantum"]["svm"]["cv_folds"])
    qsvm.fit(Kz["K_tr"], y[tr], groups=None)
    s_tr_states = fm.statevectors(Xr[tr], aoi[tr], qparams)

    def q_score(Xraw, aoi_rows):
        s = fm.statevectors(reducer.transform(Xraw), aoi_rows, qparams)
        return qsvm.decision_function(fidelity_gram(s, s_tr_states))

    handles = {"qsvm_crosslayer": (q_score, 0.0)}
    for m in ("random_forest", "xgboost"):
        import joblib
        mp = CK / f"model_{m}.joblib"
        if mp.exists():
            est = joblib.load(mp)
        else:
            _check_budget()
            est = build_baseline(m, cfgs["baselines"], seed=SEED)
            est.fit(Xr[tr], y[tr])
            joblib.dump(est, mp)
        handles[m] = (
            lambda Xraw, _a, e=est: baseline_scores(e, reducer.transform(Xraw)),
            0.5)

    rows = []
    try:
        for env_name, env_feats in ENVELOPES.items():
            env_cols = [names.index(f) for f in env_feats]
            cpath = CK / f"cand_{env_name}{suffix}.npz"
            if cpath.exists():
                cand = np.load(cpath)["cand"]
                if cand.shape[1] != k_cand:
                    raise SystemExit(
                        f"cached {cpath.name} has K={cand.shape[1]} != --k-cand "
                        f"{k_cand}; pass a distinct --out-suffix for sweeps")
            else:
                _check_budget()
                cand = _candidates(X, env_cols, atk_te, benign_tr, rng, k_cand)
                np.savez_compressed(cpath, cand=cand)
            n, K, F = cand.shape

            # common-donor (transfer) mode: pick each window's candidate ONCE
            # against the surrogate; every model then faces the SAME realisation.
            sel = None
            if args.common_donor:
                spath = CK / f"sel_{env_name}_{args.surrogate}{suffix}.npz"
                if spath.exists():
                    sel = np.load(spath)["sel"]
                else:
                    _check_budget()
                    sur_fn = handles[args.surrogate][0]
                    aoi_rep = np.repeat(aoi[atk_te], K)
                    s_sur = sur_fn(cand.reshape(n * K, F), aoi_rep).reshape(n, K)
                    sel = np.argmin(s_sur, axis=1)
                    np.savez_compressed(spath, sel=sel)
                X_adv = cand[np.arange(n), sel]

            for model, (score_fn, thr) in handles.items():
                rpath = CK / f"res_{env_name}_{model}{suffix}.json"
                if rpath.exists():
                    rows.extend(json.loads(rpath.read_text()))
                    continue
                _check_budget()
                s_before = score_fn(X[atk_te], aoi[atk_te])
                if args.common_donor:
                    s_after = score_fn(X_adv, aoi[atk_te])
                else:
                    aoi_rep = np.repeat(aoi[atk_te], K)
                    s_cand = score_fn(cand.reshape(n * K, F),
                                      aoi_rep).reshape(n, K)
                    s_after = s_cand.min(axis=1)      # white-box best-of-K
                extra = ({"k_cand": K, "selection": f"surrogate:{args.surrogate}"}
                         if args.common_donor else
                         ({"k_cand": K, "selection": "per_model"} if suffix else {}))
                mrows = []
                for a in sorted(set(atk_name[atk_te])):
                    m_ = atk_name[atk_te] == a
                    mrows.append({
                        "envelope": env_name, "model": model, "attack": a,
                        "n": int(m_.sum()),
                        "recall_before": float((s_before[m_] > thr).mean()),
                        "recall_after": float((s_after[m_] > thr).mean()), **extra})
                mrows.append({
                    "envelope": env_name, "model": model, "attack": "ALL",
                    "n": int(n),
                    "recall_before": float((s_before > thr).mean()),
                    "recall_after": float((s_after > thr).mean()), **extra})
                rpath.write_text(json.dumps(mrows, default=float))
                rows.extend(mrows)
                if args.dump_scores:
                    zz = np.load(NCK / "assembled.npz", allow_pickle=True)
                    np.savez_compressed(
                        OUT / f"scores_{env_name}_{model}{suffix}.npz",
                        s_before=np.asarray(s_before, float),
                        s_after=np.asarray(s_after, float),
                        attack=atk_name[atk_te].astype(str),
                        groups=np.asarray(zz["groups"])[atk_te],
                        idx=np.asarray(atk_te), thr=float(thr), k_cand=int(K))
    except BudgetExceeded as exc:
        print(json.dumps({"status": "in_progress", "reason": str(exc)}))
        return

    res = pd.DataFrame(rows)
    res.to_csv(OUT / f"evasion_results{suffix}.csv", index=False)
    print(res[res.attack == "ALL"].round(3).to_string(index=False))
    print("status: done")


if __name__ == "__main__":
    main()
