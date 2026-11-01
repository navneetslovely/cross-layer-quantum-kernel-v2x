"""qubit_scaling.py -- qubit-count scaling of the trained vs fixed cross-layer map on VeReMi NextGen
(Section VII-B of the paper: trained ROC-AUC vs n, with kernel variance against the Haar floor).

n in {4,...,12} is run here on CPU (5 training seeds, 200 epochs); n in {14,...,20} was run with the
pure-torch engine in scripts/gpu_engine/ (prep_scaling_inputs.py -> scaling_gpu.py ->
postprocess_scaling.py), which appends rows in exactly the same format. `--merge` aggregates all
scaling_raw_n*.csv files into results/qubit_scaling/scaling.{csv,json}.

Usage:  python scripts/qubit_scaling.py --only-n 8      # one qubit count (parallelisable)
        python scripts/qubit_scaling.py --merge          # aggregate -> scaling.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _study_lib import REPO, RESULTS, now_iso, save_table, write_metadata  # noqa: E402

OUTDIR = RESULTS / "qubit_scaling"

# ---- parameters ----
SEEDS = [11, 23, 41, 57, 71]
SCALING_N_GRID = [4, 6, 8, 10, 12]
SCALING_N_EVAL = 400
SCALING_EPOCHS = 200


# ---------------------------------------------------------------- shared helpers
def _eff_dim(K):
    ev = np.clip(np.real(np.linalg.eigvalsh(K)), 0, None)
    s = ev.sum()
    return float(s * s / np.sum(ev * ev)) if s > 0 else 0.0


def _load():
    from qkv2x.experiments.common import ExperimentData
    z = np.load(RESULTS / "veremi_nextgen" / "checkpoints" / "assembled.npz", allow_pickle=True)
    data = ExperimentData(
        X=z["X"], y=z["y"].astype(int), attack_name=z["attack_name"].astype(str),
        groups=z["groups"].astype(int), aoi=z["aoi"], t_end=z["t_end"],
        feature_names=[str(s) for s in z["feature_names"]],
        blocks=[str(s) for s in z["blocks"]], points=pd.DataFrame(), meta=pd.DataFrame())
    split = z["split"].astype(str)
    idx = {"tr": np.where(split == "Train")[0], "te": np.where(split == "Test")[0]}
    return data, idx


def _balanced(y, pool, n, rng):
    pool = np.asarray(pool)
    pos, neg = pool[y[pool] == 1], pool[y[pool] == 0]
    k = min(n // 2, len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    rng.shuffle(sel)
    return sel


def _make_fm(n_mob, n_net, entangler, quantum_cfg):
    from qkv2x.quantum.featuremap import CrossLayerFeatureMap
    fmcfg = quantum_cfg["feature_map"]
    return CrossLayerFeatureMap(n_mobility=n_mob, n_network=n_net,
                                n_layers=int(fmcfg.get("n_layers", 2)), entangler=entangler)


def _reduce(data, red_cfg, idx_tr):
    from qkv2x.experiments.common import fit_reducer
    return fit_reducer(red_cfg, data.X, data.blocks, idx_tr).transform(data.X)


def _eval_metrics(fm, params, Xb, ab, n_tr, y_tr, y_te):
    from qkv2x.quantum.kernels import fidelity_gram
    from qkv2x.quantum.diagnostics import centered_kta
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score, average_precision_score
    K = np.asarray(fidelity_gram(fm.statevectors(Xb, ab, params, chunk=256)), float)
    iu = np.triu_indices_from(K, k=1)
    sigma_K2 = float(K[iu].var())
    d_eff = _eff_dim(K)
    kta = float(centered_kta(K[n_tr:, n_tr:], np.where(y_te > 0, 1.0, -1.0)))
    svc = SVC(kernel="precomputed", C=1.0).fit(K[:n_tr, :n_tr], y_tr)
    sc = svc.decision_function(K[n_tr:, :n_tr])
    return (sigma_K2, d_eff, kta, float(roc_auc_score(y_te, sc)),
            float(average_precision_score(y_te, sc)), K)


class RawWriter:
    """Append raw rows to disk incrementally (never lose a heavy run)."""
    def __init__(self, path):
        self.path = Path(path); self.header = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.path.unlink()

    def add(self, row: dict):
        df = pd.DataFrame([row])
        df.to_csv(self.path, mode="a", header=not self.header, index=False)
        self.header = True


# ---------------------------------------------------------------- study: trajectory (multi-seed)
def _scaling_one_n(n, quantum_cfg, base_red, data, idx, raw):
    from qkv2x.quantum.train import train_feature_map
    nm, nn = n // 2, n - n // 2
    red = copy.deepcopy(base_red); red["n_mobility"], red["n_network"] = nm, nn
    Xr = _reduce(data, red, idx["tr"]); aoi = np.asarray(data.aoi)  # reducer independent of seed
    d = 2.0 ** n; haar_var = (d - 1) / (d ** 2 * (d + 1))
    out = []
    for training in ["fixed", "kta_trained"]:
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            tr_e = _balanced(data.y, idx["tr"], SCALING_N_EVAL, rng)
            te_e = _balanced(data.y, idx["te"], SCALING_N_EVAL, rng)
            both = np.concatenate([tr_e, te_e]); n_tr = len(tr_e)
            fm = _make_fm(nm, nn, "cross_layer", quantum_cfg)
            if training == "fixed":
                params = fm.fixed_params()
            else:
                params, _ = train_feature_map(
                    fm, Xr[idx["tr"]], data.y[idx["tr"]], aoi[idx["tr"]],
                    {"seed": sd, "lr": 0.05, "epochs": SCALING_EPOCHS, "batch": 256, "chunk": 256,
                     "init_w": 1.0, "init_u": 1.0, "init_gamma": 0.5})
            s2, de, kta, roc, pr, _ = _eval_metrics(fm, params, Xr[both], aoi[both], n_tr,
                                                    data.y[tr_e], data.y[te_e])
            shots95 = float(0.25 / max((np.sqrt(s2) * np.sqrt(1 / 0.95 ** 2 - 1)) ** 2, 1e-15))
            r = {"n_qubits": n, "training": training, "seed": sd, "sigma_K2": s2, "d_eff": de,
                 "kta_eval": kta, "roc_auc": roc, "pr_auc": pr, "haar_var_floor": haar_var,
                 "shots_for_r0.95": shots95, "epochs": SCALING_EPOCHS, "n_eval": SCALING_N_EVAL}
            out.append(r); raw.add(r)
            print(f"  n={n} {training:11s} s{sd}: sigmaK2={s2:.4f} d_eff={de:7.1f} ROC={roc:.3f}", flush=True)
    return out


def study_scaling_only_n(n):
    from qkv2x.config import load_yaml
    quantum_cfg = load_yaml(REPO / "configs" / "quantum.yaml")
    base_red = load_yaml(REPO / "configs" / "features_veremi_nextgen.yaml")["reduction"]
    data, idx = _load()
    raw = RawWriter(OUTDIR / f"scaling_raw_n{n}.csv")
    _scaling_one_n(n, quantum_cfg, base_red, data, idx, raw)
    print(f"[scaling n={n}] raw -> {OUTDIR}/scaling_raw_n{n}.csv", flush=True)


def _scaling_aggregate(df):
    agg = df.groupby(["n_qubits", "training"]).agg(
        sigma_K2_mean=("sigma_K2", "mean"), sigma_K2_std=("sigma_K2", "std"),
        d_eff_mean=("d_eff", "mean"), d_eff_std=("d_eff", "std"),
        roc_mean=("roc_auc", "mean"), roc_std=("roc_auc", "std"),
        pr_mean=("pr_auc", "mean"), haar_var_floor=("haar_var_floor", "first"),
        n_seeds=("seed", "nunique")).reset_index()
    tr = agg[agg.training == "kta_trained"].sort_values("n_qubits")
    fx = agg[agg.training == "fixed"].sort_values("n_qubits")
    ratio = {int(r.n_qubits): round(float(r.sigma_K2_mean) / float(
        fx[fx.n_qubits == r.n_qubits].sigma_K2_mean.iloc[0]), 1) for _, r in tr.iterrows()}
    summ = {"seeds": SEEDS, "n_grid": sorted(df.n_qubits.unique().tolist()), "n_eval": SCALING_N_EVAL,
            "epochs": SCALING_EPOCHS,
            "trained_sigmaK2_above_haar_all_n": bool((tr.sigma_K2_mean.values > tr.haar_var_floor.values).all()),
            "trained_sigmaK2_by_n": {int(r.n_qubits): round(float(r.sigma_K2_mean), 4) for _, r in tr.iterrows()},
            "trained_sigmaK2_std_by_n": {int(r.n_qubits): round(float(r.sigma_K2_std), 4) for _, r in tr.iterrows()},
            "fixed_sigmaK2_by_n": {int(r.n_qubits): round(float(r.sigma_K2_mean), 5) for _, r in fx.iterrows()},
            "trained_over_fixed_sigmaK2_ratio_by_n": {str(k): v for k, v in ratio.items()},
            "fixed_over_haar_floor_ratio_by_n": {int(r.n_qubits): round(float(r.sigma_K2_mean) / float(r.haar_var_floor), 1) for _, r in fx.iterrows()},
            "trained_roc_by_n": {int(r.n_qubits): round(float(r.roc_mean), 3) for _, r in tr.iterrows()},
            "trained_d_eff_by_n": {int(r.n_qubits): round(float(r.d_eff_mean), 1) for _, r in tr.iterrows()}}
    return agg, summ


def study_scaling_merge():
    parts = sorted(OUTDIR.glob("scaling_raw_n*.csv"))
    if not parts:
        raise SystemExit("no scaling_raw_n*.csv found; run --only-n first")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True).sort_values(["n_qubits", "training", "seed"])
    agg, summ = _scaling_aggregate(df)
    save_table(df, OUTDIR, "scaling_raw.csv");
    save_table(agg, OUTDIR, "scaling.csv")
    (OUTDIR / "scaling.json").write_text(json.dumps(summ, indent=2, default=float))
    print("[scaling merge] summary:", json.dumps(summ, default=float)[:500])
    print(f"[scaling merge] {len(df)} raw rows across n={summ['n_grid']} -> cache")
    return df, summ



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only-n", type=int, default=None, help="run one qubit count and save scaling_raw_n<N>.csv")
    ap.add_argument("--merge", action="store_true", help="aggregate scaling_raw_n*.csv -> scaling.{csv,json}")
    a = ap.parse_args()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time(); started = now_iso()
    if a.only_n is not None:
        study_scaling_only_n(a.only_n); return
    if a.merge:
        df, summ = study_scaling_merge()
    else:
        data, idx = _load()
        from qkv2x.config import load_yaml
        qc = load_yaml(REPO / "configs" / "quantum.yaml")
        br = load_yaml(REPO / "configs" / "features_veremi_nextgen.yaml")["reduction"]
        raw = RawWriter(OUTDIR / "scaling_raw.csv")
        allrows = []
        for n in SCALING_N_GRID:
            allrows += _scaling_one_n(n, qc, br, data, idx, raw)
        df = pd.DataFrame(allrows); agg, summ = _scaling_aggregate(df)
        save_table(agg, OUTDIR, "scaling.csv")
        (OUTDIR / "scaling.json").write_text(json.dumps(summ, indent=2, default=float))
    write_metadata(OUTDIR, study="qubit_scaling", started_iso=started, t0=t0,
                   results_summary=summ, notes="Multi-seed qubit-count scaling; raw rows saved; no QPU.")


if __name__ == "__main__":
    main()
