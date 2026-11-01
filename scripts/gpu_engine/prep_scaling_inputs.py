"""prep_scaling_inputs.py -- LOCAL prep for the n=14..20 GPU scaling extension.

Produces ONE self-contained npz per n with everything the remote GPU engine needs, so the
box needs zero qkv2x imports and zero RNG calls of its own:

  - Xr        : reduced features [5999, n] from the REAL fit_reducer (per-block PCA,
                train-fit only) -- identical code path to scripts/qubit_scaling.py
  - aoi, y
  - tr_e/te_e : per-seed balanced 400-row eval subsets (SAME rng stream as _scaling_one_n:
                default_rng(seed) -> _balanced(tr) -> _balanced(te))
  - batch_idx : per-seed [epochs, batch] KTA batch indices INTO the train pool (SAME stream
                as train_feature_map: default_rng(seed) -> _balanced_batch per epoch)

Register splits (mobility has only 9 raw features, network 14 -- DISCLOSED):
  n=14 -> 7+7, n=16 -> 8+8, n=18 -> 9+9 (mobility PCA at full rank), n=20 -> 9+11 (asymmetric).

Run:  python scripts/gpu_engine/prep_scaling_inputs.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
OUT = REPO / "results" / "gpu_engine_scaling_inputs"
OUT.mkdir(parents=True, exist_ok=True)

SEEDS = [11, 23, 41, 57, 71]          # = qubit_scaling.SEEDS
N_EVAL = 400                          # = SCALING_N_EVAL
EPOCHS = 200                          # = SCALING_EPOCHS
BATCH = 256                           # = _scaling_one_n train cfg
SPLITS = {14: (7, 7), 16: (8, 8), 18: (9, 9), 20: (9, 11)}


def _balanced(y, pool, n, rng):
    pool = np.asarray(pool)
    pos, neg = pool[y[pool] == 1], pool[y[pool] == 0]
    k = min(n // 2, len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    rng.shuffle(sel)
    return sel


def _balanced_batch(y, batch, rng):
    pos = np.where(y > 0)[0]
    neg = np.where(y <= 0)[0]
    half = max(1, batch // 2)
    pi = rng.choice(pos, size=half, replace=len(pos) < half)
    ni = rng.choice(neg, size=half, replace=len(neg) < half)
    idx = np.concatenate([pi, ni])
    rng.shuffle(idx)
    return idx


def main() -> None:
    import qubit_scaling as Q
    from qkv2x.config import load_yaml

    data, idx = Q._load()
    base_red = load_yaml(REPO / "configs" / "features_veremi_nextgen.yaml")["reduction"]
    y = data.y.astype(int)
    aoi = np.asarray(data.aoi, float)
    ytr = y[idx["tr"]]

    for n, (nm, nn) in SPLITS.items():
        red = copy.deepcopy(base_red)
        red["n_mobility"], red["n_network"] = nm, nn
        Xr = Q._reduce(data, red, idx["tr"])          # the REAL fit_reducer, train-fit only
        assert Xr.shape[1] == n, (n, Xr.shape)

        tr_e = np.zeros((len(SEEDS), N_EVAL), int)
        te_e = np.zeros((len(SEEDS), N_EVAL), int)
        batch_idx = np.zeros((len(SEEDS), EPOCHS, BATCH), np.int32)
        for si, sd in enumerate(SEEDS):
            rng = np.random.default_rng(sd)           # _scaling_one_n eval-subset stream
            tr_e[si] = _balanced(y, idx["tr"], N_EVAL, rng)
            te_e[si] = _balanced(y, idx["te"], N_EVAL, rng)
            rng2 = np.random.default_rng(sd)          # train_feature_map batch stream
            for ep in range(EPOCHS):
                batch_idx[si, ep] = _balanced_batch(ytr, BATCH, rng2)

        np.savez_compressed(
            OUT / f"scaling_inputs_n{n}.npz",
            Xr=Xr.astype(np.float64), y=y, aoi=aoi,
            idx_tr=idx["tr"], idx_te=idx["te"],
            tr_e=tr_e, te_e=te_e, batch_idx=batch_idx,
            seeds=np.array(SEEDS), n_mobility=nm, n_network=nn,
            epochs=EPOCHS, batch=BATCH, kappa=0.5, n_layers=2,
        )
        print(f"n={n} ({nm}+{nn}): Xr {Xr.shape} -> scaling_inputs_n{n}.npz")

    print("done ->", OUT)


if __name__ == "__main__":
    main()
