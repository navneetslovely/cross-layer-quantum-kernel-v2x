"""gram_inputs_prep.py -- input preparation for the GPU Gram engine (scripts/gpu_engine/gram_gpu.py).

Builds self-contained npz inputs for (a) entanglement_ablation, the entanglement-topology ablation
(qsvm_none / qsvm_sparse / anchors qsvm_crosslayer + qsvm_zz) and (b) label_rule, the
window-label-rule sensitivity (majority / fully_active / drop-onset), all on the
PUBLISHED dev_grid seed-7 protocol (dev_multiseed): group_split(0.30, 0.15, 7)
-> group-wise train cap 3000 (=> 3,023 windows) -> group-wise test cap 6000, blockwise
PCA 4+4 fit on train (cached results/dev_grid_multiseed/checkpoints/reduced_s7.npz).

The remote engine (scripts/gpu_engine/gram_gpu.py) needs ZERO qkv2x imports and ZERO
RNG of its own: couples lists and the full per-epoch KTA batch-index streams
(default_rng(seed=11) -> _balanced_batch, exactly train_feature_map's stream) are
precomputed here with the real qkv2x code.

label_rule design (documented for the record): 'majority' and 'fully_active' RELABEL the
identical window set under the identical split (isolating the labelling effect);
'drop_onset' removes windows with 0 < attack_frac < 1 (after which all label rules
coincide) and refits the reducer on the retained train windows. KTA batches depend
on labels, so each rule gets its own precomputed batch stream.

Run: python scripts/studies/gram_inputs_prep.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from qkv2x.config import load_yaml                                   # noqa: E402
from qkv2x.experiments.common import (_cap_train_groupwise,          # noqa: E402
                                      group_split, load_experiment_data)
from qkv2x.features.crosslayer import FEATURE_SPEC                   # noqa: E402
from qkv2x.features.reduce import BlockReducer                       # noqa: E402
from qkv2x.quantum.featuremap import CrossLayerFeatureMap            # noqa: E402
from qkv2x.quantum.train import _balanced_batch                      # noqa: E402

OUT = REPO / "results" / "studies" / "gram_inputs"
CKPT = REPO / "results" / "dev_grid_multiseed" / "checkpoints"

EPOCHS, BATCH, TRAIN_SEED = 200, 512, 11        # configs/quantum.yaml::train
LR, INIT_W, INIT_U, INIT_G = 0.05, 1.0, 1.0, 0.5
SPLIT_SEED, CAP_TRAIN, CAP_TEST = 7, 3000, 6000  # dev_multiseed protocol


def _load_parquet():
    df = pd.read_parquet(REPO / "runs" / "dev_grid" / "features.parquet")
    names = [s[0] for s in FEATURE_SPEC]
    blocks = [s[3] for s in FEATURE_SPEC]
    return df, names, blocks


def _batches(y_tr: np.ndarray) -> np.ndarray:
    """[EPOCHS, BATCH] int32 -- exactly train_feature_map's stream."""
    rng = np.random.default_rng(TRAIN_SEED)
    out = np.zeros((EPOCHS, BATCH), np.int32)
    for ep in range(EPOCHS):
        out[ep] = _balanced_batch(y_tr, BATCH, rng)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df, names, blocks = _load_parquet()
    X = df[names].to_numpy()
    y = df["y"].to_numpy().astype(int)
    groups = df["group"].to_numpy()
    aoi = df["aoi"].to_numpy().astype(float)

    fcfg = load_yaml(str(REPO / "configs" / "features.yaml"))
    s = fcfg["splits"]
    tr, va, te = group_split(groups, y, s["test_frac"], s["val_frac"], SPLIT_SEED)
    tr = _cap_train_groupwise(tr, groups, CAP_TRAIN, SPLIT_SEED)
    te_cap = _cap_train_groupwise(te, groups, CAP_TEST, SPLIT_SEED + 1)
    assert len(tr) == 3023, f"train cap mismatch: {len(tr)} != 3023 (published)"
    assert len(te_cap) == 6000, f"test cap mismatch: {len(te_cap)} != 6000 (published)"

    Xr = np.load(CKPT / "reduced_s7.npz")["Xr"]
    assert Xr.shape == (len(df), 8), Xr.shape

    # sanity anchor: the cached published Gram must match these indices' count
    Kz = np.load(CKPT / "gram_qsvm_crosslayer_s7.npz")
    assert Kz["K_tr"].shape == (len(tr), len(tr))
    assert Kz["K_te"].shape == (len(te_cap), len(tr))

    # ---- entanglement_ablation: couples per topology variant (uses the real featuremap code) ----
    couples = {}
    for tag, ent in [("none", "none"), ("sparse", "sparse_random"),
                     ("cross", "cross_layer"), ("zz", "zz_full")]:
        fm = CrossLayerFeatureMap(4, 4, entangler=ent)
        couples[tag] = np.array(fm._couples, dtype=np.int64).reshape(-1, 2)
    assert (len(couples["none"]), len(couples["sparse"]),
            len(couples["cross"]), len(couples["zz"])) == (0, 16, 16, 28)

    np.savez_compressed(
        OUT / "entanglement_ablation_inputs.npz",
        Xr=Xr.astype(np.float64), y=y, aoi=aoi, groups=groups,
        idx_tr=tr, idx_va=va, idx_te=te_cap,
        batch_idx=_batches(y[tr]),
        couples_none=couples["none"], couples_sparse=couples["sparse"],
        couples_cross=couples["cross"], couples_zz=couples["zz"],
        epochs=EPOCHS, batch=BATCH, train_seed=TRAIN_SEED, lr=LR,
        init_w=INIT_W, init_u=INIT_U, init_gamma=INIT_G,
        kappa=0.5, n_layers=2, n_mobility=4, n_network=4, chunk=256,
    )
    print(f"entanglement_ablation_inputs.npz: tr={len(tr)} va={len(va)} te={len(te_cap)} "
          f"couples none/sparse/cross/zz = 0/16/16/28")

    # ---- label_rule: rebuild windows to obtain attack_frac, then relabel ----
    data = load_experiment_data(REPO / "runs" / "dev_grid", fcfg)
    assert np.allclose(data.X, X), "rebuilt X != features.parquet (order/content)"
    assert (data.y == y).all(), "rebuilt y != features.parquet"
    frac = data.meta["attack_frac"].to_numpy().astype(float)
    assert ((frac > 0).astype(int) == y).all(), "attack_frac inconsistent with 'any' labels"

    y_maj = (frac >= 0.5).astype(int)
    y_full = (frac == 1.0).astype(int)
    keep = (frac == 0.0) | (frac == 1.0)          # drop-onset retention mask

    # class presence per split per rule (protocol guard)
    stats = {}
    for rule, yr in [("any", y), ("majority", y_maj), ("fully_active", y_full)]:
        stats[rule] = {p: {"n": int(len(ix)), "pos": int(yr[ix].sum())}
                       for p, ix in [("tr", tr), ("va", va), ("te", te_cap)]}
        for p, ix in [("tr", tr), ("va", va), ("te", te_cap)]:
            assert 0 < yr[ix].sum() < len(ix), (rule, p, "single-class split")

    # onset variant: subset indices (split membership preserved), reducer REFIT on
    # the retained train windows (protocol-honest), labels = frac==1 on retained.
    tr_o, va_o, te_o = (ix[keep[ix]] for ix in (tr, va, te_cap))
    red_o = BlockReducer(fcfg["reduction"]).fit(X[tr_o], blocks)
    Xr_o = red_o.transform(X)
    y_o = y_full                                   # == any == majority on retained
    stats["drop_onset"] = {p: {"n": int(len(ix)), "pos": int(y_o[ix].sum())}
                           for p, ix in [("tr", tr_o), ("va", va_o), ("te", te_o)]}
    for p, ix in [("tr", tr_o), ("va", va_o), ("te", te_o)]:
        assert 0 < y_o[ix].sum() < len(ix), ("drop_onset", p, "single-class split")

    np.savez_compressed(
        OUT / "label_rule_inputs.npz",
        Xr=Xr.astype(np.float64), Xr_onset=Xr_o.astype(np.float64),
        y_any=y, y_majority=y_maj, y_fully_active=y_full, attack_frac=frac,
        aoi=aoi, groups=groups,
        idx_tr=tr, idx_va=va, idx_te=te_cap,
        idx_tr_onset=tr_o, idx_va_onset=va_o, idx_te_onset=te_o,
        batch_idx_majority=_batches(y_maj[tr]),
        batch_idx_fully_active=_batches(y_full[tr]),
        batch_idx_onset=_batches(y_o[tr_o]),
        couples_cross=couples["cross"],
        epochs=EPOCHS, batch=BATCH, train_seed=TRAIN_SEED, lr=LR,
        init_w=INIT_W, init_u=INIT_U, init_gamma=INIT_G,
        kappa=0.5, n_layers=2, n_mobility=4, n_network=4, chunk=256,
    )
    summary = {
        "protocol": {
            "split": {"test_frac": s["test_frac"], "val_frac": s["val_frac"],
                      "seed": SPLIT_SEED, "cap_train": CAP_TRAIN,
                      "cap_test": CAP_TEST, "n_tr": len(tr), "n_va": len(va),
                      "n_te": len(te_cap)},
            "kta": {"epochs": EPOCHS, "batch": BATCH, "seed": TRAIN_SEED,
                    "lr": LR, "init": [INIT_W, INIT_U, INIT_G]},
            "label_rule_design": "majority/fully_active relabel the identical window set "
                          "under the identical split; drop_onset removes "
                          "0<frac<1 windows and refits the reducer on retained train",
        },
        "label_rule_stats": stats,
        "n_mixed_windows": int((~keep).sum()),
        "mixed_fraction": float((~keep).mean()),
    }
    (OUT / "gram_inputs_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["label_rule_stats"], indent=1))
    print(f"mixed (onset-straddling) windows: {int((~keep).sum())} "
          f"({100 * float((~keep).mean()):.2f}%)  -> label_rule_inputs.npz")


if __name__ == "__main__":
    main()
