"""gram_gpu.py -- GPU engine: detector-protocol Grams for the entanglement_ablation
entanglement-topology ablation and the label_rule label-rule sensitivity (dev_grid, seed 7).

A thin variant of scaling_gpu.py (imports its PROVEN circuit ops / KTA trainer
verbatim -- do not reimplement) that differs ONLY in:
  * couples come from the inputs npz (cross_layer / zz_full / none / sparse_random),
  * labels + KTA batch-index streams come from the inputs npz (precomputed locally
    with the real qkv2x code: default_rng(11) -> _balanced_batch, train_feature_map's
    exact stream),
  * outputs are the DETECTOR-protocol Gram blocks K_tr [n_tr, n_tr] and
    K_te [n_te, n_tr] (float32, c64 states -- matching the published
    qkv2x.quantum.kernels.fidelity_gram path), not the 800x800 evaluation Grams.

SVM fitting and every reported metric happen LOCALLY from the pulled Grams
(scripts/studies/entanglement_ablation.py / label_rule_sensitivity.py).

Variants:
  entanglement_ablation  --inputs entanglement_ablation_inputs.npz --variant {none,sparse,cross,zz}
       (zz = fixed Havlicek-style params w=u=1, gamma=1, NO training -- the
        published untrained-ZZ protocol; others are KTA-trained like cross-layer)
  label_rule  --inputs label_rule_inputs.npz --variant {majority,fully_active,onset}
       (cross-layer couples; per-rule labels/batches; onset uses the refit
        reducer coordinates Xr_onset + the retained-window split indices)

Usage (on a GPU machine, one job per GPU):
    python gram_gpu.py --inputs entanglement_ablation_inputs.npz --variant none   --gpu 0
    python gram_gpu.py --inputs label_rule_inputs.npz --variant onset  --gpu 1
    python gram_gpu.py --selftest        # runs scaling_gpu's gate selftest
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scaling_gpu import Engine, fidelity_gram, selftest, train_kta  # noqa: E402


class GramEngine(Engine):
    """scaling_gpu.Engine with the coupling topology injected instead of hardcoded."""

    def __init__(self, nm, nn, L, kappa, device, couples, cdtype=torch.complex64):
        super().__init__(nm, nn, L, kappa, device, cdtype)
        self.couples = [(int(i), int(j)) for i, j in couples]


def _variant_arrays(z, variant):
    """Return (Xr, y, aoi, idx_tr, idx_te, couples, batch_idx|None, trained: bool)."""
    if variant in ("none", "sparse", "cross", "zz"):            # entanglement_ablation file
        couples = z[f"couples_{variant}"]
        batch = None if variant == "zz" else z["batch_idx"]
        return (z["Xr"], z["y"], z["aoi"], z["idx_tr"], z["idx_te"],
                couples, batch, variant != "zz")
    # entanglement_ablation 2x2 completion -- fixed bipartite (untrained cross-layer)
    # and trained all-pairs ZZ, so topology and training are fully crossed on the
    # MAIN dataset (complementing the NextGen 2x2).
    if variant == "cross_fixed":
        return (z["Xr"], z["y"], z["aoi"], z["idx_tr"], z["idx_te"],
                z["couples_cross"], None, False)
    if variant == "zz_trained":
        return (z["Xr"], z["y"], z["aoi"], z["idx_tr"], z["idx_te"],
                z["couples_zz"], z["batch_idx"], True)
    if variant in ("majority", "fully_active"):                 # label_rule, relabel only
        return (z["Xr"], z[f"y_{variant}"], z["aoi"], z["idx_tr"], z["idx_te"],
                z["couples_cross"], z[f"batch_idx_{variant}"], True)
    if variant == "onset":                                      # label_rule, subset + refit
        return (z["Xr_onset"], z["y_fully_active"], z["aoi"],
                z["idx_tr_onset"], z["idx_te_onset"],
                z["couples_cross"], z["batch_idx_onset"], True)
    raise SystemExit(f"unknown variant {variant!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs")
    ap.add_argument("--variant")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mb", type=int, default=4096)
    ap.add_argument("--outdir", default="grams")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return

    z = np.load(a.inputs)
    Xr, y, aoi, itr, ite, couples, batch_idx, trained = _variant_arrays(z, a.variant)
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    nm, nn = int(z["n_mobility"]), int(z["n_network"])
    eng = GramEngine(nm, nn, int(z["n_layers"]), float(z["kappa"]), dev,
                   couples.reshape(-1, 2))
    lr = float(z["lr"])
    print(f"variant={a.variant} couples={len(eng.couples)} trained={trained} "
          f"tr={len(itr)} te={len(ite)} dev={dev}", flush=True)

    t0 = time.time()
    if trained:
        best, best_a, hist = train_kta(eng, Xr[itr], aoi[itr], y[itr],
                                       batch_idx, lr=lr, mb=a.mb)
    else:
        L, n = eng.L, eng.n
        nc = max(1, len(eng.couples))
        best = {"w": torch.ones(L, n, dtype=eng.rdtype, device=eng.device),
                "u": torch.ones(L, n, dtype=eng.rdtype, device=eng.device),
                "gamma": torch.ones(L, nc, dtype=eng.rdtype, device=eng.device)}
        best_a, hist = float("nan"), []
    t_train = time.time() - t0

    t0 = time.time()
    with torch.no_grad():
        S_tr = eng.states(Xr[itr], aoi[itr], best, mb=a.mb)
        S_te = eng.states(Xr[ite], aoi[ite], best, mb=a.mb)
        K_tr = fidelity_gram(S_tr).cpu().numpy().astype(np.float32)
        o = S_te @ S_tr.conj().transpose(0, 1)
        K_te = (o.real ** 2 + o.imag ** 2).cpu().numpy().astype(np.float32)
    t_eval = time.time() - t0

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"gram_{a.variant}.npz"
    np.savez_compressed(
        out, K_tr=K_tr, K_te=K_te,
        w=best["w"].cpu().numpy(), u=best["u"].cpu().numpy(),
        gamma=best["gamma"].cpu().numpy(),
        best_alignment=best_a, alignment_hist=np.array(hist, dtype=np.float64),
        couples=np.array(eng.couples, dtype=np.int64).reshape(-1, 2),
        variant=a.variant, trained=trained,
        t_train_s=t_train, t_eval_s=t_eval,
        torch_version=torch.__version__, device=dev,
    )
    meta = {"variant": a.variant, "n_couples": len(eng.couples),
            "trained": bool(trained), "best_alignment": best_a,
            "t_train_s": round(t_train, 1), "t_eval_s": round(t_eval, 1),
            "K_tr": list(K_tr.shape), "K_te": list(K_te.shape),
            "device": dev, "torch": torch.__version__}
    (outdir / f"gram_{a.variant}.json").write_text(json.dumps(meta))
    print("DONE", json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
