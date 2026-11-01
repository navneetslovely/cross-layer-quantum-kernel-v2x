"""paper hardware-feasibility run on the *dev_grid* (SUMO+ns-3 co-sim) dataset.

The NextGen hardware runs (measured kernels in results/veremi_nextgen/ibm_hardware/) use the
VeReMi-NextGen cross-layer kernel. This script runs the SAME compute-uncompute fidelity-kernel
feasibility experiment on paper's PRIMARY dataset — the dev_grid main co-simulation task —
using the multiseed-trained cross-layer checkpoint (`results/dev_grid_multiseed/checkpoints/
qtrain_qsvm_crosslayer_s{seed}.pt` + `reduced_s{seed}.npz`). It uses the circuit/sim
helpers in `hw_kernel.py`; only the input loader differs (dev_grid split +
AoI pre-modulation, replicating `dev_multiseed.py`).

Budget-safe & layout-aware: same free-tier guards as the NextGen runners, submission requires
--confirm, and --initial-layout pins a chosen 8-qubit patch. A self-check verifies the pre-modulated
simulated kernel reproduces the native featuremap kernel byte-for-byte before anything is run.

Usage:
  # local validation (no IBM account; exact + noisy Aer):
  python scripts/ibm_devgrid_kernel.py --backend aer
  python scripts/ibm_devgrid_kernel.py --backend aer-noisy
  # real hardware (pinned layout), after placing your IBM Quantum token (README, Section 5):
  python scripts/ibm_devgrid_kernel.py --backend ibm_marrakesh \
      --initial-layout "33,34,35,39,51,52,53,58" --confirm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
import hw_kernel as hk  # _u_gate/_pair_circuit/_sim_kernel  # noqa: E402

CK = REPO / "results" / "dev_grid_multiseed" / "checkpoints"
OUT = REPO / "results" / "dev_grid_multiseed" / "ibm_hardware"
KAPPA = 0.5      # configs/features.yaml::aoi.kappa
N_MOB = 4        # reduced register split: [0..3]=mobility, [4..7]=network (per_block_pca)


def _load_inputs_devgrid(n_tr: int, n_te: int, seed: int):
    """dev_grid stratified train/test block + trained params + AoI-PRE-modulated reduced coords,
    replicating scripts/dev_multiseed.py (group_split + reduced_s{seed}.npz + best_params)."""
    import pandas as pd
    import torch

    from qkv2x.config import load_yaml
    from qkv2x.experiments.common import group_split
    from qkv2x.features.crosslayer import FEATURE_SPEC

    df = pd.read_parquet(REPO / "runs" / "dev_grid" / "features.parquet")
    names = [s[0] for s in FEATURE_SPEC]
    y = df["y"].to_numpy()
    groups = df["group"].to_numpy()
    aoi = df["aoi"].to_numpy().astype(np.float64)

    s = load_yaml(str(REPO / "configs/features.yaml"))["splits"]
    tr_all, _va, te_all = group_split(groups, y, s["test_frac"], s["val_frac"], seed)

    Xr = np.load(CK / f"reduced_s{seed}.npz")["Xr"].astype(np.float64)   # reducer fit on tr (dev_multiseed.py)

    rng = np.random.default_rng(seed)

    def _pick(pool, n):
        pos = pool[y[pool] == 1]
        neg = pool[y[pool] == 0]
        k = n // 2
        return np.sort(np.concatenate([rng.choice(pos, k, replace=False),
                                       rng.choice(neg, n - k, replace=False)]))

    tr = _pick(tr_all, n_tr)
    te = _pick(te_all, n_te)

    st = torch.load(CK / f"qtrain_qsvm_crosslayer_s{seed}.pt", weights_only=False)
    params = {k: v.detach().cpu().numpy().astype(np.float64)
              for k, v in st["best_params"].items()}

    # AoI pre-modulation (featuremap.modulate with aoi_modulation=False downstream): x_net *= lam
    x_tr = Xr[tr].copy()
    x_te = Xr[te].copy()
    x_tr[:, N_MOB:] *= np.exp(-aoi[tr] / KAPPA)[:, None]
    x_te[:, N_MOB:] *= np.exp(-aoi[te] / KAPPA)[:, None]
    return x_tr, y[tr], x_te, y[te], params, tr, te, Xr, aoi


def _selfcheck(x_te, x_tr, params, Xr, aoi, te, tr):
    """Confirm the pre-modulated _sim_kernel == native featuremap kernel (aoi_modulation ON)."""
    import torch

    from qkv2x.quantum.featuremap import CrossLayerFeatureMap
    from qkv2x.quantum.kernels import fidelity_gram
    fm = CrossLayerFeatureMap(4, 4, n_layers=params["w"].shape[0], aoi_modulation=True, kappa=KAPPA)
    p = {k: torch.as_tensor(v, dtype=fm.rdtype) for k, v in params.items()}
    K_native = np.asarray(fidelity_gram(fm.statevectors(Xr[te], aoi[te], p),
                                        fm.statevectors(Xr[tr], aoi[tr], p)))
    K_premod = np.asarray(hk._sim_kernel(x_te, x_tr, params))
    return float(np.max(np.abs(K_native - K_premod)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="aer",
                    help="aer | aer-noisy | ibm_marrakesh | ibm_kingston | ibm_fez")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--initial-layout", default=None,
                    help="comma-separated 8 physical qubits to pin (coherence-aware)")
    ap.add_argument("--est-cost", type=int, default=55)
    ap.add_argument("--margin", type=int, default=25)
    ap.add_argument("--max-circuits", type=int, default=150)
    ap.add_argument("--confirm", action="store_true")
    ap.add_argument("--no-wait", action="store_true")
    ap.add_argument("--fetch", default=None, metavar="JOB_ID")
    ap.add_argument("--include-train-gram", action="store_true",
                    help="also measure the n_train x n_train training Gram (upper triangle) so the whole "
                         "scoring chain (train Gram + test block) is hardware-measured")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    init_layout = None
    if args.initial_layout:
        init_layout = [int(q) for q in args.initial_layout.split(",")]
        if len(init_layout) != 8:
            sys.exit(f"[ABORT] --initial-layout needs 8 qubits, got {len(init_layout)}")

    x_tr, y_tr, x_te, y_te, params, tr_idx, te_idx, Xr, aoi = _load_inputs_devgrid(
        args.n_train, args.n_test, args.seed)

    # ---- correctness self-check (free, must pass before any execution) -------
    dmax = _selfcheck(x_te, x_tr, params, Xr, aoi, te_idx, tr_idx)
    print(f"[selfcheck] max|K_native - K_premod| = {dmax:.2e}  "
          f"({'OK' if dmax < 1e-5 else 'FAIL — loader mismatch'})")
    if dmax >= 1e-5:
        sys.exit("[ABORT] dev_grid loader does not reproduce the native featuremap kernel.")

    pairs = [(i, j) for i in range(args.n_test) for j in range(args.n_train)]
    tr_pairs = ([(i, j) for i in range(args.n_train) for j in range(i + 1, args.n_train)]
                if args.include_train_gram else [])
    n_circ = len(pairs) + len(tr_pairs)
    circs = ([hk._pair_circuit(x_te[i], x_tr[j], params) for i, j in pairs]
             + [hk._pair_circuit(x_tr[i], x_tr[j], params) for i, j in tr_pairs])
    print(f"[plan] dataset=dev_grid seed={args.seed} backend={args.backend} "
          f"{n_circ} circ x {args.shots} shots (test {args.n_test} x train {args.n_train}"
          f"{' + train Gram ' + str(len(tr_pairs)) if tr_pairs else ''})")
    if n_circ > args.max_circuits:
        sys.exit(f"[ABORT] {n_circ} circuits > --max-circuits {args.max_circuits}")

    K_tr_sim = hk._sim_kernel(x_tr, x_tr, params)
    K_te_sim = hk._sim_kernel(x_te, x_tr, params)

    # ---- execute ------------------------------------------------------------
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime import SamplerV2
    t0 = time.time()
    is_ibm = args.backend.startswith("ibm")
    if is_ibm:
        import glob

        # credential resolution lives in qkv2x.quantum.ibm_account:
        # env QISKIT_IBM_TOKEN -> saved qiskit account (your own credentials).
        from qkv2x.quantum.ibm_account import get_service
        service, _acct = get_service()
        if args.fetch:
            job = service.job(args.fetch)
            if "DONE" not in str(job.status()).upper():
                sys.exit(f"[ibm] job {args.fetch}: {job.status()} — not done.")
            result = job.result(); usage = _usage(job); backend_name = job.backend().name
        else:
            rem = service.usage().get("usage_remaining_seconds")
            need = args.est_cost + args.margin
            print(f"[budget] remaining {rem}s; est ~{args.est_cost}s (+{args.margin}s)")
            if rem is None or rem < need:
                sys.exit(f"[ABORT] budget {rem}s < needed ~{need}s.")
            backend = service.backend(args.backend)
            print(f"[ibm] {backend.name} queue={backend.status().pending_jobs}"
                  + (f"  PINNED layout={init_layout}" if init_layout else ""))
            if not args.confirm:
                sys.exit("[dry-run] re-run with --confirm to submit.")
            pm = generate_preset_pass_manager(backend=backend, optimization_level=3,
                                              seed_transpiler=args.seed,
                                              initial_layout=init_layout)
            job = SamplerV2(mode=backend).run(pm.run(circs), shots=args.shots)
            print(f"[ibm] job {job.job_id()} submitted")
            if args.no_wait:
                (OUT / f"pending_devgrid_{args.backend}_{args.seed}.json").write_text(json.dumps(
                    {"job_id": job.job_id(), "backend": args.backend, "seed": args.seed,
                     "n_train": args.n_train, "n_test": args.n_test,
                     "initial_layout": init_layout, "include_train_gram": bool(tr_pairs),
                     "train_idx": tr_idx.tolist(), "test_idx": te_idx.tolist()}, indent=2))
                print("[ibm] --no-wait: saved pending stub; fetch later with --fetch."); return
            result = job.result(); usage = _usage(job); backend_name = backend.name
    elif args.backend == "statevector":
        # dependency-free EXACT validation: P(0..0) of compute-uncompute = fidelity kernel.
        from types import SimpleNamespace

        from qiskit.quantum_info import Statevector
        z8 = "0" * 8
        result = []
        for c in circs:
            p0 = float(np.abs(Statevector(c.remove_final_measurements(inplace=False)).data[0]) ** 2)
            counts = {z8: int(round(p0 * args.shots))}
            result.append(SimpleNamespace(data=SimpleNamespace(
                c=SimpleNamespace(get_counts=lambda _c=counts: _c))))
        usage = None; backend_name = "statevector(exact)"
    else:
        from qiskit_aer import AerSimulator
        if args.backend == "aer-noisy":
            from qiskit_aer.noise import NoiseModel, depolarizing_error
            nm = NoiseModel()
            nm.add_all_qubit_quantum_error(depolarizing_error(0.002, 1), ["ry", "rz", "rx", "sx", "x"])
            nm.add_all_qubit_quantum_error(depolarizing_error(0.01, 2), ["rzz", "cx", "cz", "ecr"])
            backend = AerSimulator(noise_model=nm)
        else:
            backend = AerSimulator()
        pm = generate_preset_pass_manager(backend=backend, optimization_level=1, seed_transpiler=args.seed)
        result = SamplerV2(mode=backend).run(pm.run(circs), shots=args.shots).result()
        usage = None; backend_name = f"aer({args.backend})"
    wall = time.time() - t0

    # ---- hardware kernel + agreement + paper detection -------------------
    zeros = "0" * 8

    def _p0(res):
        counts = res.data.c.get_counts() if hasattr(res.data, "c") else \
            next(iter(res.data.values())).get_counts()
        return counts.get(zeros, 0) / args.shots

    K_hw = np.zeros((args.n_test, args.n_train))
    for (i, j), res in zip(pairs, result[:len(pairs)]):
        K_hw[i, j] = _p0(res)
    K_tr_hw = None
    if tr_pairs:
        K_tr_hw = np.eye(args.n_train)
        for (i, j), res in zip(tr_pairs, result[len(pairs):]):
            K_tr_hw[i, j] = K_tr_hw[j, i] = _p0(res)

    from scipy.stats import pearsonr
    from sklearn.metrics import average_precision_score, roc_auc_score

    from qkv2x.models.qsvm import QuantumKernelSVM
    hw, sim = K_hw.ravel(), K_te_sim.ravel()
    r = float(pearsonr(hw, sim)[0]); a, b = np.polyfit(sim, hw, 1)
    svm = QuantumKernelSVM(C_grid=[0.1, 1.0, 10.0, 100.0], cv_folds=3)
    svm.fit(K_tr_sim, y_tr)
    det = {"roc_auc_hw": float(roc_auc_score(y_te, svm.decision_function(K_hw))),
           "roc_auc_sim": float(roc_auc_score(y_te, svm.decision_function(K_te_sim))),
           "pr_auc_hw": float(average_precision_score(y_te, svm.decision_function(K_hw))),
           "pr_auc_sim": float(average_precision_score(y_te, svm.decision_function(K_te_sim))),
           "y_test": y_te.tolist()}
    if K_tr_hw is not None:
        from qkv2x.quantum.kernels import psd_clip
        svm_hw = QuantumKernelSVM(C_grid=[0.1, 1.0, 10.0, 100.0], cv_folds=3)
        svm_hw.fit(psd_clip(K_tr_hw), y_tr)
        trh, trs = K_tr_hw[np.triu_indices(args.n_train, 1)], K_tr_sim[np.triu_indices(args.n_train, 1)]
        det["hw_chain"] = {"roc_auc_hw_train_hw_test": float(roc_auc_score(y_te, svm_hw.decision_function(K_hw))),
                           "pr_auc_hw_train_hw_test": float(average_precision_score(y_te, svm_hw.decision_function(K_hw))),
                           "C_hw": svm_hw.best_C_, "C_sim": svm.best_C_,
                           "train_gram_pearson_r": float(pearsonr(trh, trs)[0]),
                           "train_gram_slope": float(np.polyfit(trs, trh, 1)[0])}
    out = {
        "dataset": "dev_grid", "backend": backend_name, "seed": args.seed, "shots": args.shots,
        "n_circuits": n_circ, "n_train": args.n_train, "n_test": args.n_test,
        "initial_layout": init_layout, "selfcheck_max_abs_diff": dmax,
        "wall_time_s": round(wall, 1), "qpu_usage_s": str(usage) if usage is not None else None,
        "kernel_agreement": {"pearson_r": r, "contraction_slope_a": float(a), "intercept_b": float(b),
                             "sim_mean": float(sim.mean()), "hw_mean": float(hw.mean())},
        "detection": det,
        "K_hw": K_hw.tolist(), "K_te_sim": K_te_sim.tolist(),
        "K_tr_hw": (K_tr_hw.tolist() if K_tr_hw is not None else None), "K_tr_sim": K_tr_sim.tolist(),
        "n_test_pairs": len(pairs), "n_train_gram_pairs": len(tr_pairs),
        "account": (_acct if is_ibm else None), "y_train": y_tr.tolist(),
        "train_idx": tr_idx.tolist(), "test_idx": te_idx.tolist(),
    }
    tag = backend_name.replace("(", "_").replace(")", "")
    path = OUT / f"devgrid_hw_{tag}_s{args.seed}_n{args.n_test}{'_trgram' if tr_pairs else ''}_{int(time.time())}.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps({"kernel_agreement": out["kernel_agreement"], "detection": det}, indent=2))
    print(f"[done] {backend_name} r={r:.3f} a={a:.3f} ROC hw={det['roc_auc_hw']:.3f} "
          f"sim={det['roc_auc_sim']:.3f}  saved -> {path}")


def _usage(job):
    try:
        return job.usage()
    except Exception:
        return None


if __name__ == "__main__":
    main()
