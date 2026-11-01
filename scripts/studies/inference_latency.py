#!/usr/bin/env python3
"""inference_latency — per-window inference latency of the cross-layer QSVM.

Uses the published seed-7 detector of results/dev_grid_multiseed (trained map parameters
from qtrain_qsvm_crosslayer_s7.pt; SVM refit on the stored training Gram with the same C
grid) and measures, on this machine's CPU:
  * statevector preparation for one window (and amortised per window in batches),
  * the kernel row against the support vectors only (n_sv) and against all training
    windows (n_train),
  * the SVM decision,
and reports ms per window. Hardware cost is derived from the measured 0.30 s/circuit.

Output: results/studies/inference_latency.json
"""
from __future__ import annotations

import json
import platform
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
                                      group_split, make_feature_map)  # noqa: E402
from qkv2x.features.crosslayer import FEATURE_SPEC                   # noqa: E402
from qkv2x.models.qsvm import QuantumKernelSVM                       # noqa: E402
from qkv2x.quantum.kernels import fidelity_gram                      # noqa: E402

OUT = REPO / "results" / "studies"
CKPT_DIR = REPO / "results" / "dev_grid_multiseed" / "checkpoints"
SEED = 7
QPU_S_PER_CIRCUIT = 0.30


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


def _timeit(fn, reps):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); ts.append(time.perf_counter() - t0)
    return {"median_ms": float(np.median(ts) * 1e3), "mean_ms": float(np.mean(ts) * 1e3),
            "min_ms": float(np.min(ts) * 1e3), "reps": reps}


def main():
    data = _load()
    cfgs = {"features": load_yaml(str(REPO / "configs/features.yaml")),
            "quantum": load_yaml(str(REPO / "configs/quantum.yaml"))}
    s = cfgs["features"]["splits"]
    tr, va, te = group_split(data.groups, data.y, s["test_frac"], s["val_frac"], SEED)
    tr = _cap_train_groupwise(tr, data.groups, cfgs["quantum"].get("max_train", 3000), SEED)
    te = _cap_train_groupwise(te, data.groups, 6000, SEED + 1)
    Xr = np.load(CKPT_DIR / f"reduced_s{SEED}.npz")["Xr"]
    st = torch.load(CKPT_DIR / "qtrain_qsvm_crosslayer_s7.pt", weights_only=False)
    params = {k: v.detach().clone() for k, v in st["best_params"].items()}
    z = np.load(CKPT_DIR / "gram_qsvm_crosslayer_s7.npz")
    K_tr = z["K_tr"]
    qc = cfgs["quantum"]
    svm = QuantumKernelSVM(C_grid=qc["svm"]["C_grid"], cv_folds=qc["svm"]["cv_folds"])
    svm.fit(K_tr, data.y[tr], groups=data.groups[tr])
    sv = svm.svc_.support_
    fm = make_feature_map("qsvm_crosslayer", qc, cfgs["features"]["reduction"], cfgs["features"])
    torch.set_num_threads(max(1, torch.get_num_threads()))
    with torch.no_grad():
        s_tr = fm.statevectors(Xr[tr], data.aoi[tr], params)
        s_sv = s_tr[sv]
        x1, a1 = Xr[te[:1]], data.aoi[te[:1]]
        xb, ab = Xr[te[:1000]], data.aoi[te[:1000]]
        # warm-up
        fm.statevectors(x1, a1, params); fm.statevectors(xb, ab, params)
        res = {
            "n_train": int(len(tr)), "n_sv": int(len(sv)), "n_qubits": fm.n, "C": svm.best_C_,
            "cpu": platform.processor() or platform.machine(), "machine": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "single_window": {
                "statevector": _timeit(lambda: fm.statevectors(x1, a1, params), 50),
                "kernel_row_vs_sv": _timeit(lambda: fidelity_gram(fm.statevectors(x1, a1, params), s_sv), 50),
                "kernel_row_vs_all_train": _timeit(lambda: fidelity_gram(fm.statevectors(x1, a1, params), s_tr), 50),
            },
            "batch_1000_windows": {
                "statevectors_total": _timeit(lambda: fm.statevectors(xb, ab, params), 5),
                "kernel_rows_vs_sv_total": _timeit(lambda: fidelity_gram(fm.statevectors(xb, ab, params), s_sv), 5),
            },
        }
        K1 = fidelity_gram(fm.statevectors(x1, a1, params), s_tr)
        res["single_window"]["svm_decision"] = _timeit(lambda: svm.decision_function(K1), 200)
        Kb = fidelity_gram(fm.statevectors(xb, ab, params), s_tr)
        res["batch_1000_windows"]["svm_decision_total"] = _timeit(lambda: svm.decision_function(Kb), 5)
    res["per_window_ms"] = {
        "cpu_single_sv": res["single_window"]["kernel_row_vs_sv"]["median_ms"] + res["single_window"]["svm_decision"]["median_ms"],
        "cpu_batched_sv": (res["batch_1000_windows"]["kernel_rows_vs_sv_total"]["median_ms"]
                           + res["batch_1000_windows"]["svm_decision_total"]["median_ms"]) / 1000.0,
    }
    res["hardware_derived"] = {"circuits_per_window_sv": int(len(sv)), "qpu_s_per_circuit": QPU_S_PER_CIRCUIT,
                               "qpu_s_per_window_sv": float(len(sv) * QPU_S_PER_CIRCUIT),
                               "qpu_s_per_window_all_train": float(len(tr) * QPU_S_PER_CIRCUIT)}
    (OUT / "inference_latency.json").write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps(res["per_window_ms"], indent=1)); print(json.dumps(res["hardware_derived"], indent=1))
    print(json.dumps({"status": "done", "n_sv": int(len(sv))}))


if __name__ == "__main__":
    main()
