"""postprocess_scaling.py -- turn the n=14..20 GPU Grams (results/gpu_scaling_grams/) into scaling_raw_n*.csv rows
with exactly the metric semantics of scripts/qubit_scaling.py, then merge into the scaling series.

Metrics per (n, training, seed), mirroring qubit_scaling._eval_metrics on the 800x800 Gram:
  sigma_K2 = upper-triangle off-diagonal variance of the FULL Gram
  d_eff    = (tr K)^2 / tr(K^2)
  kta_eval = centred KTA of the test block K[n_tr:, n_tr:] vs y_te  (schema compatibility)
  roc/pr   = SVC(kernel='precomputed', C=1) trained on K[:n_tr,:n_tr], scored on test rows
  haar_var_floor = (d-1)/(d^2 (d+1)),  shots_for_r0.95 as in _scaling_one_n

Usage:
  python scripts/gpu_engine/postprocess_scaling.py             # process local grams + merge
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GRAMS = REPO / "results" / "gpu_scaling_grams"   # copy the gram_n*_*.npz files from the GPU host here
OUTDIR = REPO / "results" / "qubit_scaling"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))


SPLITS = {14: (7, 7), 16: (8, 8), 18: (9, 9), 20: (9, 11)}
SEEDS = [11, 23, 41, 57, 71]
EPOCHS, N_EVAL = 200, 400


def _eff_dim(K):
    return float(np.trace(K) ** 2 / max(np.trace(K @ K), 1e-300))


def process() -> None:
    from qkv2x.quantum.diagnostics import centered_kta
    from sklearn.svm import SVC
    from sklearn.metrics import roc_auc_score, average_precision_score

    rows = []
    for n in SPLITS:
        for training_file, training_row in [("kta", "kta_trained"), ("fixed", "fixed")]:
            for sd in SEEDS:
                f = GRAMS / f"gram_n{n}_{training_file}_s{sd}.npz"
                if not f.exists():
                    print(f"  MISSING {f.name}"); continue
                z = np.load(f)
                K = z["K"]; n_tr = int(z["n_tr"]); y_both = z["y_both"].astype(int)
                y_tr, y_te = y_both[:n_tr], y_both[n_tr:]
                iu = np.triu_indices_from(K, k=1)
                s2 = float(K[iu].var())
                de = _eff_dim(K)
                kta = float(centered_kta(K[n_tr:, n_tr:], np.where(y_te > 0, 1.0, -1.0)))
                svc = SVC(kernel="precomputed", C=1.0).fit(K[:n_tr, :n_tr], y_tr)
                sc = svc.decision_function(K[n_tr:, :n_tr])
                d = 2.0 ** n
                haar = (d - 1) / (d ** 2 * (d + 1))
                shots95 = float(0.25 / max((np.sqrt(s2) * np.sqrt(1 / 0.95 ** 2 - 1)) ** 2, 1e-15))
                rows.append({
                    "n_qubits": n, "training": training_row, "seed": sd,
                    "sigma_K2": s2, "d_eff": de, "kta_eval": kta,
                    "roc_auc": float(roc_auc_score(y_te, sc)),
                    "pr_auc": float(average_precision_score(y_te, sc)),
                    "haar_var_floor": haar, "shots_for_r0.95": shots95,
                    "epochs": EPOCHS, "n_eval": N_EVAL,
                    "engine": "torch-gpu-c64-vjp", "register_split": f"{SPLITS[n][0]}+{SPLITS[n][1]}",
                    "best_alignment": float(z["best_alignment"]),
                    "wall_train_s": float(z["wall_train_s"]),
                })
    df = pd.DataFrame(rows).sort_values(["n_qubits", "training", "seed"])
    for n in sorted(set(df.n_qubits)):
        sub = df[df.n_qubits == n]
        sub.to_csv(OUTDIR / f"scaling_raw_n{n}.csv", index=False)
        print(f"wrote scaling_raw_n{n}.csv ({len(sub)} rows)")

    # merged aggregate over the FULL range (existing n4..12 files + new)
    import qubit_scaling as Q
    Q.study_scaling_merge()

    # quick headline print
    ag = df.groupby(["n_qubits", "training"]).agg(
        s2=("sigma_K2", "mean"), s2sd=("sigma_K2", "std"),
        roc=("roc_auc", "mean"), de=("d_eff", "mean"),
        haar=("haar_var_floor", "first")).reset_index()
    print("\n=== NEW POINTS (5-seed means) ===")
    for _, r in ag.iterrows():
        rat = r.s2 / r.haar
        print(f"  n={int(r.n_qubits):2d} {r.training:11s}: sigma_K2={r.s2:.4f}+-{r.s2sd:.4f}  "
              f"d_eff={r.de:7.1f}  ROC={r.roc:.3f}  sigma/haar={rat:.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    a = ap.parse_args()
    process()
