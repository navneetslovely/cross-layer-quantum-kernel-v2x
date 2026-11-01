"""Multi-seed the single-seed NextGen baselines.

The NextGen tables had 3 seeds for qsvm/rf/rbf but a single seed for the secondary
baselines, so frontier/operating-point/info-level tables couldn't carry error bars for
them. This trains the tabular single-seed baselines (knn_bagging, mlp, xgboost) on the
cached official-split assemblies (assembled{,_s17,_s27}.npz) at seeds {7,17,27} via the
same run_model path the pipeline uses, and aggregates mean+/-std with grouped-bootstrap CIs.

(lstm_seq is a sequence model that needs the per-window message sequences, which the cached
tabular assemblies do not store; multi-seeding it requires the heavier veremi_nextgen
sequence pipeline and is left as a follow-up -- noted in metadata.)

Outputs: results/nextgen_multiseed/{per_seed.csv, seed_summary.csv}.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _study_lib import REPO, RESULTS, now_iso, save_table, split_meta, write_metadata  # noqa: E402

MODELS = ["knn_bagging", "mlp", "xgboost"]
SEEDS = [7, 17, 27]


def _load_nextgen(seed: int):
    from qkv2x.experiments.common import ExperimentData

    cand = RESULTS / "veremi_nextgen" / "checkpoints" / ("assembled.npz" if seed == 7 else f"assembled_s{seed}.npz")
    z = np.load(cand, allow_pickle=True)
    data = ExperimentData(
        X=z["X"], y=z["y"].astype(int), attack_name=z["attack_name"].astype(str),
        groups=z["groups"].astype(int), aoi=z["aoi"], t_end=z["t_end"],
        feature_names=[str(s) for s in z["feature_names"]],
        blocks=[str(s) for s in z["blocks"]], points=pd.DataFrame(), meta=pd.DataFrame(),
    )
    split = z["split"].astype(str)
    idx = {"tr": np.where(split == "Train")[0], "va": np.where(split == "Validation")[0],
           "te": np.where(split == "Test")[0]}
    return data, idx


def main() -> None:
    t0 = time.time()
    started = now_iso()
    from qkv2x.config import load_yaml
    from qkv2x.experiments.common import fit_reducer, run_model
    from qkv2x.models.metrics import detection_report, grouped_bootstrap_ci

    feats = load_yaml(REPO / "configs" / "features_veremi_nextgen.yaml")
    cfgs = {"quantum": load_yaml(REPO / "configs" / "quantum.yaml"),
            "baselines": load_yaml(REPO / "configs" / "baselines.yaml"), "features": feats}

    rows = []
    last_splits = {}
    for seed in SEEDS:
        data, idx = _load_nextgen(seed)
        reducer = fit_reducer(feats["reduction"], data.X, data.blocks, idx["tr"])
        Xr = reducer.transform(data.X)
        last_splits = split_meta(data.y, idx)
        for model in MODELS:
            res = run_model(model, data, Xr, idx, cfgs)
            rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
            ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"], n_boot=200, seed=seed)
            rows.append({"model": model, "seed": seed, "pr_auc": rep["pr_auc"], "roc_auc": rep["roc_auc"],
                         "f1": rep["f1"], "mcc": rep["mcc"], "n_test": rep["n"], "n_pos": rep["n_pos"],
                         "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"]})
            print(f"[a4] {model:13s} seed={seed} PR-AUC={rep['pr_auc']:.3f} ROC={rep['roc_auc']:.3f}")

    df = pd.DataFrame(rows)
    out = RESULTS / "nextgen_multiseed"
    save_table(df, out, "per_seed.csv")
    summ = df.groupby("model").agg(
        n_seeds=("seed", "nunique"), pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
        roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
        f1_mean=("f1", "mean"), mcc_mean=("mcc", "mean")).reset_index()
    save_table(summ, out, "seed_summary.csv")

    write_metadata(
        out, study="A4_multiseed_baselines", started_iso=started, t0=t0,
        config_path="configs/features_veremi_nextgen.yaml",
        hyperparams={"models": MODELS, "seeds": SEEDS, "n_boot": 200},
        splits=last_splits,
        results_summary={"seed_summary": summ.round(4).to_dict("records"),
                         "lstm_note": "lstm_seq multi-seed needs the sequence pipeline (follow-up)"},
        notes="Multi-seed NextGen tabular baselines (knn/mlp/xgboost) on cached official-split assemblies.",
    )
    print("done -> results/nextgen_multiseed/seed_summary.csv")
    print(summ.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
