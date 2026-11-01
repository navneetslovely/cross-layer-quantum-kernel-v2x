#!/usr/bin/env python3
"""Feature-block ablation: cross-layer vs mobility-only vs network-only.

Question: how much of the detectability comes from the MOBILITY block (claimed
kinematics + their residuals) versus the NETWORK block (RSSI / timing / overlap /
off-road cross-layer observables)? We answer it for the QUANTUM detector AND the
strong classical baselines, on BOTH datasets, so the ablation is model- and
register-agnostic.

Mechanism (identical to the existing NextGen `netblind` ablation,
`experiments/veremi_nextgen_exp.py::_stage_reduce`): for each variant we ZERO the
ablated raw-feature block *before* fitting the BlockReducer, so that register
carries no information through the per-block PCA, then run the unchanged model
zoo via `experiments.common.run_model`.

  variant  zeroed block  keeps
  ----------------   --------------   ---------------------------------
  crosslayer (full)  none  mobility + network  (the headline)
  mobility_only  network  claimed-kinematics register only
  network_only  mobility  cross-layer physics register only

Datasets:
  dev_grid  — RSSI register; assembled fresh from runs/dev_grid (SUMO+ns-3).
  nextgen   — timing/map/overlap register; reuses the cached official-split
              assemblies under results/veremi_nextgen/checkpoints/assembled*.npz
              (no re-simulation), exactly the arrays the NextGen experiment scored.

Outputs (results/feature_ablation/):
  feature_ablation.csv  per (dataset, variant, model, seed): pr_auc, roc_auc,
                                f1, mcc, recall + grouped-by-transmitter bootstrap CI.
  feature_ablation_summary.csv  mean±std over seeds per (dataset, variant, model).
  provenance.json  git hash, seeds, platform, wall-time, caps.

Determinism: fixed seeds; dev_grid split via group_split(seed); NextGen uses the
official split (model-stochastic seeds only). Every number traces to this CSV.

Usage:
  python scripts/feature_ablation.py                  # full run
  python scripts/feature_ablation.py --datasets nextgen --models rbf_svm random_forest
  python scripts/feature_ablation.py --quick          # tiny caps, CPU seconds (quick check)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "results" / "feature_ablation"

VARIANTS = {                      # variant name -> raw block to ZERO (None = keep all)
    "crosslayer": None,
    "mobility_only": "network",
    "network_only": "mobility",
}
DEFAULT_MODELS = ["qsvm_crosslayer", "rbf_svm", "random_forest", "xgboost"]
DEFAULT_SEEDS = [7, 17, 27]


# --------------------------------------------------------------------------- #
# Dataset loaders -> (ExperimentData, idx{tr,va,te})
# --------------------------------------------------------------------------- #
def _load_dev_grid(features_cfg: dict, seed: int, max_windows: int | None):
    from qkv2x.experiments.common import group_split, load_experiment_data
    data = load_experiment_data(REPO / "runs" / "dev_grid", features_cfg)
    if max_windows:
        data = _cap_windows(data, max_windows, seed)
    sp = features_cfg["splits"]
    tr, va, te = group_split(data.groups, data.y, sp["test_frac"], sp["val_frac"], seed)
    return data, {"tr": tr, "va": va, "te": te}


def _load_nextgen(seed: int, max_windows: int | None):
    from qkv2x.experiments.common import ExperimentData
    ck = REPO / "results" / "veremi_nextgen" / "checkpoints"
    cand = ck / ("assembled.npz" if seed == 7 else f"assembled_s{seed}.npz")
    if not cand.exists():
        raise FileNotFoundError(
            f"NextGen assembly {cand.name} missing — run the veremi_nextgen experiment "
            "first (it writes the checkpointed assemblies this script reuses).")
    z = np.load(cand, allow_pickle=True)
    data = ExperimentData(
        X=z["X"], y=z["y"].astype(int), attack_name=z["attack_name"].astype(str),
        groups=z["groups"].astype(int), aoi=z["aoi"], t_end=z["t_end"],
        feature_names=[str(s) for s in z["feature_names"]],
        blocks=[str(s) for s in z["blocks"]],
        points=pd.DataFrame(), meta=pd.DataFrame(),
    )
    split = z["split"].astype(str)
    idx = {"tr": np.where(split == "Train")[0],
           "va": np.where(split == "Validation")[0],
           "te": np.where(split == "Test")[0]}
    if max_windows:                      # cap train only (keeps the official test intact)
        data, idx = _cap_train(data, idx, max_windows, seed)
    return data, idx


def _cap_windows(data, max_windows: int, seed: int):
    """Group-aware down-sample of an ExperimentData to ~max_windows (quick mode only)."""
    rng = np.random.default_rng(seed)
    gs = rng.permutation(np.unique(data.groups))
    keep, tot = [], 0
    for g in gs:
        keep.append(g)
        tot += int((data.groups == g).sum())
        if tot >= max_windows:
            break
    mask = np.isin(data.groups, keep)
    return data.subset(mask)


def _cap_train(data, idx, max_windows: int, seed: int):
    from qkv2x.experiments.common import _cap_train_groupwise
    idx = dict(idx)
    idx["tr"] = _cap_train_groupwise(idx["tr"], data.groups, max_windows, seed)
    return data, idx


# --------------------------------------------------------------------------- #
# Ablated reduction
# --------------------------------------------------------------------------- #
def _reduce_variant(data, idx_tr, reduction_cfg, zero_block):
    """Zero `zero_block` raw features, then fit BlockReducer on the train rows and
    transform all rows — the same recipe as the NextGen netblind ablation."""
    from qkv2x.features.reduce import BlockReducer
    X = data.X.copy()
    if zero_block is not None:
        cols = [i for i, b in enumerate(data.blocks) if b == zero_block]
        X[:, cols] = 0.0
    reducer = BlockReducer(reduction_cfg).fit(X[idx_tr], list(data.blocks))
    return reducer.transform(X)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    from qkv2x.config import load_yaml
    from qkv2x.experiments.common import run_model
    from qkv2x.models.metrics import detection_report, grouped_bootstrap_ci

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="+", default=["dev_grid", "nextgen"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--max-windows", type=int, default=None,
                    help="group-aware cap (dev_grid: whole set; nextgen: train only).")
    ap.add_argument("--quick", action="store_true",
                    help="tiny preset for a quick check (nextgen, 1 seed, small caps).")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    features_cfg = load_yaml(REPO / "configs" / "features.yaml")
    quantum_cfg = load_yaml(REPO / "configs" / "quantum.yaml")
    baselines_cfg = load_yaml(REPO / "configs" / "baselines.yaml")

    if args.quick:                       # fast, deterministic quick path
        args.datasets = [d for d in args.datasets if d == "nextgen"] or ["nextgen"]
        args.seeds = args.seeds[:1]
        args.n_boot = min(args.n_boot, 50)
        args.max_windows = args.max_windows or 300
        quantum_cfg = {**quantum_cfg, "max_train": min(quantum_cfg.get("max_train", 3000), 200),
                       "train": {**quantum_cfg.get("train", {}), "epochs": 3}}

    t0 = time.time()
    rows = []
    for dataset in args.datasets:
        for seed in args.seeds:
            base_feats = {**features_cfg,
                          "splits": {**features_cfg["splits"], "seed": seed}}
            if dataset == "dev_grid":
                data, idx = _load_dev_grid(base_feats, seed, args.max_windows)
            elif dataset == "nextgen":
                data, idx = _load_nextgen(seed, args.max_windows)
            else:
                raise ValueError(f"unknown dataset {dataset}")
            cfgs = {"quantum": quantum_cfg, "baselines": baselines_cfg,
                    "features": base_feats}
            for variant in args.variants:
                Xr = _reduce_variant(data, idx["tr"], base_feats["reduction"],
                                     VARIANTS[variant])
                for model in args.models:
                    res = run_model(model, data, Xr, idx, cfgs)
                    rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
                    ci = grouped_bootstrap_ci(res["y_true"], res["scores"],
                                              res["groups"], n_boot=args.n_boot,
                                              seed=seed)
                    rows.append({
                        "dataset": dataset, "variant": variant, "model": model,
                        "seed": seed, "n_test": rep["n"], "n_pos": rep["n_pos"],
                        "pr_auc": rep["pr_auc"], "roc_auc": rep["roc_auc"],
                        "f1": rep["f1"], "mcc": rep["mcc"], "recall": rep["recall"],
                        "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
                        "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
                    })
                    print(f"[a3] {dataset:8s} {variant:13s} {model:16s} seed={seed} "
                          f"PR-AUC={rep['pr_auc']:.3f} ROC={rep['roc_auc']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "feature_ablation.csv", index=False)

    summ = (df.groupby(["dataset", "variant", "model"], as_index=False)
              .agg(pr_auc_mean=("pr_auc", "mean"), pr_auc_std=("pr_auc", "std"),
                   roc_auc_mean=("roc_auc", "mean"), roc_auc_std=("roc_auc", "std"),
                   f1_mean=("f1", "mean"), mcc_mean=("mcc", "mean"),
                   n_seeds=("seed", "nunique")))
    summ.to_csv(out / "feature_ablation_summary.csv", index=False)

    prov = _provenance(args, t0)
    (out / "provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"[a3] wrote {len(df)} rows -> {out}/feature_ablation.csv "
          f"({time.time()-t0:.1f}s)")
    return df


def _provenance(args, t0):
    import platform
    prov = {
        "script": "scripts/feature_ablation.py",
        "datasets": args.datasets, "variants": args.variants,
        "models": args.models, "seeds": args.seeds, "n_boot": args.n_boot,
        "max_windows": args.max_windows, "quick": args.quick,
        "wall_time_s": round(time.time() - t0, 2),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    try:
        from qkv2x.utils.provenance import capture_provenance
        prov["git"] = capture_provenance().get("git", {})
    except Exception:
        import subprocess
        try:
            prov["git_commit"] = subprocess.check_output(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                text=True).strip()
        except Exception:
            prov["git_commit"] = "unknown"
    return prov


if __name__ == "__main__":
    main()
