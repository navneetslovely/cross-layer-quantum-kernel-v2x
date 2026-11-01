"""qkv2x command-line interface.

Subcommands
  cosim  run the SUMO+ns-3 co-simulation sweep (writes runs/<scenario>/...)
  features  build + cache the windowed cross-layer feature table for a run set
  baselines  evaluate the classical baselines on a run set (quick holdout)
  quantum  evaluate the quantum-kernel detectors on a run set (quick holdout)
  experiment  run a full experiment (robustness | lowlabel | crossattack | veremi)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import load_yaml


# --------------------------------------------------------------------------- #
def _cmd_cosim(args):
    from .cosim.run_matrix import run_matrix
    summary = run_matrix(args.config)
    print(json.dumps(summary, indent=2))


def _build_combined(runs, features_cfg):
    from .experiments.common import load_experiment_data
    return load_experiment_data(runs, features_cfg)


def _cmd_features(args):
    features_cfg = load_yaml(args.config)
    data = _build_combined(args.runs, features_cfg)
    df = pd.DataFrame(data.X, columns=data.feature_names)
    df["y"] = data.y
    df["attack_name"] = data.attack_name
    df["group"] = data.groups
    df["aoi"] = data.aoi
    df["t_end"] = data.t_end
    for col in data.points.columns:
        df[f"point_{col}"] = data.points[col].to_numpy()
    out = Path(args.out) if args.out else Path(args.runs) / "features.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    print(f"windows: {len(df)}  positives: {int(data.y.sum())} "
          f"({100*np.mean(data.y):.2f}%)  features: {len(data.feature_names)}")
    print(f"attacks: {pd.Series(data.attack_name).value_counts().to_dict()}")
    print(f"saved -> {out}")


def _evaluate(args, models):
    from .experiments.common import (fit_reducer, group_split,
                                     load_experiment_data, run_model)
    from .models.metrics import detection_report
    features_cfg = load_yaml(args.features)
    cfgs = {"features": features_cfg,
            "quantum": load_yaml(args.quantum) if getattr(args, "quantum", None) else load_yaml("configs/quantum.yaml"),
            "baselines": load_yaml(args.config) if args.config else load_yaml("configs/baselines.yaml")}
    data = load_experiment_data(args.runs, features_cfg)
    s = features_cfg["splits"]
    tr, va, te = group_split(data.groups, data.y, s["test_frac"], s["val_frac"], s["seed"])
    reducer = fit_reducer(features_cfg["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)
    rows = []
    for m in models:
        res = run_model(m, data, Xr, {"tr": tr, "va": va, "te": te}, cfgs)
        rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
        rows.append({"model": m, **{k: rep.get(k) for k in
                     ["pr_auc", "roc_auc", "f1", "mcc", "fpr_at_tpr"]}})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.out, index=False)


def _cmd_baselines(args):
    models = ["rbf_svm", "random_forest", "knn_bagging", "mlp", "rules"]
    _evaluate(args, models)


def _cmd_quantum(args):
    models = ["qsvm_crosslayer", "qsvm_zz"]
    _evaluate(args, models)


def _cmd_experiment(args):
    cfg = load_yaml(args.config)
    kind = cfg.get("kind")
    if kind is None:
        raise SystemExit("experiment config must declare 'kind:' "
                         "(robustness|packetloss|lowlabel|crossattack|veremi|"
                         "veremi_nextgen|ablation|intensity)")
    if kind == "robustness":
        from .experiments.robustness import run
    elif kind == "packetloss":
        from .experiments.packetloss import run
    elif kind == "lowlabel":
        from .experiments.lowlabel import run
    elif kind == "crossattack":
        from .experiments.crossattack import run
    elif kind == "veremi":
        from .experiments.veremi_exp import run
    elif kind == "veremi_nextgen":
        from .experiments.veremi_nextgen_exp import run
    elif kind == "ablation":
        from .experiments.ablation import run
    elif kind == "intensity":
        from .experiments.intensity import run
    else:
        raise SystemExit(f"unknown experiment kind: {kind}")
    print(json.dumps(run(args.config), indent=2, default=float))


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="qkv2x", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("cosim", help="run the SUMO+ns-3 co-simulation sweep")
    c.add_argument("--config", required=True)
    c.set_defaults(func=_cmd_cosim)

    f = sub.add_parser("features", help="build the windowed feature table")
    f.add_argument("--runs", required=True)
    f.add_argument("--config", required=True, help="features.yaml")
    f.add_argument("--out", default=None)
    f.set_defaults(func=_cmd_features)

    b = sub.add_parser("baselines", help="evaluate classical baselines")
    b.add_argument("--runs", required=True)
    b.add_argument("--features", required=True)
    b.add_argument("--config", default=None, help="baselines.yaml")
    b.add_argument("--quantum", default=None)
    b.add_argument("--out", default=None)
    b.set_defaults(func=_cmd_baselines)

    q = sub.add_parser("quantum", help="evaluate quantum-kernel detectors")
    q.add_argument("--runs", required=True)
    q.add_argument("--features", required=True)
    q.add_argument("--config", default=None, help="baselines.yaml (for shared cfg)")
    q.add_argument("--quantum", default=None, help="quantum.yaml")
    q.add_argument("--out", default=None)
    q.set_defaults(func=_cmd_quantum)

    e = sub.add_parser("experiment", help="run a full experiment")
    e.add_argument("--config", required=True)
    e.set_defaults(func=_cmd_experiment)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
