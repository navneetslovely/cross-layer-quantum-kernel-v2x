"""H2 — robustness vs packet loss, via controlled injection.

After the BUG-4 jitter fix the 802.11p channel no longer saturates at the swept
densities (reception ~95-99%), so sweeping vehicle density produces an almost flat
degradation curve. This experiment instead injects packet loss *directly*: at each
level p in `packet_loss_grid`, a fraction p of receptions is dropped at random before
windowing (faithful message-level loss; windows thin, some fall below min_messages),
and every model is trained AND evaluated on that thinned data. The result is the H2
degradation curve — detection metric vs packet loss — with grouped-bootstrap CIs and a
quantum-vs-classical significance test at each level.

Comparison is matched-N (quantum can't use the full split, O(N^2) Gram) with a
`<model>__full` classical-ceiling context, exactly as in crossattack/robustness.

CAVEAT (survivorship): because loss preferentially removes *sparse* windows (those near
`min_messages`), the surviving population at high p is a denser, easier-to-classify subset
— so metrics can RISE at moderate loss before the windowable population collapses. The
curve therefore mixes feature-quality degradation (hurts) with window selection (helps).
Read it as "detection on windows that still form under loss," and report the survivorship
honestly. A cleaner, strictly-monotonic feature-degradation curve would fix a test
population at p=0 and degrade only those windows' features (retaining sub-min windows) —
a larger change, deferred unless the headline needs it.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_yaml
from ..models.metrics import (detection_report, grouped_bootstrap_ci,
                              paired_bootstrap_auc_diff)
from ..utils.plotting import degradation_curves
from ..utils.provenance import capture_provenance
from .common import (CLASSICAL_MODELS, _cap_train_groupwise, fit_reducer,
                     group_split, load_experiment_data, matched_train_idx,
                     predictions_long, run_model)


def run(config_path: str) -> dict:
    t0 = time.time()
    started = datetime.now().isoformat(timespec="seconds")
    cfg = load_yaml(config_path)
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_boot = int(cfg.get("bootstrap", 200))
    full_cap = int(cfg.get("full_cap", 15000))

    features_cfg = load_yaml(cfg["features_config"])
    cfgs = {
        "features": features_cfg,
        "quantum": load_yaml(cfg["quantum_config"]),
        "baselines": load_yaml(cfg["baselines_config"]),
    }
    splits = features_cfg["splits"]
    seed = int(splits["seed"])
    p_grid = [float(p) for p in cfg["packet_loss_grid"]]

    rows = []
    pred_rows = []
    res_by = {}                 # (p, model) -> result, for paired significance
    for p in p_grid:
        # Rebuild the windowed dataset with p of receptions dropped (message-level loss).
        data = load_experiment_data(cfg["runs_root"], features_cfg,
                                    drop_frac=p, drop_seed=seed)
        tr, va, te = group_split(data.groups, data.y, splits["test_frac"],
                                 splits["val_frac"], seed)
        reducer = fit_reducer(features_cfg["reduction"], data.X, data.blocks, tr)
        Xr = reducer.transform(data.X)
        tr_match = matched_train_idx(tr, data, cfgs)     # matched-N for the significance test
        print(f"[packetloss] p={p:.2f}: {len(data)} windows "
              f"({100*np.mean(data.y):.1f}% pos), train={len(tr_match)} test={len(te)}",
              file=sys.stderr)

        for model in cfg["models"]:
            res = run_model(model, data, Xr, {"tr": tr_match, "va": va, "te": te}, cfgs)
            res_by[(p, model)] = res
            pred_rows.extend(predictions_long(res, packet_loss=p))
            rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
            ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"],
                                      n_boot=n_boot, seed=seed)
            rows.append({
                "model": model, "packet_loss": p, "n_train": int(len(tr_match)),
                "n_windows": int(len(data)),
                **{k: rep.get(k) for k in
                   ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall", "fpr_at_tpr"]},
                "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
                "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
                "n_test": rep["n"], "n_pos": rep["n_pos"],
            })

        # classical@full-data context (bounded by full_cap; quantum can't reach it).
        tr_full = _cap_train_groupwise(tr, data.groups, full_cap, seed)
        if len(tr_match) < len(tr_full):
            for model in [m for m in cfg["models"] if m in CLASSICAL_MODELS]:
                res = run_model(model, data, Xr,
                                {"tr": tr_full, "va": va, "te": te}, cfgs)
                rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
                rows.append({
                    "model": f"{model}__full", "packet_loss": p,
                    "n_train": int(len(tr_full)), "n_windows": int(len(data)),
                    **{k: rep.get(k) for k in
                       ["pr_auc", "roc_auc", "f1", "mcc", "precision", "recall", "fpr_at_tpr"]},
                    "n_test": rep["n"], "n_pos": rep["n_pos"]})

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / "packetloss_results.csv", index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_parquet(out_dir / "predictions.parquet")

    # significance: each quantum model vs rbf_svm at each loss level (same test windows).
    sig_rows = []
    for p in p_grid:
        ref = res_by.get((p, "rbf_svm"))
        if ref is None:
            continue
        for qm in ("qsvm_crosslayer", "qsvm_zz", "qsvm_projected"):
            r = res_by.get((p, qm))
            if r is None:
                continue
            for metric in ("roc_auc", "pr_auc"):
                d = paired_bootstrap_auc_diff(r["y_true"], r["scores"], ref["scores"],
                                              r["groups"], n_boot=n_boot,
                                              seed=seed, metric=metric)
                sig_rows.append({"packet_loss": p, "model": qm, "vs": "rbf_svm", **d})
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(out_dir / "significance.csv", index=False)

    # degradation curves: metric vs packet loss, with CI bands (matched models only).
    if not results.empty:
        matched = results[~results.model.str.endswith("__full")]
        for metric in ("pr_auc", "roc_auc"):
            degradation_curves(matched, "packet_loss", metric, "model",
                               out_dir / f"{metric}_vs_packetloss.png",
                               f"{metric.upper()} vs packet loss (H2)",
                               ci=(f"{metric}_lo", f"{metric}_hi"))

    elapsed = round(time.time() - t0, 1)
    summary = {"out_dir": str(out_dir), "n_rows": len(results),
               "packet_loss_grid": p_grid, "models": cfg["models"],
               "n_bootstrap": n_boot, "wall_time_s": elapsed,
               "artifacts": ["packetloss_results.csv", "predictions.parquet",
                             "significance.csv", "pr_auc_vs_packetloss.{png,pdf}",
                             "roc_auc_vs_packetloss.{png,pdf}", "provenance.json"]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    capture_provenance(out_dir, timestamp=started, config_path=config_path,
                       extra={"experiment": "packetloss", "wall_time_s": elapsed,
                              "split_seed": seed, "n_bootstrap": n_boot})
    return summary
