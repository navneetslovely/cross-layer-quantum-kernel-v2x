#!/usr/bin/env python3
"""kta_restarts — KTA restart + validation-selection study.

Detector protocol of scripts/dev_multiseed.py (dev_grid; splits 7/17/27; train
group-capped at quantum.yaml::max_train -> 3,023; test group-capped at 6,000 with
seed+1; reducer from results/dev_grid_multiseed/checkpoints/reduced_s{seed}.npz).

For the cross-layer map on every split, and for the topology arms (random-sparse,
no-entanglement, trained all-pairs ZZ) on seed 7, alignment training is repeated from
R=5 restarts (restart 0 = the published batch stream, seed 11, constant init; restarts
1-4 = new batch streams + 10% relative jitter of the initial angles). Alignment is also
evaluated every epoch on a fixed class-balanced 2,048-window subset of the (otherwise
unused) validation split. Three selection rules are scored on the test set:
best training batch (the published rule), best validation alignment, final epoch.

Outputs (results/studies/):
  kta_restarts.csv  one row per (seed, model, restart, rule)
  kta_restarts_summary.json  means/sds, variance decomposition, topology contrasts,
                             best-batch optimism, protocol/metadata
  kta_restarts_curves.npz  per-run training / validation alignment curves
  kta_restarts_alignment_curves.png  figure: train vs validation alignment
Checkpoints: results/studies/kta_restarts_ckpt/ (resume-safe per run).
"""
from __future__ import annotations

import json
import subprocess
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
                                      fit_reducer, group_split,
                                      make_feature_map)              # noqa: E402
from qkv2x.features.crosslayer import FEATURE_SPEC                   # noqa: E402
from qkv2x.models.metrics import detection_report, grouped_bootstrap_ci  # noqa: E402
from qkv2x.models.qsvm import QuantumKernelSVM                       # noqa: E402
from qkv2x.quantum.kernels import fidelity_gram                      # noqa: E402
from qkv2x.quantum.train import train_feature_map_validated          # noqa: E402

OUT = REPO / "results" / "studies"
CK = OUT / "kta_restarts_ckpt"
CKPT_DIR = REPO / "results" / "dev_grid_multiseed" / "checkpoints"
SEEDS = [7, 17, 27]
RESTART_SEEDS = [11, 23, 41, 57, 71]
JITTER = [0.0, 0.10, 0.10, 0.10, 0.10]
RULES = ["best_batch", "best_val", "final"]
ARMS_SEED7 = ["qsvm_sparse", "qsvm_none", "qsvm_zz_trained"]
N_VAL = 2048
N_BOOT = 200


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


def _split(data, cfgs, seed):
    s = cfgs["features"]["splits"]
    tr, va, te = group_split(data.groups, data.y, s["test_frac"], s["val_frac"], seed)
    tr = _cap_train_groupwise(tr, data.groups, cfgs["quantum"].get("max_train", 3000), seed)
    te = _cap_train_groupwise(te, data.groups, 6000, seed + 1)
    rpath = CKPT_DIR / f"reduced_s{seed}.npz"
    if rpath.exists():
        Xr = np.load(rpath)["Xr"]
    else:
        red = fit_reducer(cfgs["features"]["reduction"], data.X, data.blocks, tr)
        Xr = red.transform(data.X)
    # fixed class-balanced validation subset (never the test split)
    rng = np.random.default_rng(1000 + seed)
    pos = va[data.y[va] > 0]
    neg = va[data.y[va] <= 0]
    k = min(N_VAL // 2, len(pos), len(neg))
    val = np.concatenate([rng.choice(pos, k, replace=False), rng.choice(neg, k, replace=False)])
    return tr, va, te, val, Xr


def _fm_for(model, cfgs):
    base = "qsvm_zz" if model == "qsvm_zz_trained" else model
    return make_feature_map(base, cfgs["quantum"], cfgs["features"]["reduction"], cfgs["features"])


def _train(model, seed, r, data, Xr, tr, val, cfgs):
    ck = CK / f"train_{model}_s{seed}_r{r}.pt"
    if ck.exists():
        st = torch.load(ck, weights_only=False)
        return st["params"], st["history"]
    fm = _fm_for(model, cfgs)
    cfg = dict(cfgs["quantum"]["train"])
    cfg["seed"] = RESTART_SEEDS[r]
    cfg["init_jitter"] = JITTER[r]
    cfg["jitter_seed"] = 100 * seed + r
    t0 = time.time()
    params, hist = train_feature_map_validated(
        fm, Xr[tr], data.y[tr], data.aoi[tr], cfg, Xr[val], data.y[val], data.aoi[val])
    hist["train_seconds"] = time.time() - t0
    torch.save({"params": params, "history": hist, "cfg": cfg}, ck)
    return params, hist


def _score(model, seed, r, rule, params, data, Xr, tr, te, cfgs):
    ck = CK / f"score_{model}_s{seed}_r{r}_{rule}.json"
    if ck.exists():
        return json.loads(ck.read_text())
    fm = _fm_for(model, cfgs)
    with torch.no_grad():
        s_tr = fm.statevectors(Xr[tr], data.aoi[tr], params)
        K_tr = fidelity_gram(s_tr)
        K_te = fidelity_gram(fm.statevectors(Xr[te], data.aoi[te], params), s_tr)
    qc = cfgs["quantum"]
    svm = QuantumKernelSVM(C_grid=qc["svm"]["C_grid"], cv_folds=qc["svm"]["cv_folds"])
    svm.fit(K_tr, data.y[tr], groups=data.groups[tr])
    scores, preds = svm.decision_function(K_te), svm.predict(K_te)
    rep = detection_report(data.y[te], scores, preds)
    ci = grouped_bootstrap_ci(data.y[te], scores, data.groups[te], n_boot=N_BOOT, seed=seed)
    out = {"roc_auc": rep["roc_auc"], "pr_auc": rep["pr_auc"], "f1": rep.get("f1"),
           "mcc": rep.get("mcc"), "roc_lo": ci["roc_auc"]["lo"], "roc_hi": ci["roc_auc"]["hi"],
           "pr_lo": ci["pr_auc"]["lo"], "pr_hi": ci["pr_auc"]["hi"], "C": svm.best_C_,
           "n_train": int(len(tr)), "n_test": int(len(te)), "n_sv": int(len(svm.svc_.support_))}
    ck.write_text(json.dumps(out, default=float))
    return out


def _git_head():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip()
    except Exception:
        return None


def main():
    CK.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    data = _load()
    cfgs = {"features": load_yaml(str(REPO / "configs/features.yaml")),
            "quantum": load_yaml(str(REPO / "configs/quantum.yaml")),
            "baselines": load_yaml(str(REPO / "configs/baselines.yaml"))}
    rows, curves = [], {}
    jobs = [(s, "qsvm_crosslayer") for s in SEEDS] + [(7, m) for m in ARMS_SEED7]
    for seed, model in jobs:
        tr, va, te, val, Xr = _split(data, cfgs, seed)
        for r in range(len(RESTART_SEEDS)):
            params, hist = _train(model, seed, r, data, Xr, tr, val, cfgs)
            curves[f"{model}_s{seed}_r{r}_train"] = np.asarray(hist["alignment"])
            curves[f"{model}_s{seed}_r{r}_val"] = np.asarray(hist["val_alignment"])
            plateau = float(np.mean(hist["alignment"][-50:]))
            for rule in RULES:
                sc = _score(model, seed, r, rule, params[rule], data, Xr, tr, te, cfgs)
                rows.append({"seed": seed, "model": model, "restart": r,
                             "batch_seed": RESTART_SEEDS[r], "init_jitter": JITTER[r],
                             "rule": rule, **sc,
                             "best_train_align": hist["best_alignment"],
                             "best_train_epoch": hist["best_epoch"],
                             "best_val_align": hist["best_val_alignment"],
                             "best_val_epoch": hist["best_val_epoch"],
                             "final_train_align": hist["final_alignment"],
                             "final_val_align": hist["final_val_alignment"],
                             "train_plateau_last50": plateau,
                             "train_seconds": hist.get("train_seconds")})
                print(f"[kta_restarts] {model} s{seed} r{r} {rule:10s} ROC={sc['roc_auc']:.4f} "
                      f"PR={sc['pr_auc']:.4f} C={sc['C']}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kta_restarts.csv", index=False)
    np.savez_compressed(OUT / "kta_restarts_curves.npz", **curves)

    # ---------- summary ----------
    summ = {"protocol": {
        "dataset": "dev_grid (runs/dev_grid/features.parquet)",
        "splits": SEEDS, "train_cap": cfgs["quantum"].get("max_train", 3000),
        "test_cap": 6000, "reducer": "results/dev_grid_multiseed/checkpoints/reduced_s{seed}.npz",
        "restarts": len(RESTART_SEEDS), "batch_seeds": RESTART_SEEDS, "init_jitter": JITTER,
        "restart0_note": "restart 0 = published batch stream (seed 11), constant init -> reproduces the published run",
        "validation_subset": f"class-balanced {N_VAL} windows from the validation split, rng 1000+seed, evaluated every epoch",
        "rules": RULES, "kta": dict(cfgs["quantum"]["train"]),
        "svm": dict(cfgs["quantum"]["svm"]), "n_boot": N_BOOT,
        "interpreter": sys.executable, "git_head": _git_head(),
        "torch": torch.__version__, "runtime_s": None}}
    cl = df[df.model == "qsvm_crosslayer"]
    per_rule = {}
    for rule, g in cl.groupby("rule"):
        seed_means = g.groupby("seed")["roc_auc"].mean()
        within = g.groupby("seed")["roc_auc"].var(ddof=1)
        per_rule[rule] = {
            "roc_mean": float(g.roc_auc.mean()), "roc_sd_all15": float(g.roc_auc.std(ddof=1)),
            "pr_mean": float(g.pr_auc.mean()), "pr_sd_all15": float(g.pr_auc.std(ddof=1)),
            "roc_by_seed_mean": {int(k): float(v) for k, v in seed_means.items()},
            "roc_by_seed_restart_sd": {int(k): float(np.sqrt(v)) for k, v in within.items()},
            "restart_sd_pooled": float(np.sqrt(within.mean())),
            "split_sd_of_restart_means": float(seed_means.std(ddof=1)),
            "var_within_restarts": float(within.mean()),
            "var_between_splits": float(seed_means.var(ddof=1)),
        }
    summ["crosslayer_by_rule"] = per_rule
    # rule deltas vs the published rule, paired by (seed, restart)
    piv = cl.pivot_table(index=["seed", "restart"], columns="rule", values="roc_auc")
    summ["rule_deltas_vs_best_batch"] = {
        rule: {"mean": float((piv[rule] - piv["best_batch"]).mean()),
               "sd": float((piv[rule] - piv["best_batch"]).std(ddof=1)),
               "n": int(piv.shape[0])} for rule in ("best_val", "final")}
    # published anchor: seed 7, restart 0, best_batch should equal the published 0.9254
    anchor = cl[(cl.seed == 7) & (cl.restart == 0) & (cl.rule == "best_batch")]
    summ["anchor_seed7_restart0_best_batch_roc"] = (float(anchor.roc_auc.iloc[0]) if len(anchor) else None)
    summ["published_seed7_roc"] = 0.9254389652792645
    # best-batch optimism
    opt = cl[cl.rule == "best_batch"]
    summ["best_batch_optimism"] = {
        "best_train_minus_plateau_mean": float((opt.best_train_align - opt.train_plateau_last50).mean()),
        "best_train_minus_final_mean": float((opt.best_train_align - opt.final_train_align).mean()),
        "best_train_epochs": opt.best_train_epoch.tolist(),
        "best_val_epochs": opt.best_val_epoch.tolist()}
    # topology contrasts at seed 7 (per rule), over restarts
    contrasts = {}
    s7 = df[df.seed == 7]
    for rule in RULES:
        base = s7[(s7.model == "qsvm_crosslayer") & (s7.rule == rule)].sort_values("restart").roc_auc.to_numpy()
        contrasts[rule] = {"crosslayer_mean": float(base.mean()), "crosslayer_sd": float(base.std(ddof=1))}
        for arm in ARMS_SEED7:
            a = s7[(s7.model == arm) & (s7.rule == rule)].sort_values("restart").roc_auc.to_numpy()
            if len(a) == 0:
                continue
            d = base - a                     # paired by restart index
            n = len(d)
            tcrit = 2.776 if n == 5 else 2.0
            se = d.std(ddof=1) / np.sqrt(n)
            # Welch (unpaired) as well
            sw = np.sqrt(base.var(ddof=1) / len(base) + a.var(ddof=1) / len(a))
            contrasts[rule][f"crosslayer_minus_{arm}"] = {
                "arm_mean": float(a.mean()), "arm_sd": float(a.std(ddof=1)),
                "delta_mean": float(d.mean()), "delta_ci95_paired": [float(d.mean() - tcrit * se), float(d.mean() + tcrit * se)],
                "delta_ci95_welch": [float(d.mean() - 2.306 * sw), float(d.mean() + 2.306 * sw)],
                "n_restarts": int(n)}
    summ["topology_contrasts_seed7"] = contrasts
    summ["protocol"]["runtime_s"] = time.time() - t_start
    (OUT / "kta_restarts_summary.json").write_text(json.dumps(summ, indent=1, default=float))

    # ---------- figure ----------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), sharey=True)
        for ax, seed in zip(axes, SEEDS):
            for r in range(len(RESTART_SEEDS)):
                tr_c = curves[f"qsvm_crosslayer_s{seed}_r{r}_train"]
                va_c = curves[f"qsvm_crosslayer_s{seed}_r{r}_val"]
                ax.plot(tr_c, color="0.6", lw=0.6, alpha=0.7, label="training batch" if r == 0 else None)
                ax.plot(va_c, color="C0", lw=1.0, label="validation subset" if r == 0 else None)
            ax.set_title(f"split seed {seed}")
            ax.set_xlabel("epoch")
        axes[0].set_ylabel("centred alignment")
        axes[0].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT / "kta_restarts_alignment_curves.png", dpi=200)
    except Exception as exc:  # figure is optional
        print(f"[kta_restarts] figure skipped: {exc}")
    print(json.dumps({"status": "done", "rows": len(df), "runtime_s": summ["protocol"]["runtime_s"]}))


if __name__ == "__main__":
    main()
