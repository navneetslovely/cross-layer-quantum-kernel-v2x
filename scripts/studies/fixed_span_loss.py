#!/usr/bin/env python3
"""fixed_span -- fixed-span (time-based) packet-loss degradation curve.

Provides a time-based window alternative that removes the packet-loss
survivorship problem AT THE SOURCE rather than accounting for it after the fact.

WHY THE PUBLISHED CURVE IS NOT AN END-TO-END DEGRADATION CURVE
--------------------------------------------------------------
`src/qkv2x/experiments/packetloss.py` rebuilds the whole windowed dataset at each
loss level p. Because a published window is W=10 consecutive received MESSAGES of
a (RSU, claimed-id) stream kept only if it holds >= 5 of them, injecting loss
RE-WINDOWS every stream: window identity changes, sparse windows fall below the
floor, and the population shrinks 90,553 -> 71,783 -> 53,048 -> 34,484 -> 16,393
windows across p = 0, .2, .4, .6, .8. The survivors at high loss are a denser,
easier subset, so the reported AUCs RISE with loss. That module's own docstring
(L15-22) names the fix: "fix a test population at p=0 and degrade only those
windows' features (retaining sub-min windows)". This script implements exactly that.

WHAT THIS SCRIPT DOES
---------------------
  (A) GUARD. Rebuilds dev_grid through the ordinary published entry point
      (`load_experiment_data(runs/dev_grid, configs/features.yaml)`) with NO new
      config keys and asserts the count path is byte-identical to the shipped
      `runs/dev_grid/features.parquet`: 90,553 windows, and exact array equality
      on all 19 features, y, attack_name, group, aoi, t_end.
  (B) SPANS. Derives the fixed-duration span population ONCE, from the p=0 build,
      with the new opt-in `window.mode='time'` (span_s = stride_s = 1.0 s, a
      non-overlapping tiling of the global grid [k, k+1)). The 1.0 s choice is
      VERIFIED against the data, not assumed: the published 10-message windows
      have a median duration of ~0.92 s and the observed within-stream
      inter-arrival is ~0.1 s (10 Hz BSMs), so a 1.0 s span carries the same
      nominal 10 messages as the published window.
  (C) DEGRADE. For each p in {0, .2, .4, .6, .8} (the grid of
      configs/experiments/packetloss.yaml) messages are dropped with EXACTLY
      build_dataset's own seeded rule (drop_frac=p, drop_seed=7+run_idx, applied
      after the RSU filter), then features are recomputed ON THE FROZEN SPANS with
      `build_dataset(..., fixed_spans=...)`, which applies no min_messages filter,
      so thin and even EMPTY spans are retained (meta['n_msgs'] identifies them).
      Since the per-run uniforms are drawn from the same seed and the same row
      count at every level, the retained message sets are NESTED in p (verified).
  (D) EVALUATE. One group-aware split is drawn ONCE on the frozen population and
      reused at every level, so the train / val / test SPANS are literally the
      same objects at every p; labels and groups are frozen at p=0 (label re-flips
      under loss are counted and reported, never applied). Two arms:
        retrained  -- the detector is refit at each level on that level's train
                      spans (the "operator retrains for the channel" reading);
        frozen_p0  -- the p=0 detector is deployed unchanged and only its INPUT
                      degrades (the "deployed detector" reading; classical only,
                      because run_model does not expose a fitted quantum model).
      Operating point: threshold = the (1-0.01) quantile of the model's scores on
      the p=0 VALIDATION benign spans (threshold_protocol's protocol; positive iff score > thr),
      derived once per (arm, model, level) and APPLIED UNCHANGED to the level-p
      test spans, so the realised FPR is allowed to drift -- that drift is part of
      the answer. A re-tuned-on-test comparator is reported alongside as an
      upper bound, never as the headline.
  (E) COMPARE. Three curves: this fixed-span curve, the published survivors-only
      curve (results/packetloss_dev/packetloss_results.csv) and the
      failures-counted variant (results/studies/packetloss_failures_variant.csv,
      from packetloss_retention.py).

SRC SUPPORT (guarded and opt-in):
  src/qkv2x/features/dataset.py gains (i) `window.mode = 'count' | 'time'`
  (default 'count'), (ii) `window.span_s` / `window.stride_s`, and (iii) a
  `fixed_spans=` argument. With none of them supplied the count path is unchanged
  -- which is what step (A) asserts on the real 90,553-window dataset.

Determinism: split seed 7 (features.yaml), drop seeds 7+run_idx (the published
rule), model seeds from baselines.yaml/quantum.yaml, bootstrap seed 20260807.
Every protocol choice is echoed into fixed_span_summary.json["protocol"].

Usage:
    python scripts/studies/fixed_span_loss.py \
        2>&1 | tee results/studies/fixed_span_run.log

Outputs (results/studies/):
    fixed_span_loss.csv  per (arm, model, packet_loss) metrics + CIs
    fixed_span_three_curves.csv  fixed-span vs survivors-only vs failures-counted
    fixed_span_population.csv  per-level frozen-population census (n_msgs etc.)
    fixed_span_predictions.parquet  per-span test scores (all arms/models/levels)
    fixed_span_features.npz  frozen spans + per-level features (quantum-arm
                                 inputs, so the study is re-analysable without a
                                 rebuild)
    fixed_span_summary.json  protocol + verified checks + headline numbers
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from qkv2x.config import load_yaml  # noqa: E402
from qkv2x.cosim.loader import iter_runs, load_meta, load_rx  # noqa: E402
from qkv2x.experiments.common import (ExperimentData, _cap_train_groupwise,  # noqa: E402
                                      fit_reducer, group_split,
                                      matched_train_idx, run_model)
from qkv2x.features.dataset import build_dataset  # noqa: E402
from qkv2x.models.baselines import baseline_scores, build_baseline  # noqa: E402

RUNS = REPO / "runs" / "dev_grid"
OUT = REPO / "results" / "studies"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------- protocol constants ----------------------------
SPAN_S = 1.0             # span duration [s]  (justified against the data in step B)
STRIDE_S = 1.0           # stride [s]; == SPAN_S => non-overlapping tiling
P_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]      # configs/experiments/packetloss.yaml
PUBLISHED_N_WINDOWS = 90_553            # the count-path guard value
FPR_TARGET = 0.01
N_BOOT = 500
BOOT_SEED = 20260807
VAL_THR_CAP = 8000       # p=0 validation spans used for the threshold (memory cap)
VAL_CAP_SEED = 4242
CLASSICAL = ["rbf_svm", "random_forest"]
QUANTUM = ["qsvm_crosslayer", "qsvm_zz"]
SPAN_COLS = ["rx_node", "claimed_id", "t_start", "t_end"]
POP_NPZ = OUT / "fixed_span_features.npz"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def thr_at_val_fpr(y_val: np.ndarray, s_val: np.ndarray,
                   alpha: float = FPR_TARGET) -> tuple[float, float]:
    """threshold_protocol's protocol: thr = quantile(validation BENIGN scores, 1-alpha), numpy
    'linear' interpolation; predict positive iff score > thr. Returns
    (threshold, realised validation FPR)."""
    ben = np.asarray(s_val, float)[np.asarray(y_val).astype(int) == 0]
    if ben.size == 0:
        return float("nan"), float("nan")
    thr = float(np.quantile(ben, 1.0 - alpha))
    return thr, float(np.mean(ben > thr))


def _auc(y, s, kind):
    y = np.asarray(y).astype(int)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s) if kind == "roc"
                 else average_precision_score(y, s))


def metrics_at(y, s, thr):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    pred = (s > thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    npos = int((y == 1).sum())
    nneg = int((y == 0).sum())
    return {"recall": tp / npos if npos else float("nan"),
            "fpr": fp / nneg if nneg else float("nan"),
            "precision": tp / (tp + fp) if (tp + fp) else float("nan")}


def grouped_boot(y, s, g, thr, n_boot=None, seed=BOOT_SEED) -> dict:
    """Percentile CIs resampling whole transmitters with replacement. The
    operating threshold is held FIXED (it came from the p=0 validation split, not
    from these test rows), so only sampling noise on the test spans is priced in."""
    n_boot = int(N_BOOT if n_boot is None else n_boot)
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    g = np.asarray(g)
    uniq = np.unique(g)
    idx_by_g = {u: np.where(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    acc = {"roc_auc": [], "pr_auc": [], "recall_at_thr": [], "fpr_at_thr": []}
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[u] for u in pick])
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        acc["roc_auc"].append(_auc(yb, sb, "roc"))
        acc["pr_auc"].append(_auc(yb, sb, "pr"))
        m = metrics_at(yb, sb, thr)
        acc["recall_at_thr"].append(m["recall"])
        acc["fpr_at_thr"].append(m["fpr"])
    out = {}
    for k, v in acc.items():
        a = np.asarray([x for x in v if np.isfinite(x)], float)
        out[k] = ({"lo": float(np.percentile(a, 2.5)),
                   "hi": float(np.percentile(a, 97.5)), "n": int(a.size)}
                  if a.size else {"lo": float("nan"), "hi": float("nan"), "n": 0})
    return out


def _mk_data(X, y, attack, groups, aoi, t_end, feat_names, blocks) -> ExperimentData:
    return ExperimentData(
        X=X, y=y, attack_name=attack, groups=groups, aoi=aoi, t_end=t_end,
        feature_names=feat_names, blocks=blocks,
        points=pd.DataFrame(index=range(len(y))), meta=pd.DataFrame())


# --------------------------------------------------------------------------- #
# (A) guard: the default count path must be byte-identical
# --------------------------------------------------------------------------- #
def step_a_guard(features_cfg: dict) -> tuple[dict, pd.DataFrame]:
    from qkv2x.experiments.common import load_experiment_data
    t0 = time.time()
    print("[fixed_span][A] rebuilding dev_grid through the DEFAULT count path "
          "(no new config keys)...", flush=True)
    data = load_experiment_data(RUNS, features_cfg)
    pub = pd.read_parquet(RUNS / "features.parquet")
    names = list(data.feature_names)
    checks = {
        "n_windows_rebuilt": int(len(data)),
        "n_windows_expected": PUBLISHED_N_WINDOWS,
        "n_windows_match": bool(len(data) == PUBLISHED_N_WINDOWS),
        "n_windows_parquet": int(len(pub)),
        "feature_names_match": bool(names == [c for c in pub.columns
                                              if c in set(names)]),
    }
    Xp = pub[names].to_numpy(dtype=np.float64)
    checks["features_bit_identical"] = bool(np.array_equal(data.X, Xp))
    checks["features_max_abs_diff"] = float(np.max(np.abs(data.X - Xp)))
    checks["labels_identical"] = bool(np.array_equal(
        data.y.astype(np.int64), pub["y"].to_numpy().astype(np.int64)))
    checks["attack_name_identical"] = bool(
        list(map(str, data.attack_name)) == list(map(str, pub["attack_name"])))
    checks["group_identical"] = bool(np.array_equal(
        data.groups.astype(np.int64), pub["group"].to_numpy().astype(np.int64)))
    checks["aoi_bit_identical"] = bool(np.array_equal(
        data.aoi, pub["aoi"].to_numpy(dtype=np.float64)))
    checks["t_end_bit_identical"] = bool(np.array_equal(
        data.t_end, pub["t_end"].to_numpy(dtype=np.float64)))
    checks["n_positive"] = int(data.y.sum())

    # span-duration justification, measured on the published count windows
    m = data.meta
    dur = (m["t_end"] - m["t_start"]).to_numpy(dtype=float)
    nm = m["n_msgs"].to_numpy(dtype=float)
    ia = np.divide(dur, np.maximum(nm - 1.0, 1.0))
    checks["count_window_duration_s"] = {
        "median": float(np.median(dur)), "mean": float(np.mean(dur)),
        "q10": float(np.quantile(dur, 0.10)), "q90": float(np.quantile(dur, 0.90)),
    }
    checks["observed_interarrival_s"] = {
        "median": float(np.median(ia)), "mean": float(np.mean(ia)),
        "q90": float(np.quantile(ia, 0.90)),
    }
    checks["count_window_n_msgs_median"] = float(np.median(nm))
    checks["wall_time_s"] = round(time.time() - t0, 1)

    ok = all(checks[k] for k in ("n_windows_match", "features_bit_identical",
                                 "labels_identical", "attack_name_identical",
                                 "group_identical", "aoi_bit_identical",
                                 "t_end_bit_identical"))
    print(f"[fixed_span][A] n={len(data)} (expected {PUBLISHED_N_WINDOWS}) "
          f"bit-identical={checks['features_bit_identical']} "
          f"max|dX|={checks['features_max_abs_diff']:.3g} "
          f"labels={checks['labels_identical']} groups={checks['group_identical']} "
          f"[{checks['wall_time_s']}s]", flush=True)
    print(f"[fixed_span][A] published count-window duration median "
          f"{checks['count_window_duration_s']['median']:.4f} s; observed "
          f"inter-arrival median {checks['observed_interarrival_s']['median']:.4f} s "
          f"-> SPAN_S={SPAN_S}s chosen", flush=True)
    if not ok:
        raise SystemExit("FATAL: the DEFAULT count path is no longer byte-identical "
                         "to runs/dev_grid/features.parquet -- revert the src change.")
    return checks, m


# --------------------------------------------------------------------------- #
# (B)+(C) frozen spans and their per-level features
# --------------------------------------------------------------------------- #
def step_bc_population(features_cfg: dict) -> dict:
    t0 = time.time()
    time_cfg = copy.deepcopy(features_cfg)
    time_cfg["window"]["mode"] = "time"
    time_cfg["window"]["span_s"] = SPAN_S
    time_cfg["window"]["stride_s"] = STRIDE_S
    drop_seed_base = int(features_cfg["splits"]["seed"])

    per_level_X = {p: [] for p in P_GRID}
    per_level_aoi = {p: [] for p in P_GRID}
    per_level_nmsg = {p: [] for p in P_GRID}
    per_level_y = {p: [] for p in P_GRID}
    spans_all, y0, atk0, grp0 = [], [], [], []
    feat_names = blocks = None
    group_offset = 0

    for run_idx, run_dir in enumerate(iter_runs(RUNS)):
        rx = load_rx(run_dir)
        meta = load_meta(run_dir)
        nominal = float(meta.get("bsm_rate_hz", 10.0))
        ds0 = build_dataset(rx, time_cfg, features_cfg["propagation"], nominal)
        spans = ds0.meta[SPAN_COLS].reset_index(drop=True)
        feat_names, blocks = ds0.feature_names, ds0.blocks
        spans_all.append(spans.assign(run_idx=run_idx))
        y0.append(ds0.y)
        atk0.append(ds0.attack_name)
        grp0.append(ds0.groups + group_offset)
        group_offset += int(ds0.groups.max()) + 1 if len(ds0.groups) else 0
        for p in P_GRID:
            ds = ds0 if p == 0.0 else build_dataset(
                rx, features_cfg, features_cfg["propagation"], nominal,
                drop_frac=p, drop_seed=drop_seed_base + run_idx, fixed_spans=spans)
            assert len(ds) == len(spans), (
                f"span population changed at p={p} in {run_dir}: "
                f"{len(ds)} != {len(spans)}")
            per_level_X[p].append(ds.X)
            per_level_aoi[p].append(ds.aoi)
            per_level_nmsg[p].append(ds.meta["n_msgs"].to_numpy())
            per_level_y[p].append(ds.y)
        print(f"[fixed_span][B] {run_dir.parent.name[:28]}/{run_dir.name}: "
              f"{len(spans)} spans  [{time.time()-t0:6.1f}s]", flush=True)
        del rx

    pop = {
        "spans": pd.concat(spans_all, ignore_index=True),
        "y": np.concatenate(y0).astype(int),
        "attack": np.asarray([str(a) for a in np.concatenate(atk0)]),
        "groups": np.concatenate(grp0).astype(np.int64),
        "X": {p: np.vstack(per_level_X[p]) for p in P_GRID},
        "aoi": {p: np.concatenate(per_level_aoi[p]) for p in P_GRID},
        "n_msgs": {p: np.concatenate(per_level_nmsg[p]).astype(int) for p in P_GRID},
        "y_recomputed": {p: np.concatenate(per_level_y[p]).astype(int) for p in P_GRID},
        "feature_names": feat_names,
        "blocks": blocks,
        "wall_time_s": round(time.time() - t0, 1),
    }
    n = len(pop["y"])
    assert all(pop["X"][p].shape[0] == n for p in P_GRID)
    print(f"[fixed_span][C] frozen population: {n} spans, {int(pop['y'].sum())} positive "
          f"({100*pop['y'].mean():.2f}%), {len(np.unique(pop['groups']))} groups "
          f"[{pop['wall_time_s']}s]", flush=True)
    return pop


def _tag(p) -> str:
    return f"p{str(p).replace('.', '')}"


def save_population(pop: dict) -> None:
    """Freeze the population + every level's features so the study can be
    re-analysed (or a quantum arm added) without rebuilding from rx.csv."""
    np.savez(POP_NPZ,
             feature_names=np.asarray(pop["feature_names"]),
             blocks=np.asarray(pop["blocks"]),
             y=pop["y"], groups=pop["groups"], attack=pop["attack"],
             spans=pop["spans"].to_numpy(dtype=np.float64),
             span_cols=np.asarray(list(pop["spans"].columns)),
             p_grid=np.asarray(P_GRID, dtype=float),
             **{f"X_{_tag(p)}": pop["X"][p] for p in P_GRID},
             **{f"aoi_{_tag(p)}": pop["aoi"][p] for p in P_GRID},
             **{f"nmsgs_{_tag(p)}": pop["n_msgs"][p] for p in P_GRID},
             **{f"yrec_{_tag(p)}": pop["y_recomputed"][p] for p in P_GRID})


def load_population() -> dict:
    z = np.load(POP_NPZ, allow_pickle=False)
    cols = [str(c) for c in z["span_cols"]]
    return {
        "spans": pd.DataFrame(z["spans"], columns=cols),
        "y": z["y"].astype(int), "groups": z["groups"].astype(np.int64),
        "attack": np.asarray([str(a) for a in z["attack"]]),
        "feature_names": [str(s) for s in z["feature_names"]],
        "blocks": [str(s) for s in z["blocks"]],
        "X": {p: z[f"X_{_tag(p)}"] for p in P_GRID},
        "aoi": {p: z[f"aoi_{_tag(p)}"] for p in P_GRID},
        "n_msgs": {p: z[f"nmsgs_{_tag(p)}"].astype(int) for p in P_GRID},
        "y_recomputed": {p: z[f"yrec_{_tag(p)}"].astype(int) for p in P_GRID},
        "wall_time_s": 0.0,
    }


def population_census(pop: dict) -> tuple[pd.DataFrame, dict]:
    """Per-level census + the nestedness / label-stability verifications."""
    y0 = pop["y"]
    rows, nested_ok, prev = [], True, None
    label_flips = {}
    for p in P_GRID:
        nm = pop["n_msgs"][p]
        if prev is not None:
            nested_ok &= bool((nm <= prev).all())
        prev = nm
        yr = pop["y_recomputed"][p]
        label_flips[str(p)] = {
            "pos_to_neg": int(((y0 == 1) & (yr == 0)).sum()),
            "neg_to_pos": int(((y0 == 0) & (yr == 1)).sum()),
        }
        rows.append({
            "packet_loss": p, "n_spans": int(len(y0)),
            "n_pos": int(y0.sum()), "n_neg": int((y0 == 0).sum()),
            "mean_msgs_per_span": float(nm.mean()),
            "median_msgs_per_span": float(np.median(nm)),
            "mean_msgs_pos": float(nm[y0 == 1].mean()),
            "mean_msgs_neg": float(nm[y0 == 0].mean()),
            "frac_spans_empty": float((nm == 0).mean()),
            "frac_spans_below_min5": float((nm < 5).mean()),
            "message_retention": float(nm.sum() / pop["n_msgs"][0.0].sum()),
            "label_flips_pos_to_neg": label_flips[str(p)]["pos_to_neg"],
            "label_flips_neg_to_pos": label_flips[str(p)]["neg_to_pos"],
        })
    return pd.DataFrame(rows), {"nested_message_sets": nested_ok,
                                "label_flips": label_flips}


# --------------------------------------------------------------------------- #
# (D) evaluation
# --------------------------------------------------------------------------- #
def evaluate(pop: dict, cfgs: dict, quantum: bool) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    features_cfg = cfgs["features"]
    splits = features_cfg["splits"]
    seed = int(splits["seed"])
    y0, groups, attack = pop["y"], pop["groups"], pop["attack"]
    n = len(y0)

    # ---- ONE split on the frozen population, reused at every loss level ------
    tr_all, va, te = group_split(groups, y0, splits["test_frac"],
                                 splits["val_frac"], seed)
    tr = matched_train_idx(tr_all, _mk_data(pop["X"][0.0], y0, attack, groups,
                                            pop["aoi"][0.0], np.zeros(n),
                                            pop["feature_names"], pop["blocks"]),
                           cfgs)
    va_thr = _cap_train_groupwise(va, groups, VAL_THR_CAP, VAL_CAP_SEED)
    split_info = {
        "n_train_full": int(len(tr_all)), "n_train_matched": int(len(tr)),
        "n_val": int(len(va)), "n_val_used_for_threshold": int(len(va_thr)),
        "n_test": int(len(te)), "n_test_pos": int(y0[te].sum()),
        "n_test_neg": int((y0[te] == 0).sum()),
        "n_test_groups": int(len(np.unique(groups[te]))),
        "disjoint_groups": bool(
            not (set(groups[tr_all]) & set(groups[te]))
            and not (set(groups[va]) & set(groups[te]))),
    }
    print(f"[fixed_span][D] split: train_full={len(tr_all)} matched={len(tr)} "
          f"val={len(va)} (thr subset {len(va_thr)}) test={len(te)} "
          f"pos={int(y0[te].sum())} groups={split_info['n_test_groups']}",
          flush=True)

    models = CLASSICAL + (QUANTUM if quantum else [])
    rows, preds = [], []

    # ---------- arm 1: retrained at each level ------------------------------ #
    n_va = len(va_thr)
    for p in P_GRID:
        Xp, aoip = pop["X"][p], pop["aoi"][p]
        # augment with the p=0 features of the threshold-validation spans so ONE
        # fit yields both the level-p test scores and the p=0-validation scores
        X_aug = np.vstack([Xp, pop["X"][0.0][va_thr]])
        aoi_aug = np.concatenate([aoip, pop["aoi"][0.0][va_thr]])
        y_aug = np.concatenate([y0, y0[va_thr]])
        g_aug = np.concatenate([groups, groups[va_thr]])
        a_aug = np.concatenate([attack, attack[va_thr]])
        t_aug = np.zeros(len(y_aug))
        data = _mk_data(X_aug, y_aug, a_aug, g_aug, aoi_aug, t_aug,
                        pop["feature_names"], pop["blocks"])
        reducer = fit_reducer(features_cfg["reduction"], X_aug, pop["blocks"], tr)
        Xr = reducer.transform(X_aug)
        te_aug = np.concatenate([te, np.arange(n, n + n_va)])
        for model in models:
            t1 = time.time()
            res = run_model(model, data, Xr, {"tr": tr, "va": va, "te": te_aug}, cfgs)
            s_all = np.asarray(res["scores"], float)
            s_te, s_va = s_all[: len(te)], s_all[len(te):]
            rows.append(_row("retrained", model, p, y0, te, groups, s_te,
                             va_thr, s_va, pop))
            preds.append(pd.DataFrame({
                "arm": "retrained", "model": model, "packet_loss": p,
                "span_row": te, "y_true": y0[te], "score": s_te,
                "group": groups[te], "attack_name": attack[te],
                "n_msgs": pop["n_msgs"][p][te]}))
            print(f"[fixed_span][D] retrained p={p:.1f} {model:16s} "
                  f"ROC={rows[-1]['roc_auc']:.4f} PR={rows[-1]['pr_auc']:.4f} "
                  f"rec@1%FPR(p0thr)={rows[-1]['recall_at_1pct_fpr']:.4f} "
                  f"realFPR={rows[-1]['realized_fpr']:.4f} "
                  f"[{time.time()-t1:.1f}s]", flush=True)
        del X_aug, Xr, data

    # ---------- arm 2: p=0 detector deployed unchanged (classical only) ----- #
    reducer0 = fit_reducer(features_cfg["reduction"], pop["X"][0.0],
                           pop["blocks"], tr)
    Xr0 = reducer0.transform(pop["X"][0.0])
    bcfg = cfgs["baselines"]
    fitted = {}
    for model in CLASSICAL:
        npos = int(np.bincount(y0[tr].astype(int), minlength=2).min())
        est = build_baseline(model, bcfg, seed=int(bcfg.get("seed", 0)),
                             folds=max(2, min(int(bcfg.get("cv_folds", 5)), npos)),
                             tune=(npos >= 2))
        est.fit(Xr0[tr], y0[tr])
        fitted[model] = est
        s_va = baseline_scores(est, Xr0[va_thr])
        for p in P_GRID:
            s_te = baseline_scores(est, reducer0.transform(pop["X"][p])[te])
            rows.append(_row("frozen_p0", model, p, y0, te, groups, s_te,
                             va_thr, s_va, pop))
            preds.append(pd.DataFrame({
                "arm": "frozen_p0", "model": model, "packet_loss": p,
                "span_row": te, "y_true": y0[te], "score": s_te,
                "group": groups[te], "attack_name": attack[te],
                "n_msgs": pop["n_msgs"][p][te]}))
            print(f"[fixed_span][D] frozen_p0 p={p:.1f} {model:16s} "
                  f"ROC={rows[-1]['roc_auc']:.4f} PR={rows[-1]['pr_auc']:.4f} "
                  f"rec@1%FPR(p0thr)={rows[-1]['recall_at_1pct_fpr']:.4f} "
                  f"realFPR={rows[-1]['realized_fpr']:.4f}", flush=True)
    return (pd.DataFrame(rows), pd.concat(preds, ignore_index=True),
            {**split_info, "models": models,
             "quantum_arm_included": bool(quantum)})


def _row(arm, model, p, y0, te, groups, s_te, va_thr, s_va, pop) -> dict:
    thr, val_fpr = thr_at_val_fpr(y0[va_thr], s_va)
    m = metrics_at(y0[te], s_te, thr)
    # oracle comparator: threshold re-derived on the LEVEL-p test negatives
    thr_rt, _ = thr_at_val_fpr(y0[te], s_te)
    m_rt = metrics_at(y0[te], s_te, thr_rt)
    ci = grouped_boot(y0[te], s_te, groups[te], thr)
    nm = pop["n_msgs"][p][te]
    return {
        "arm": arm, "model": model, "packet_loss": float(p),
        "n_spans": int(len(te)), "n_pos": int(y0[te].sum()),
        "n_neg": int((y0[te] == 0).sum()),
        "n_groups": int(len(np.unique(groups[te]))),
        "mean_msgs_per_span": float(nm.mean()),
        "frac_spans_empty": float((nm == 0).mean()),
        "roc_auc": _auc(y0[te], s_te, "roc"),
        "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
        "pr_auc": _auc(y0[te], s_te, "pr"),
        "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
        "threshold_p0_val": thr, "val_fpr_at_threshold": val_fpr,
        "recall_at_1pct_fpr": m["recall"],
        "recall_at_1pct_fpr_lo": ci["recall_at_thr"]["lo"],
        "recall_at_1pct_fpr_hi": ci["recall_at_thr"]["hi"],
        "realized_fpr": m["fpr"],
        "realized_fpr_lo": ci["fpr_at_thr"]["lo"],
        "realized_fpr_hi": ci["fpr_at_thr"]["hi"],
        "precision_at_thr": m["precision"],
        "recall_at_1pct_fpr_retuned": m_rt["recall"],
        "threshold_retuned_on_test": thr_rt,
        "n_boot_valid": ci["roc_auc"]["n"],
    }


# --------------------------------------------------------------------------- #
# (E) three-curve comparison
# --------------------------------------------------------------------------- #
def three_curves(res: pd.DataFrame) -> pd.DataFrame:
    pub = pd.read_csv(REPO / "results" / "packetloss_dev" / "packetloss_results.csv")
    failures_path = OUT / "packetloss_failures_variant.csv"
    fv = pd.read_csv(failures_path) if failures_path.exists() else None
    rows = []
    for model in sorted(res.model.unique()):
        for p in P_GRID:
            r = res[(res.arm == "retrained") & (res.model == model)
                    & (res.packet_loss == p)]
            pr = pub[(pub.model == model) & (pub.packet_loss == p)]
            row = {
                "model": model, "packet_loss": p,
                # curve 1: fixed-span (this script) -- population CONSTANT
                "fixedspan_n_test": int(r.n_spans.iloc[0]) if len(r) else np.nan,
                "fixedspan_roc_auc": float(r.roc_auc.iloc[0]) if len(r) else np.nan,
                "fixedspan_pr_auc": float(r.pr_auc.iloc[0]) if len(r) else np.nan,
                "fixedspan_recall_1pct": (float(r.recall_at_1pct_fpr.iloc[0])
                                          if len(r) else np.nan),
                # curve 2: published survivors-only -- population SHRINKS
                "survivors_n_test": int(pr.n_test.iloc[0]) if len(pr) else np.nan,
                "survivors_n_windows": (int(pr.n_windows.iloc[0])
                                        if len(pr) else np.nan),
                "survivors_roc_auc": float(pr.roc_auc.iloc[0]) if len(pr) else np.nan,
                "survivors_pr_auc": float(pr.pr_auc.iloc[0]) if len(pr) else np.nan,
            }
            if fv is not None:
                f = fv[(fv.model == model) & (fv.packet_loss == p)]
                # curve 3: failures-counted (packetloss_retention) -- intersection fleet,
                # absent positives counted as misses
                row["failures_recall_fixedpop"] = (float(f.recall_fixedpop.iloc[0])
                                                   if len(f) else np.nan)
                row["failures_recall_survivors"] = (
                    float(f.recall_survivors_common.iloc[0]) if len(f) else np.nan)
                row["failures_ap_conservative"] = (
                    float(f.ap_conservative_fixedpop.iloc[0]) if len(f) else np.nan)
            rows.append(row)
    df = pd.DataFrame(rows)
    for c in ("fixedspan_roc_auc", "survivors_roc_auc", "fixedspan_pr_auc",
              "survivors_pr_auc"):
        base = df.groupby("model")[c].transform("first")
        df[c.replace("_roc_auc", "_droc").replace("_pr_auc", "_dpr")] = df[c] - base
    return df


# --------------------------------------------------------------------------- #
def main() -> None:
    global N_BOOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-quantum", action="store_true",
                    help="run the classical arms only (states the omission)")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--reuse-population", action="store_true",
                    help="reload the frozen population from fixed_span_features.npz "
                         "instead of rebuilding it from rx.csv (iteration aid; the "
                         "reported run rebuilds and re-verifies from scratch)")
    args = ap.parse_args()
    N_BOOT = int(args.boot)

    t0 = time.time()
    print(f"fixed_span fixed-span packet-loss curve | interpreter={sys.executable}")
    print(f"protocol: span_s={SPAN_S} stride_s={STRIDE_S} p_grid={P_GRID} "
          f"fpr_target={FPR_TARGET} boot={args.boot} (seed {BOOT_SEED})", flush=True)

    features_cfg = load_yaml(REPO / "configs" / "features.yaml")
    cfgs = {"features": features_cfg,
            "quantum": load_yaml(REPO / "configs" / "quantum.yaml"),
            "baselines": load_yaml(REPO / "configs" / "baselines.yaml")}

    guard, _ = step_a_guard(features_cfg)
    if args.reuse_population and POP_NPZ.exists():
        print(f"[fixed_span][B] REUSING frozen population from {POP_NPZ}", flush=True)
        pop = load_population()
    else:
        pop = step_bc_population(features_cfg)
        save_population(pop)
    census, pop_checks = population_census(pop)
    census.to_csv(OUT / "fixed_span_population.csv", index=False)
    print("\n[fixed_span] frozen-population census")
    print(census.to_string(index=False, float_format=lambda v: f"{v:.4g}"), flush=True)

    res, preds, ev = evaluate(pop, cfgs, quantum=not args.no_quantum)
    res.to_csv(OUT / "fixed_span_loss.csv", index=False)
    preds.to_parquet(OUT / "fixed_span_predictions.parquet")
    tc = three_curves(res)
    tc.to_csv(OUT / "fixed_span_three_curves.csv", index=False)

    # ------------------------------ headline --------------------------------
    def cell(arm, model, p, cols=("roc_auc", "pr_auc", "recall_at_1pct_fpr",
                                  "realized_fpr", "mean_msgs_per_span")):
        r = res[(res.arm == arm) & (res.model == model) & (res.packet_loss == p)]
        if not len(r):
            return None
        out = {c: float(r[c].iloc[0]) for c in cols}
        out["roc_auc_ci"] = [float(r.roc_auc_lo.iloc[0]), float(r.roc_auc_hi.iloc[0])]
        out["pr_auc_ci"] = [float(r.pr_auc_lo.iloc[0]), float(r.pr_auc_hi.iloc[0])]
        out["recall_at_1pct_fpr_ci"] = [float(r.recall_at_1pct_fpr_lo.iloc[0]),
                                        float(r.recall_at_1pct_fpr_hi.iloc[0])]
        return out

    models_run = list(res.model.unique())
    monotone = {}
    for arm in res.arm.unique():
        for m in res[res.arm == arm].model.unique():
            s = res[(res.arm == arm) & (res.model == m)].sort_values("packet_loss")
            for met in ("roc_auc", "pr_auc", "recall_at_1pct_fpr"):
                v = s[met].to_numpy()
                monotone[f"{arm}/{m}/{met}"] = {
                    "p0": float(v[0]), "p08": float(v[-1]),
                    "delta_p0_to_p08": float(v[-1] - v[0]),
                    "monotone_nonincreasing": bool(np.all(np.diff(v) <= 1e-12)),
                }
    pub = pd.read_csv(REPO / "results" / "packetloss_dev" / "packetloss_results.csv")
    survivors_delta = {}
    for m in models_run:
        s = pub[pub.model == m].sort_values("packet_loss")
        if len(s) == len(P_GRID):
            survivors_delta[m] = {
                "roc_auc_p0": float(s.roc_auc.iloc[0]),
                "roc_auc_p08": float(s.roc_auc.iloc[-1]),
                "delta_p0_to_p08": float(s.roc_auc.iloc[-1] - s.roc_auc.iloc[0]),
                "n_windows_p0": int(s.n_windows.iloc[0]),
                "n_windows_p08": int(s.n_windows.iloc[-1]),
            }
    # ordering checks at each level (retrained arm)
    ordering = {}
    for p in P_GRID:
        sub = res[(res.arm == "retrained") & (res.packet_loss == p)]
        if len(sub):
            ordering[str(p)] = {
                "best_model_by_roc": str(sub.loc[sub.roc_auc.idxmax(), "model"]),
                "roc_by_model": {str(r.model): float(r.roc_auc)
                                 for r in sub.itertuples()},
            }
    cl_vs_zz = {}
    if "qsvm_zz" in models_run:
        for p in P_GRID:
            a = res[(res.arm == "retrained") & (res.model == "qsvm_crosslayer")
                    & (res.packet_loss == p)]
            b = res[(res.arm == "retrained") & (res.model == "qsvm_zz")
                    & (res.packet_loss == p)]
            if len(a) and len(b):
                cl_vs_zz[str(p)] = {
                    "crosslayer_roc": float(a.roc_auc.iloc[0]),
                    "zz_roc": float(b.roc_auc.iloc[0]),
                    "crosslayer_gt_zz": bool(a.roc_auc.iloc[0] > b.roc_auc.iloc[0]),
                }

    protocol = {
        "src_change": {
            "file": "src/qkv2x/features/dataset.py",
            "additions": ["window.mode = 'count'(default) | 'time'",
                          "window.span_s / window.stride_s (seconds, time mode)",
                          "build_dataset(..., fixed_spans=<rx_node, claimed_id, "
                          "t_start, t_end>) -- evaluates EXACTLY those spans, no "
                          "min_messages filter, no onset filter, order preserved, "
                          "meta['n_msgs'] emitted so thin/empty spans are visible"],
            "default_path": "byte-identical (asserted in step A on all 90,553 "
                            "dev_grid windows against runs/dev_grid/features.parquet)",
        },
        "span_definition": (f"global half-open grid [k*{STRIDE_S}, k*{STRIDE_S}+"
                            f"{SPAN_S}) seconds per (RSU receiver, claimed identity) "
                            f"stream; span_s == stride_s so the timeline is tiled "
                            f"exactly and each message belongs to one span"),
        "span_duration_rationale": (
            "the published count windows are 10 messages at a 10 Hz nominal BSM "
            "rate; measured on runs/dev_grid their duration is median "
            f"{guard['count_window_duration_s']['median']:.4f} s (q10 "
            f"{guard['count_window_duration_s']['q10']:.4f}, q90 "
            f"{guard['count_window_duration_s']['q90']:.4f}) with observed "
            f"within-stream inter-arrival median "
            f"{guard['observed_interarrival_s']['median']:.4f} s, so a 1.0 s span "
            "carries the same nominal 10 messages -- 1.0 s is the count window's "
            "own duration, not an assumption. The q90 tail (long windows) exists "
            "because a count window stretches across reception gaps; a time span "
            "cannot, which is precisely why it is loss-invariant."),
        "population_freezing": (
            "spans, labels, attack names and groups are derived ONCE from the p=0 "
            "build and reused unchanged at every loss level; only the FEATURES are "
            "recomputed. Labels recomputed under loss are reported (label_flips) "
            "but never applied, so n_spans and n_pos are constant by construction."),
        "loss_injection": (
            "build_dataset's own rule, unmodified: after the receiver_set='rsu' "
            "filter, rx = rx[default_rng(drop_seed).random(len(rx)) >= p] with "
            f"drop_seed = {int(features_cfg['splits']['seed'])} + run_idx (the "
            "published packetloss.py seeding). Identical seed and row count at "
            "every level => the retained message sets are NESTED in p "
            f"(verified: {pop_checks['nested_message_sets']})."),
        "feature_recomputation": (
            "message-level features are recomputed from the surviving messages "
            "(so RSSI/kinematic/Sybil statistics degrade too), and window rate "
            "features use the span's KNOWN duration D: rate_ratio = n_msgs/(D*rate), "
            "pdr = n_msgs/(D*rate + 1). Empty spans get an all-zero feature vector "
            "and are retained as detection failures rather than deleted."),
        "split": ("one GroupShuffleSplit on the FROZEN population "
                  f"(test_frac={features_cfg['splits']['test_frac']}, "
                  f"val_frac={features_cfg['splits']['val_frac']}, "
                  f"seed={features_cfg['splits']['seed']}, "
                  f"group_by={features_cfg['splits']['group_by']}), reused at every "
                  "loss level, so train/val/test spans are the SAME objects at every p"),
        "matched_n": (f"all arms train on the quantum max_train="
                      f"{cfgs['quantum'].get('max_train')} group-wise cap "
                      "(matched-N, exactly as the published packetloss curve's "
                      "non-__full rows). A classical@full ceiling arm was NOT run "
                      "(cost); the published curve's __full rows remain the "
                      "scaling context."),
        "arms": {
            "retrained": "detector refit at each level on that level's train spans "
                         "(reducer refit too) -- 'operator retrains for the channel'",
            "frozen_p0": "the p=0 detector and p=0 reducer deployed unchanged, only "
                         "the input features degrade -- 'deployed detector'. "
                         "CLASSICAL ONLY: run_model does not return a fitted "
                         "quantum model, so a frozen quantum arm is NOT computed "
                         "here (IRRECOVERABLE from the current API without a "
                         "further src change beyond dataset.py).",
        },
        "operating_point": (
            "threshold_protocol's protocol: thr = quantile(p=0 VALIDATION benign scores, 0.99) "
            "(numpy linear interpolation), positive iff score > thr. Derived once "
            "per (arm, model, level) on the p=0 validation spans and APPLIED "
            "UNCHANGED to the level-p test spans, so the realised test FPR is free "
            f"to drift. Validation spans are group-capped to {VAL_THR_CAP} rows "
            f"(seed {VAL_CAP_SEED}) for memory. A retuned-on-test threshold is "
            "reported as recall_at_1pct_fpr_retuned (upper bound, never headline)."),
        "ci_method": (f"grouped percentile bootstrap, B={args.boot}, seed={BOOT_SEED}: "
                      "test transmitters resampled with replacement; the operating "
                      "threshold is held fixed (it is not a function of the test "
                      "rows); CI = [2.5, 97.5] percentiles"),
        "comparators": {
            "survivors_only": "results/packetloss_dev/packetloss_results.csv "
                              "(published; population shrinks with p)",
            "failures_counted": "results/studies/packetloss_failures_variant.csv "
                                "(packetloss_retention; intersection fleet, absent positives "
                                "counted as misses)",
        },
        "interpreter": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__,
    }

    headline = {
        "guard_default_path_byte_identical": {
            "n_windows": guard["n_windows_rebuilt"],
            "expected": PUBLISHED_N_WINDOWS,
            "features_bit_identical": guard["features_bit_identical"],
            "max_abs_feature_diff": guard["features_max_abs_diff"],
            "labels_identical": guard["labels_identical"],
            "attack_name_identical": guard["attack_name_identical"],
            "group_identical": guard["group_identical"],
            "aoi_bit_identical": guard["aoi_bit_identical"],
            "t_end_bit_identical": guard["t_end_bit_identical"],
        },
        "span_choice": {
            "span_s": SPAN_S, "stride_s": STRIDE_S,
            "published_count_window_duration_median_s":
                guard["count_window_duration_s"]["median"],
            "observed_interarrival_median_s":
                guard["observed_interarrival_s"]["median"],
            "published_count_window_n_msgs_median":
                guard["count_window_n_msgs_median"],
        },
        "population": {
            "n_spans_all_levels": int(len(pop["y"])),
            "n_positive": int(pop["y"].sum()),
            "prevalence": float(pop["y"].mean()),
            "n_groups": int(len(np.unique(pop["groups"]))),
            "population_constant_across_p": True,
            "per_level": {str(r.packet_loss): {
                "mean_msgs_per_span": r.mean_msgs_per_span,
                "frac_spans_empty": r.frac_spans_empty,
                "frac_spans_below_min5": r.frac_spans_below_min5,
                "message_retention": r.message_retention,
                "label_flips_pos_to_neg": int(r.label_flips_pos_to_neg),
                "label_flips_neg_to_pos": int(r.label_flips_neg_to_pos),
            } for r in census.itertuples()},
            "published_survivors_n_windows": {
                str(p): int(pub[(pub.model == "rbf_svm")
                                & (pub.packet_loss == p)].n_windows.iloc[0])
                for p in P_GRID},
        },
        "fixed_span_curve": {arm: {m: {str(p): cell(arm, m, p) for p in P_GRID}
                                   for m in res[res.arm == arm].model.unique()}
                             for arm in res.arm.unique()},
        "monotonicity_and_deltas": monotone,
        "published_survivors_only_deltas": survivors_delta,
        "model_ordering_by_level_retrained": ordering,
        "crosslayer_vs_zz_by_level": cl_vs_zz,
        "split": ev,
    }

    summary = {
        "item": "fixed_span_fixed_span_packet_loss",
        "script": "scripts/studies/fixed_span_loss.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_s": round(time.time() - t0, 1),
        "protocol": protocol,
        "verified_checks": {**{f"guard_{k}": v for k, v in guard.items()},
                            **pop_checks},
        "headline": headline,
        "artifacts": ["fixed_span_loss.csv", "fixed_span_three_curves.csv",
                      "fixed_span_population.csv", "fixed_span_predictions.parquet",
                      "fixed_span_features.npz", "fixed_span_summary.json",
                      "fixed_span_run.log"],
    }
    (OUT / "fixed_span_summary.json").write_text(json.dumps(summary, indent=2,
                                                      default=float))

    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print("fixed_span -- fixed-span (time-based) packet-loss curve, population held constant")
    print("=" * 110)
    print(res[["arm", "model", "packet_loss", "n_spans", "n_pos",
               "mean_msgs_per_span", "frac_spans_empty", "roc_auc", "pr_auc",
               "recall_at_1pct_fpr", "realized_fpr"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nThree curves (retrained arm vs published survivors-only vs packetloss_retention failures):")
    print(tc.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nwrote {OUT/'fixed_span_loss.csv'}")
    print(f"wrote {OUT/'fixed_span_three_curves.csv'}")
    print(f"wrote {OUT/'fixed_span_population.csv'}")
    print(f"wrote {OUT/'fixed_span_predictions.parquet'}")
    print(f"wrote {OUT/'fixed_span_features.npz'}")
    print(f"wrote {OUT/'fixed_span_summary.json'}")
    print(f"done in {summary['wall_time_s']}s")


if __name__ == "__main__":
    main()
