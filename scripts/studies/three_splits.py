#!/usr/bin/env python3
"""three_splits -- three-split (multi-seed) reporting.

Reports all three leakage-safe split seeds and the split-level metrics, then the
mean and confidence interval across splits.

This script is AGGREGATION ONLY -- no model is retrained, no result is recomputed.
Existing per-seed (per-split) metrics are read from three shipped result files:

  dev_grid  results/dev_grid_multiseed/binary_results.csv
             (qsvm_crosslayer, qsvm_zz, rbf_svm, random_forest, xgboost;
              split seeds 7/17/27; the shipped summary multiseed.csv is
              cross-checked against our recomputation)
  nextgen  results/veremi_nextgen/veremi_nextgen_binary_results.csv,
             variant == "full" ONLY (qsvm_crosslayer, qsvm_zz, rbf_svm,
             random_forest; split seeds 7/17/27 -- these are exactly the
             per-seed rows behind the shipped veremi_nextgen_multiseed.csv,
             which is cross-checked)
  nextgen  results/nextgen_multiseed/per_seed.csv
             (knn_bagging, mlp, xgboost; split seeds 7/17/27; shipped
              seed_summary.csv cross-checked). Same official NextGen split
             assembly (n_test=2399, n_pos=517) as the veremi_nextgen "full"
             variant rows, so the two sources are directly comparable.

For every dataset x model x metric (roc_auc, pr_auc, f1, mcc) we report:
  * the seed list and each per-seed value EXPLICITLY (no pooling),
  * mean, sd (ddof=1), sem,
  * a two-sided 95% CI from the Student t distribution with n-1 dof.
    With n = 3 seeds, t_{0.975, 2} = 4.30265 -- these intervals are WIDE,
    as expected with three splits.

Models with fewer than 2 seeds in a source (e.g. the single-seed "netblind" /
"scale" / "sybil_receiver" variants, or knn/mlp/rules/lstm_seq inside the
veremi_nextgen file) are EXCLUDED from the CI table -- a CI cannot be computed
from one split; the exclusions are listed in the JSON.

Determinism: this script is pure deterministic aggregation (no RNG anywhere);
numpy/random seeds are nevertheless set and recorded in "protocol".

Usage:
    python scripts/studies/three_splits.py

Outputs (results/studies/):
    three_splits.csv  tidy table: dataset, source, model, metric, per-seed
                        values, mean, sd, sem, t_crit, ci95_lo/hi
    three_splits_summary.json  protocol + headline numbers + summary cross-checks
    three_splits_run.log  (via tee in the launch command)
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RESULTS = REPO / "results"
OUT = RESULTS / "studies"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 7  # no stochastic computation below; set + recorded for the record
random.seed(SEED)
np.random.seed(SEED)

METRICS = ["roc_auc", "pr_auc", "f1", "mcc"]
EXPECTED_SEEDS = [7, 17, 27]
CONF = 0.95

HEADLINE_MODELS = ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest", "xgboost"]

# --------------------------------------------------------------------------- #
# Sources (read-only)
# --------------------------------------------------------------------------- #
DEV_GRID_PER_SEED = RESULTS / "dev_grid_multiseed" / "binary_results.csv"
DEV_GRID_SUMMARY = RESULTS / "dev_grid_multiseed" / "multiseed.csv"
NEXTGEN_BASE_PER_SEED = RESULTS / "nextgen_multiseed" / "per_seed.csv"
NEXTGEN_BASE_SUMMARY = RESULTS / "nextgen_multiseed" / "seed_summary.csv"
NEXTGEN_FULL_PER_SEED = RESULTS / "veremi_nextgen" / "veremi_nextgen_binary_results.csv"
NEXTGEN_FULL_SUMMARY = RESULTS / "veremi_nextgen" / "veremi_nextgen_multiseed.csv"


def t_ci(vals: np.ndarray, conf: float = CONF):
    """Mean +/- t_{1-(1-conf)/2, n-1} * sd/sqrt(n). Requires n >= 2."""
    n = len(vals)
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    sem = sd / np.sqrt(n)
    tcrit = float(stats.t.ppf(1.0 - (1.0 - conf) / 2.0, df=n - 1))
    half = tcrit * sem
    return mean, sd, float(sem), tcrit, mean - half, mean + half


def aggregate(df: pd.DataFrame, dataset: str, source: str) -> tuple[list[dict], list[dict]]:
    """Per model x metric rows (n>=2 seeds) + list of excluded (model, n_seeds)."""
    rows, excluded = [], []
    for model, g in df.groupby("model", sort=True):
        g = g.sort_values("seed")
        seeds = [int(s) for s in g["seed"].tolist()]
        if len(seeds) != len(set(seeds)):
            raise RuntimeError(f"{dataset}/{source}/{model}: duplicate seeds {seeds}")
        if len(seeds) < 2:
            excluded.append({"model": model, "seeds": seeds,
                             "reason": "only 1 split seed on record; CI needs >= 2"})
            continue
        for metric in METRICS:
            vals = g[metric].to_numpy(dtype=float)
            mean, sd, sem, tcrit, lo, hi = t_ci(vals)
            row = {
                "dataset": dataset,
                "source": source,
                "model": model,
                "metric": metric,
                "n_seeds": len(seeds),
                "seeds": ";".join(str(s) for s in seeds),
            }
            for s, v in zip(seeds, vals):
                row[f"value_seed{s}"] = float(v)
            row.update({
                "mean": mean, "sd": sd, "sem": sem,
                "t_crit_df_nminus1": tcrit,
                "ci95_lo": lo, "ci95_hi": hi, "ci95_halfwidth": tcrit * sem,
            })
            rows.append(row)
    return rows, excluded


def crosscheck(computed: list[dict], summary_df: pd.DataFrame, dataset: str,
               source: str, mean_fmt: str, std_fmt: str) -> dict:
    """Compare our recomputed mean/sd against a shipped summary CSV."""
    diffs = []
    for row in computed:
        m, metric = row["model"], row["metric"]
        s = summary_df[summary_df["model"] == m]
        mean_col, std_col = mean_fmt.format(metric), std_fmt.format(metric)
        if len(s) != 1 or mean_col not in s.columns:
            continue
        d_mean = abs(float(s[mean_col].iloc[0]) - row["mean"])
        d_std = (abs(float(s[std_col].iloc[0]) - row["sd"])
                 if std_col in s.columns else None)
        diffs.append({"model": m, "metric": metric,
                      "abs_diff_mean": d_mean, "abs_diff_std": d_std})
    max_mean = max((d["abs_diff_mean"] for d in diffs), default=None)
    max_std = max((d["abs_diff_std"] for d in diffs
                   if d["abs_diff_std"] is not None), default=None)
    return {"dataset": dataset, "source": source, "n_compared": len(diffs),
            "max_abs_diff_mean": max_mean, "max_abs_diff_std": max_std,
            "agrees_1e-9": bool(max_mean is not None and max_mean < 1e-9
                                and (max_std is None or max_std < 1e-9))}


def main() -> None:
    t0 = time.time()
    all_rows: list[dict] = []
    excluded_all: list[dict] = []
    checks: list[dict] = []

    # ---- dev_grid ---------------------------------------------------------- #
    dg = pd.read_csv(DEV_GRID_PER_SEED)
    rows, excl = aggregate(dg, "dev_grid", DEV_GRID_PER_SEED.relative_to(REPO).as_posix())
    all_rows += rows
    excluded_all += [{"dataset": "dev_grid", **e} for e in excl]
    checks.append(crosscheck(rows, pd.read_csv(DEV_GRID_SUMMARY), "dev_grid",
                             DEV_GRID_SUMMARY.relative_to(REPO).as_posix(),
                             "{}_mean", "{}_std"))

    # ---- nextgen: qsvm/rbf/rf from the FULL variant of the main run -------- #
    ngf = pd.read_csv(NEXTGEN_FULL_PER_SEED)
    ngf_full = ngf[ngf["variant"] == "full"].copy()
    non_full_variants = sorted(ngf.loc[ngf["variant"] != "full", "variant"].unique())
    rows, excl = aggregate(
        ngf_full[["model", "seed"] + METRICS],
        "nextgen",
        NEXTGEN_FULL_PER_SEED.relative_to(REPO).as_posix() + "[variant==full]")
    all_rows += rows
    excluded_all += [{"dataset": "nextgen", **e} for e in excl]
    checks.append(crosscheck(rows, pd.read_csv(NEXTGEN_FULL_SUMMARY), "nextgen",
                             NEXTGEN_FULL_SUMMARY.relative_to(REPO).as_posix(),
                             "{}_mean", "{}_std"))

    # ---- nextgen: knn/mlp/xgboost multi-seed baseline run ------------------ #
    ngb = pd.read_csv(NEXTGEN_BASE_PER_SEED)
    rows, excl = aggregate(ngb, "nextgen",
                           NEXTGEN_BASE_PER_SEED.relative_to(REPO).as_posix())
    all_rows += rows
    excluded_all += [{"dataset": "nextgen", **e} for e in excl]
    checks.append(crosscheck(rows, pd.read_csv(NEXTGEN_BASE_SUMMARY), "nextgen",
                             NEXTGEN_BASE_SUMMARY.relative_to(REPO).as_posix(),
                             "{}_mean", "{}_std"))

    tidy = pd.DataFrame(all_rows)
    # stable column order (seed-value columns in seed order)
    lead = ["dataset", "source", "model", "metric", "n_seeds", "seeds"]
    seed_cols = [f"value_seed{s}" for s in EXPECTED_SEEDS if f"value_seed{s}" in tidy.columns]
    tail = ["mean", "sd", "sem", "t_crit_df_nminus1", "ci95_lo", "ci95_hi", "ci95_halfwidth"]
    tidy = tidy[lead + seed_cols + tail]
    csv_path = OUT / "three_splits.csv"
    tidy.to_csv(csv_path, index=False)

    # ---- seed confirmation ------------------------------------------------- #
    def seeds_of(df, label):
        per_model = {m: sorted(int(s) for s in g["seed"]) for m, g in df.groupby("model")}
        all_expected = all(v == EXPECTED_SEEDS for v in per_model.values())
        return {"per_model": per_model, "all_models_have_seeds_7_17_27": bool(all_expected)}

    seed_confirmation = {
        "dev_grid": seeds_of(dg, "dev_grid"),
        "nextgen_full_variant": seeds_of(
            ngf_full[ngf_full["model"].isin(
                ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest"])], "ngf"),
        "nextgen_baselines": seeds_of(ngb, "ngb"),
    }

    # ---- headline: mean +/- CI for the requested models -------------------- #
    def head_entry(dataset, model):
        sub = tidy[(tidy["dataset"] == dataset) & (tidy["model"] == model)]
        if sub.empty:
            return "IRRECOVERABLE: no per-seed rows for this model on this dataset"
        out = {"seeds": sub.iloc[0]["seeds"], "n_seeds": int(sub.iloc[0]["n_seeds"]),
               "source": sub.iloc[0]["source"]}
        for metric in ["roc_auc", "pr_auc"]:
            r = sub[sub["metric"] == metric].iloc[0]
            out[metric] = {
                "per_seed": {c.replace("value_", ""): round(float(r[c]), 6)
                             for c in seed_cols if pd.notna(r.get(c))},
                "mean": round(float(r["mean"]), 6),
                "sd": round(float(r["sd"]), 6),
                "ci95": [round(float(r["ci95_lo"]), 6), round(float(r["ci95_hi"]), 6)],
                "ci95_halfwidth": round(float(r["ci95_halfwidth"]), 6),
            }
        return out

    headline = {}
    for ds in ["dev_grid", "nextgen"]:
        headline[ds] = {}
        for m in HEADLINE_MODELS:
            headline[ds][m] = head_entry(ds, m)

    protocol = {
        "item": "three_splits",
        "seed_set_for_record": SEED,
        "stochastic_computation": "none -- deterministic aggregation of shipped per-seed CSVs",
        "sources": {
            "dev_grid_per_seed": DEV_GRID_PER_SEED.relative_to(REPO).as_posix(),
            "dev_grid_shipped_summary": DEV_GRID_SUMMARY.relative_to(REPO).as_posix(),
            "nextgen_full_per_seed": NEXTGEN_FULL_PER_SEED.relative_to(REPO).as_posix()
            + " (variant=='full' only; other variants excluded: "
            + ", ".join(non_full_variants) + ")",
            "nextgen_full_shipped_summary": NEXTGEN_FULL_SUMMARY.relative_to(REPO).as_posix(),
            "nextgen_baselines_per_seed": NEXTGEN_BASE_PER_SEED.relative_to(REPO).as_posix(),
            "nextgen_baselines_shipped_summary": NEXTGEN_BASE_SUMMARY.relative_to(REPO).as_posix(),
        },
        "nextgen_two_source_note": (
            "NextGen per-seed rows live in two shipped files produced by two runs on the SAME "
            "official-split assembly (n_test=2399, n_pos=517 in both): the main experiment run "
            "(qsvm_crosslayer/qsvm_zz/rbf_svm/random_forest, variant 'full') and the A4 "
            "multi-seed baseline run (knn_bagging/mlp/xgboost). Both are aggregated per source; "
            "no rows are pooled across sources for any single model."),
        "metrics": METRICS,
        "split_seeds_expected": EXPECTED_SEEDS,
        "ci_method": ("two-sided 95% Student t CI with n-1 dof on the per-seed values: "
                      "mean +/- t_{0.975,n-1} * sd/sqrt(n), sd with ddof=1. "
                      "n=3 seeds -> t_{0.975,2}=4.30265; intervals are WIDE by construction "
                      "with three splits."),
        "no_pooling": "each (dataset, source, model, metric) aggregated over its own 3 seeds only",
        "exclusions": excluded_all,
        "libraries": {"numpy": np.__version__, "pandas": pd.__version__,
                      "scipy": stats.__name__ and __import__("scipy").__version__},
    }

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_s": round(time.time() - t0, 3),
        "protocol": protocol,
        "seed_confirmation": seed_confirmation,
        "summary_crosschecks": checks,
        "n_tidy_rows": int(len(tidy)),
        "headline": headline,
    }
    json_path = OUT / "three_splits_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    print(f"[three_splits] tidy rows: {len(tidy)} -> {csv_path.relative_to(REPO)}")
    print(f"[three_splits] summary   -> {json_path.relative_to(REPO)}")
    print("[three_splits] seed confirmation:",
          {k: v["all_models_have_seeds_7_17_27"] for k, v in seed_confirmation.items()})
    for c in checks:
        print(f"[three_splits] crosscheck {c['source']}: n={c['n_compared']} "
              f"max|dmean|={c['max_abs_diff_mean']:.2e} agrees<1e-9={c['agrees_1e-9']}")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        show = tidy[tidy["metric"].isin(["roc_auc", "pr_auc"])]
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
