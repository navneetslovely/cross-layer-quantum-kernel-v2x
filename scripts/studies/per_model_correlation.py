#!/usr/bin/env python3
"""per_model -- per-model frontier correlations.

Best-over-models detectability creates selection bias. This script reports the
frontier correlation for every model separately, plus a pre-specified reference
model, and keeps the best-over-models result only as an exploratory upper bound.

What the PUBLISHED pipeline did (verified against the shipped code):
  * scripts/controllability.py::_Y_dev_grid takes the per-attack MAX of
    roc_auc/pr_auc over ALL models in results/crossattack_dev/
    crossattack_results.csv (5 matched-N models + 3 "__full" classical models).
  * scripts/controllability.py::_Y_nextgen takes the per-attack MAX over ALL
    models in results/veremi_nextgen/veremi_nextgen_per_attack.csv restricted
    to variant == "full" (9 models).
  * The published headline (results/frontier/controllability_validation.json)
    is Spearman(D_full, Y_rocauc): pooled rho=0.820, dev_grid 0.767,
    nextgen 0.886 -- an ENVELOPE (best-over-models) correlation, i.e. the
    predictand itself was model-selected per attack.

This script removes the selection: for EVERY model with per-attack results,
per dataset, per divergence metric (energy and mmd_u from the divergence
re-estimation, scope=full, NEVER combined), it correlates D_full with that
single model's per-attack ROC-AUC (and PR-AUC):
  * Spearman rho + Kendall tau (scipy, two-sided p recorded for reference);
  * EXACT one-sided permutation p in the PRE-REGISTERED positive direction
    (higher divergence -> higher detectability), a faithful port of
    scripts/meta_stats.py::exact_one_sided_p (rankdata average-tie
    ranks, the 1 - 6*sum d^2/(n(n^2-1)) statistic, count of permuted
    rho >= observed - 1e-12 over ALL n! permutations). The port is
    vectorised for speed and cross-checked against the verbatim original.
  * rbf_svm is marked as the PRE-SPECIFIED reference model (an untuned,
    classical, always-present baseline chosen before looking at results;
    no validation-selected ensemble exists in the published artifacts, so
    the reference-model option is used).
  * best-over-models is recomputed on the SAME divergence D values and appended as
    the final row per (dataset, metric), labelled exploratory_upper_bound=True.
  * SPREAD across individual models (min/median/max rho, fraction positive,
    fraction significant at 0.05) is reported to show whether
    the frontier ordering survives without envelope selection.

DEPENDS ON: results/studies/divergence.csv (fails hard if absent).
 Fully deterministic: the permutation test enumerates all n!
permutations (no sampling, no RNG).

Usage:
    python \
        scripts/studies/per_model_correlation.py \
        2>&1 | tee results/studies/per_model_run.log

Outputs (results/studies/):
    per_model_rho.csv  dataset, metric, model, n_attacks, spearman,
                            kendall, perm_p (ROC target) + *_pr twins,
                            is_reference, exploratory_upper_bound
    per_model_summary.json  protocol + coverage + spreads + verdicts
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RESULTS = REPO / "results"
OUT = RESULTS / "studies"
OUT.mkdir(parents=True, exist_ok=True)

divergence_CSV = OUT / "divergence.csv"
CROSSATTACK_CSV = RESULTS / "crossattack_dev" / "crossattack_results.csv"
NEXTGEN_CSV = RESULTS / "veremi_nextgen" / "veremi_nextgen_per_attack.csv"
PUBLISHED_VALIDATION = RESULTS / "frontier" / "controllability_validation.json"

METRICS = ("energy", "mmd_u")
SCOPE = "full"
REFERENCE_MODEL = "rbf_svm"
ENVELOPE_LABEL = "best_over_models"
EPS = 1e-12  # identical to meta_stats.exact_one_sided_p
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Exact one-sided permutation p -- verbatim port + vectorised twin
# --------------------------------------------------------------------------- #
def _spearman_from_ranks(rx: np.ndarray, ry: np.ndarray) -> float:
    """Verbatim from scripts/meta_stats.py."""
    n = len(rx)
    d2 = float(np.sum((rx - ry) ** 2))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def exact_one_sided_p_verbatim(x: np.ndarray, y: np.ndarray,
                               positive: bool) -> tuple[float, float]:
    """VERBATIM port of scripts/meta_stats.py::exact_one_sided_p."""
    rx = rankdata(x)
    ry = rankdata(y)
    obs = _spearman_from_ranks(rx, ry)
    n = len(rx)
    cnt = tot = 0
    eps = EPS
    for perm in itertools.permutations(range(n)):
        rho = _spearman_from_ranks(rx, ry[list(perm)])
        tot += 1
        if (positive and rho >= obs - eps) or (not positive and rho <= obs + eps):
            cnt += 1
    return obs, cnt / tot


_PERM_CACHE: dict[int, np.ndarray] = {}


def _all_perms(n: int) -> np.ndarray:
    if n not in _PERM_CACHE:
        _PERM_CACHE[n] = np.array(list(itertools.permutations(range(n))),
                                  dtype=np.int64)
    return _PERM_CACHE[n]


def exact_one_sided_p_fast(x: np.ndarray, y: np.ndarray,
                           positive: bool = True) -> tuple[float, float]:
    """Vectorised equivalent of exact_one_sided_p_verbatim (same ranks, same
    statistic, same eps tolerance, same exhaustive n! enumeration)."""
    rx = rankdata(x)
    ry = rankdata(y)
    n = len(rx)
    obs = _spearman_from_ranks(rx, ry)
    perms = _all_perms(n)
    ry_perm = ry[perms]                      # (n!, n)
    d2 = ((rx[None, :] - ry_perm) ** 2).sum(axis=1)
    rho_all = 1.0 - 6.0 * d2 / (n * (n * n - 1))
    if positive:
        cnt = int(np.sum(rho_all >= obs - EPS))
    else:
        cnt = int(np.sum(rho_all <= obs + EPS))
    return obs, cnt / len(rho_all)


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
def load_base_D() -> pd.DataFrame:
    """dataset x attack x metric -> D (scope=full only). Fails hard if absent."""
    if not divergence_CSV.exists():
        raise SystemExit(f"MISSING DEPENDENCY: {divergence_CSV} -- run "
                         "scripts/studies/divergence.py first.")
    d = pd.read_csv(divergence_CSV)
    d = d[(d["scope"] == SCOPE) & (d["metric"].isin(METRICS))]
    return d[["dataset", "attack", "metric", "D"]].reset_index(drop=True)


def load_per_model_auc() -> pd.DataFrame:
    """Long table: dataset, model, attack, roc_auc, pr_auc, protocol tag."""
    rows = []
    # dev_grid: held-out-attack protocol (each row = model trained WITHOUT that
    # attack, scored on it). Mirrors a1._Y_dev_grid's source, un-maxed.
    dg = pd.read_csv(CROSSATTACK_CSV)
    for _, r in dg.iterrows():
        rows.append({"dataset": "dev_grid", "model": r["model"],
                     "attack": r["holdout_attack"],
                     "roc_auc": float(r["roc_auc"]),
                     "pr_auc": float(r["pr_auc"]),
                     "protocol": "held-out-attack (crossattack_dev)"})
    # nextgen: per-attack evaluation of the standard-split models,
    # variant == "full" exactly as a1._Y_nextgen filters.
    ng = pd.read_csv(NEXTGEN_CSV)
    if "variant" in ng.columns and (ng["variant"] == "full").any():
        ng = ng[ng["variant"] == "full"]
    for _, r in ng.iterrows():
        rows.append({"dataset": "nextgen", "model": r["model"],
                     "attack": r["attack"],
                     "roc_auc": float(r["roc_auc"]),
                     "pr_auc": float(r["pr_auc"]),
                     "protocol": "per-attack eval, variant=full (veremi_nextgen)"})
    return pd.DataFrame(rows)


def published_envelope() -> dict:
    """Published best-over-models numbers, quoted for context only."""
    if not PUBLISHED_VALIDATION.exists():
        return {"note": "IRRECOVERABLE", "reason": f"{PUBLISHED_VALIDATION} missing"}
    v = json.loads(PUBLISHED_VALIDATION.read_text())
    d = v.get("screens", {}).get("D_full", {})
    return {
        "source": str(PUBLISHED_VALIDATION.relative_to(REPO)),
        "pooled_spearman_rho": d.get("pooled", {}).get("spearman_rho"),
        "dev_grid_spearman_rho": d.get("per_dataset", {}).get("dev_grid", {}).get("spearman_rho"),
        "nextgen_spearman_rho": d.get("per_dataset", {}).get("nextgen", {}).get("spearman_rho"),
        "note": ("published D_full is the a1 energy-only estimator on "
                 "self-referentially standardised features; per_model correlates the "
                 "RE-ESTIMATED divergence D (grouped cross-fit) instead, so envelope rows "
                 "here need not equal the published rho exactly."),
    }


# --------------------------------------------------------------------------- #
def correlate(D: np.ndarray, y: np.ndarray) -> dict:
    """Spearman + Kendall + exact one-sided (positive) permutation p."""
    rho, p2 = spearmanr(D, y)
    tau, taup = kendalltau(D, y)
    obs, perm_p = exact_one_sided_p_fast(D, y, positive=True)
    # obs uses the simple-formula statistic (exact match to rho when no ties)
    return {"spearman": float(rho), "spearman_p_two_sided": float(p2),
            "kendall": float(tau), "kendall_p_two_sided": float(taup),
            "perm_stat": float(obs), "perm_p": float(perm_p)}


def main() -> None:
    t0 = time.time()
    divergence = load_base_D()
    auc = load_per_model_auc()
    pub = published_envelope()

    # ---- sanity: rank agreement between the two divergence metrics ----------
    # (rank correlations depend only on the ORDER of D, so if energy and mmd_u
    #  order the attacks identically within a dataset, every per-model rho is
    #  numerically identical across the two metric blocks -- a data fact, not
    #  a bug; report it so the duplicate-looking rows are explained.)
    metric_rank_agreement = {}
    for ds, sub in divergence.groupby("dataset"):
        piv = sub.pivot(index="attack", columns="metric", values="D")
        if all(m in piv.columns for m in METRICS):
            rho_m = float(spearmanr(piv["energy"], piv["mmd_u"])[0])
            metric_rank_agreement[ds] = {
                "spearman_D_energy_vs_D_mmd_u": rho_m,
                "orderings_identical": bool(abs(rho_m - 1.0) < 1e-12),
            }

    # ---- coverage bookkeeping (models differ per dataset -- report honestly) --
    coverage = {}
    for ds, sub in auc.groupby("dataset"):
        divergence_attacks = sorted(divergence[divergence.dataset == ds]["attack"].unique())
        coverage[ds] = {
            "models": sorted(sub["model"].unique()),
            "n_models": int(sub["model"].nunique()),
            "divergence_attacks": divergence_attacks,
            "n_base_attacks": len(divergence_attacks),
            "auc_attacks": sorted(sub["attack"].unique()),
            "protocol": sub["protocol"].iloc[0],
        }
    only_dev = sorted(set(coverage.get("dev_grid", {}).get("models", []))
                      - set(coverage.get("nextgen", {}).get("models", [])))
    only_ng = sorted(set(coverage.get("nextgen", {}).get("models", []))
                     - set(coverage.get("dev_grid", {}).get("models", [])))
    shared = sorted(set(coverage.get("dev_grid", {}).get("models", []))
                    & set(coverage.get("nextgen", {}).get("models", [])))

    # ---- envelope predictand recomputed from the SAME per-model table --------
    env = (auc.groupby(["dataset", "attack"])
              .agg(roc_auc=("roc_auc", "max"), pr_auc=("pr_auc", "max"))
              .reset_index())
    env["model"] = ENVELOPE_LABEL
    env["protocol"] = "max over all listed models (exploratory upper bound)"
    auc_plus = pd.concat([auc, env[auc.columns.tolist()]], ignore_index=True)

    # ---- port faithfulness check: vectorised == verbatim on real cases -------
    port_checks = []
    for ds in sorted(auc["dataset"].unique()):
        sub_auc = auc_plus[(auc_plus.dataset == ds)
                           & (auc_plus.model == REFERENCE_MODEL)]
        sub_D = divergence[(divergence.dataset == ds) & (divergence.metric == "energy")]
        m = sub_D.merge(sub_auc, on=["dataset", "attack"], how="inner")
        x = m["D"].to_numpy()
        y = m["roc_auc"].to_numpy()
        o_v, p_v = exact_one_sided_p_verbatim(x, y, positive=True)
        o_f, p_f = exact_one_sided_p_fast(x, y, positive=True)
        ok = (abs(o_v - o_f) < 1e-12) and (abs(p_v - p_f) < 1e-15)
        port_checks.append({"dataset": ds, "model": REFERENCE_MODEL, "n": len(m),
                            "verbatim": {"stat": o_v, "p": p_v},
                            "vectorised": {"stat": o_f, "p": p_f}, "match": ok})
        if not ok:
            raise SystemExit(f"PORT CHECK FAILED on {ds}: {port_checks[-1]}")
    print("port check (vectorised == verbatim exact_one_sided_p): "
          + "; ".join(f"{c['dataset']} n={c['n']} p={c['verbatim']['p']:.6f} OK"
                      for c in port_checks), flush=True)

    # ---- the per-model correlation table -------------------------------------
    rows = []
    dropped = []
    for ds in sorted(auc_plus["dataset"].unique()):
        for metric in METRICS:
            dsub = divergence[(divergence.dataset == ds) & (divergence.metric == metric)]
            models = sorted(auc_plus[auc_plus.dataset == ds]["model"].unique())
            # envelope last, reference flagged
            models = ([m for m in models if m != ENVELOPE_LABEL]
                      + [m for m in models if m == ENVELOPE_LABEL])
            for model in models:
                msub = auc_plus[(auc_plus.dataset == ds)
                                & (auc_plus.model == model)]
                merged = dsub.merge(msub, on=["dataset", "attack"], how="inner")
                n = len(merged)
                lost = sorted(set(dsub["attack"]) - set(merged["attack"]))
                if lost:
                    dropped.append({"dataset": ds, "metric": metric,
                                    "model": model, "attacks_without_auc": lost})
                if n < 4:
                    dropped.append({"dataset": ds, "metric": metric,
                                    "model": model,
                                    "reason": f"only {n} matched attacks (<4)"})
                    continue
                D = merged["D"].to_numpy()
                roc = correlate(D, merged["roc_auc"].to_numpy())
                pr = correlate(D, merged["pr_auc"].to_numpy())
                rows.append({
                    "dataset": ds, "metric": metric, "model": model,
                    "n_attacks": n,
                    "spearman": roc["spearman"], "kendall": roc["kendall"],
                    "perm_p": roc["perm_p"],
                    "spearman_p_two_sided": roc["spearman_p_two_sided"],
                    "spearman_pr": pr["spearman"], "kendall_pr": pr["kendall"],
                    "perm_p_pr": pr["perm_p"],
                    "is_reference": model == REFERENCE_MODEL,
                    "exploratory_upper_bound": model == ENVELOPE_LABEL,
                })
    df = pd.DataFrame(rows)
    csv_path = OUT / "per_model_rho.csv"
    df.to_csv(csv_path, index=False)

    # ---- spread across INDIVIDUAL models (envelope excluded) ----------------
    spreads = {}
    for (ds, metric), sub in df[~df.exploratory_upper_bound].groupby(
            ["dataset", "metric"]):
        ref = df[(df.dataset == ds) & (df.metric == metric) & df.is_reference]
        envr = df[(df.dataset == ds) & (df.metric == metric)
                  & df.exploratory_upper_bound]
        r = sub["spearman"]
        spreads[f"{ds}|{metric}"] = {
            "n_models": int(len(sub)),
            "rho_min": float(r.min()),
            "rho_median": float(r.median()),
            "rho_max": float(r.max()),
            "model_at_min": sub.loc[r.idxmin(), "model"],
            "model_at_max": sub.loc[r.idxmax(), "model"],
            "n_rho_positive": int((r > 0).sum()),
            "n_perm_p_le_0.05": int((sub["perm_p"] <= ALPHA).sum()),
            "reference_rbf_svm": ({"spearman": float(ref["spearman"].iloc[0]),
                                   "kendall": float(ref["kendall"].iloc[0]),
                                   "perm_p": float(ref["perm_p"].iloc[0])}
                                  if len(ref) else "IRRECOVERABLE"),
            "envelope_exploratory": ({"spearman": float(envr["spearman"].iloc[0]),
                                      "kendall": float(envr["kendall"].iloc[0]),
                                      "perm_p": float(envr["perm_p"].iloc[0])}
                                     if len(envr) else "IRRECOVERABLE"),
            "pr_auc_rho_min_median_max": [float(sub["spearman_pr"].min()),
                                          float(sub["spearman_pr"].median()),
                                          float(sub["spearman_pr"].max())],
        }

    # ---- mechanical verdicts -------------------------------------------------
    ind = df[~df.exploratory_upper_bound]
    n_cells = int(len(ind))
    n_pos = int((ind["spearman"] > 0).sum())
    n_sig = int((ind["perm_p"] <= ALPHA).sum())
    ref_rows = df[df.is_reference]
    ref_all_pos = bool((ref_rows["spearman"] > 0).all()) if len(ref_rows) else None
    ref_all_sig = bool((ref_rows["perm_p"] <= ALPHA).all()) if len(ref_rows) else None
    env_rows = df[df.exploratory_upper_bound]

    def _fmt_cell(r) -> str:
        return (f"{r['dataset']}/{r['metric']}: rho={r['spearman']:+.3f} "
                f"perm_p={r['perm_p']:.4f}")

    verdict = {
        "n_model_dataset_metric_cells": n_cells,
        "fraction_rho_positive": f"{n_pos}/{n_cells}",
        "fraction_perm_p_le_0.05": f"{n_sig}/{n_cells}",
        "reference_model_rows": [_fmt_cell(r) for _, r in ref_rows.iterrows()],
        "reference_positive_in_all_cells": ref_all_pos,
        "reference_significant_in_all_cells": ref_all_sig,
        "envelope_rows_exploratory": [_fmt_cell(r) for _, r in env_rows.iterrows()],
    }
    if n_pos == n_cells:
        survives = ("YES -- the positive divergence->detectability ordering holds "
                    "for EVERY individual model (all "
                    f"{n_cells} model x dataset x metric cells rho>0), not only "
                    "the best-over-models envelope; the envelope is therefore an "
                    "upper bound on strength, not the source of the sign.")
    elif n_pos >= int(np.ceil(0.75 * n_cells)):
        survives = (f"MOSTLY -- rho>0 in {n_pos}/{n_cells} individual model cells "
                    f"(perm p<=0.05 in {n_sig}); the ordering does not depend on "
                    "envelope selection for the large majority of models, but is "
                    "NOT universal -- see spreads for the exceptions.")
    else:
        survives = (f"NO -- rho>0 in only {n_pos}/{n_cells} individual model "
                    "cells; the published envelope correlation is substantially "
                    "a selection effect and must be reported as exploratory only.")
    verdict["frontier_ordering_survives_without_envelope"] = survives
    verdict["terminology"] = (
        "'model-free frontier' is misleading for the VALIDATION step: D(a) "
        "itself uses no detector, but the published predictand Y(a) was a "
        "per-attack max over models. Preferred wording: "
        "'detector-free divergence, validated per model' with best-over-models "
        "kept only as an exploratory upper bound.")

    # ---- summary JSON --------------------------------------------------------
    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "interpreter": sys.executable,
        "runtime_s": round(time.time() - t0, 1),
        "item": ("per_model -- per-model frontier correlations (selection-bias "
                 "correction for best-over-models detectability)"),
        "protocol": {
            "divergence_source": str(divergence_CSV.relative_to(REPO)),
            "divergence_column": "D",
            "divergence_scope": SCOPE,
            "metrics_reported_separately": list(METRICS),
            "never_combined_note": ("energy and mmd_u are correlated with AUCs "
                                    "strictly separately; no averaging of "
                                    "divergences, models, or datasets."),
            "auc_sources": {
                "dev_grid": str(CROSSATTACK_CSV.relative_to(REPO))
                + " (held-out-attack protocol; models incl. matched-N "
                  "qsvm_crosslayer/qsvm_zz/rbf_svm/random_forest/mlp and the "
                  "full-train classical '__full' variants -- all kept as "
                  "distinct models, mirroring a1._Y_dev_grid's max universe)",
                "nextgen": str(NEXTGEN_CSV.relative_to(REPO))
                + " restricted to variant=='full' exactly as a1._Y_nextgen",
            },
            "targets": ["roc_auc (primary, matches published Y_rocauc)",
                        "pr_auc (secondary, *_pr columns)"],
            "correlations": ("Spearman rho + Kendall tau (scipy two-sided p "
                             "recorded for reference only)"),
            "permutation_test": ("EXACT one-sided permutation p, faithful port "
                                 "of scripts/meta_stats.py::"
                                 "exact_one_sided_p: rankdata average-tie "
                                 "ranks, statistic 1-6*sum d^2/(n(n^2-1)), "
                                 "count(rho_perm >= rho_obs - 1e-12)/n! over "
                                 "ALL n! permutations (dev_grid 9!=362880, "
                                 "nextgen 6!=720); direction PRE-REGISTERED "
                                 "positive (higher D -> higher AUC, the "
                                 "published frontier direction)."),
            "port_faithfulness_check": port_checks,
            "reference_model": {
                "model": REFERENCE_MODEL,
                "why": ("pre-specified reference: untuned classical RBF-SVM, "
                        "present in both datasets, fixed before inspecting "
                        "per-model correlations; no validation-selected "
                        "ensemble exists in the published artifacts, so the "
                        "reference-model option is used"),
            },
            "envelope_handling": ("best-over-models recomputed from the SAME "
                                  "per-model AUC table (per-attack max), "
                                  "appended as final row per dataset x metric "
                                  "with exploratory_upper_bound=True; never "
                                  "pooled with individual models in spreads"),
            "min_attacks_per_cell": 4,
            "seeds": ("none needed -- fully deterministic: exhaustive n! "
                      "permutation enumeration, no sampling anywhere in per_model; "
                      "upstream divergence seeds are recorded in divergence.csv "
                      "'seeds' column"),
        },
        "published_envelope_for_context": pub,
        "sanity": {
            "metric_rank_agreement_within_dataset": metric_rank_agreement,
            "note": ("where energy and mmd_u order the attacks identically "
                     "(rho=1.0), per-model rank correlations are identical "
                     "across the two metric blocks by construction of rank "
                     "statistics -- expected, not an error"),
        },
        "coverage": {
            "per_dataset": coverage,
            "models_only_in_dev_grid": only_dev,
            "models_only_in_nextgen": only_ng,
            "models_in_both": shared,
            "dropped_or_partial_cells": dropped if dropped else "none",
        },
        "spreads_across_individual_models": spreads,
        "verdict": verdict,
        "per_model_table": df.to_dict(orient="records"),
    }
    json_path = OUT / "per_model_summary.json"
    json_path.write_text(json.dumps(summary, indent=2))

    # ---- console report ------------------------------------------------------
    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print("PER-MODEL Spearman(D_full, ROC-AUC)  [perm_p = exact one-sided, "
          "positive direction]")
    print("=" * 110)
    show = df[["dataset", "metric", "model", "n_attacks", "spearman", "kendall",
               "perm_p", "spearman_pr", "perm_p_pr", "is_reference",
               "exploratory_upper_bound"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

    print("\n" + "=" * 110)
    print("SPREAD ACROSS INDIVIDUAL MODELS (envelope excluded)")
    print("=" * 110)
    for key, s in spreads.items():
        ref = s["reference_rbf_svm"]
        envs = s["envelope_exploratory"]
        print(f"  {key:20s} n={s['n_models']}  rho min/med/max = "
              f"{s['rho_min']:+.3f}/{s['rho_median']:+.3f}/{s['rho_max']:+.3f}  "
              f"pos {s['n_rho_positive']}/{s['n_models']}  "
              f"p<=.05 {s['n_perm_p_le_0.05']}/{s['n_models']}  "
              f"ref(rbf_svm) rho={ref['spearman']:+.3f} p={ref['perm_p']:.4f}  "
              f"envelope rho={envs['spearman']:+.3f} (exploratory)")

    print("\nVERDICT: " + verdict["frontier_ordering_survives_without_envelope"])
    print("\nwrote " + str(csv_path))
    print("wrote " + str(json_path))


if __name__ == "__main__":
    main()
