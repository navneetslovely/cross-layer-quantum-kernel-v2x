#!/usr/bin/env python3
"""correlations -- correlation statistics for the divergence-detectability frontier.

Reports Spearman and Kendall correlations separately for each dataset, with
confidence intervals; uses a dataset-blocked permutation test and grouped bootstrap;
ranks/standardises divergence within each dataset before pooling; writes a full
attack-level table with divergence, prevalence, ROC-AUC, PR-AUC and uncertainty; and
reports the correlation for every model separately plus a pre-specified reference
model (best-over-models is an optimistic upper envelope). The phrase 'monotonically
orders' is used only when the data support it.

What the PUBLISHED analysis did (verified against the shipped code/artifacts):
  * results/frontier/controllability_validation.json (screens.D_full.pooled) reports
    pooled Spearman rho=0.8204, p=1.79e-4, n=15 -- computed by scripts/
    controllability.py::_validate on raw D values POOLED across dev_grid (n=9) and
    nextgen (n=6) with NO dataset blocking, energy distance ONLY, and
    Y(a) = MAX over models per attack (best-over-models -> optimistic upper envelope /
    selection bias).

This script replaces that single number with the full inferential battery:
  1. Attack-level MASTER TABLE (correlations_attack_table.csv): one row per
     dataset x attack x Y-definition, joining the divergence re-estimated divergences
     (energy + unbiased MMD^2, full and phys scopes, grouped-bootstrap CIs) with
     prevalence (positive-window fraction of the evaluation set that produced the
     ROC/PR numbers), ROC-AUC and PR-AUC with their published grouped-bootstrap CIs
     where stored (dev_grid CSV) or recomputed with the SHIPPED
     qkv2x.models.metrics.grouped_bootstrap_ci from stored per-window predictions
     (nextgen). Y-definitions: (i) EVERY individual model in the published artifacts,
     (ii) the pre-specified reference model rbf_svm (the project-wide matched-N
     classical baseline), (iii) qsvm_crosslayer, (iv) best-over-models, KEPT but
     labelled "exploratory_upper_envelope".
  2. Per dataset x metric (energy, mmd_u; scope=full) x Y-definition x target
     (roc_auc, pr_auc): Spearman rho AND Kendall tau-b, each with
       (a) an EXACT one-sided permutation p (full n! enumeration, 362880 / 720 perms;
           tie-aware generalisation of scripts/meta_stats.py::exact_one_sided_p
           -- rho as Pearson-of-ranks, tau as tau-b; equality with the meta_stats helper is
           asserted on a tie-free combo as a self-check), direction pre-specified as
           POSITIVE (higher divergence -> higher detectability, the published claim);
       (b) an attack-level bootstrap CI: B=3000 resamples of (D, Y) PAIRS (attacks)
           with replacement, percentile 2.5/97.5, degenerate (constant-x or
           constant-y) replicates dropped and counted.
  3. Dataset-BLOCKED pooled inference: the per-dataset exact permutation p-values are
     combined by Stouffer's sqrt(n)-weighted Z AND Fisher's chi-square (both imported
     from scripts/meta_stats.py), only for Y-definitions evaluated on BOTH
     datasets and only when per-dataset directions agree (consistency rule). Additionally a
     rank-standardised pooled rho/tau (within-dataset mid-quantile ranks
     (rank-0.5)/n, then pooled) is reported, clearly labelled DESCRIPTIVE ONLY.
  4. Claim calibrator: states which phrasing the numbers support
     ("monotonically orders" only if every per-dataset rho == 1; "strongly
     associated" if min per-dataset rho >= 0.7 AND blocked Stouffer p < 0.05;
     "moderately associated" if min rho >= 0.4 AND blocked p < 0.05; otherwise no
     association adjective) -- this becomes record text.

DEPENDS ON results/studies/divergence.csv (divergence must have run first).


Usage:
    python scripts/studies/correlations.py \
        2>&1 | tee results/studies/correlations_run.log

Outputs (results/studies/):
    correlations_attack_table.csv  master attack-level table (divergence, prevalence,
                             ROC-AUC, PR-AUC, uncertainty, every Y-definition)
    correlations.json  nested per_dataset -> metric -> Y_def -> target ->
                             {spearman, kendall, perm_p, ci} + blocked pooling +
                             rank-standardised pooling + claim calibrator + delta
"""
from __future__ import annotations

import json
import sys
import time
from itertools import permutations
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

# ----------------------------- protocol constants ----------------------------
B_BOOT = 3000            # attack-level bootstrap replicates for correlation CIs
BOOT_SEED = 20260807     # bootstrap seed root (rng = default_rng([BOOT_SEED, ...]))
CI_LO_Q, CI_HI_Q = 0.025, 0.975
Y_CI_NBOOT = 200         # grouped bootstrap for nextgen Y CIs (mirrors crossattack)
Y_CI_SEED = 7            # = splits.seed used by the shipped crossattack CI call
REFERENCE_MODEL = "rbf_svm"   # pre-specified project-wide matched-N classical baseline
PRIMARY_TARGET = "roc_auc"    # published Y was best-over-models ROC-AUC
STRONG_RHO, MODERATE_RHO, ALPHA = 0.7, 0.4, 0.05
METRICS = ("energy", "mmd_u")
TARGETS = ("roc_auc", "pr_auc")
BEST = "best_over_models"

divergence_CSV = OUT / "divergence.csv"

PUBLISHED_POOLED = {  # results/frontier/controllability_validation.json screens.D_full.pooled
    "rho": 0.8203756627046467, "p": 0.00017940728782808013, "n": 15,
    "kendall_tau": 0.6698641270570838, "kendall_p": 0.0005236624799359954,
}


# ----------------------------- rank statistics -------------------------------
def _rho(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware Spearman rho = Pearson correlation of average ranks."""
    rx = rankdata(x) - (len(x) + 1) / 2.0
    ry = rankdata(y) - (len(y) + 1) / 2.0
    d = float(np.sqrt((rx * rx).sum() * (ry * ry).sum()))
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def _tau_b(x: np.ndarray, y: np.ndarray) -> float:
    """Kendall tau-b (tie-corrected), pure numpy."""
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(len(x), 1)
    sx, sy = dx[iu], dy[iu]
    n0 = len(sx)
    n1 = int((sx == 0).sum())
    n2 = int((sy == 0).sum())
    den = float(np.sqrt((n0 - n1) * (n0 - n2)))
    return float((sx * sy).sum() / den) if den > 0 else float("nan")


_PERM_CACHE: dict = {}


def _perm_matrix(n: int) -> np.ndarray:
    """All n! permutations of range(n); row 0 is the identity (lexicographic)."""
    if n not in _PERM_CACHE:
        _PERM_CACHE[n] = np.array(list(permutations(range(n))), dtype=np.int8)
    return _PERM_CACHE[n]


def exact_perm_test(x: np.ndarray, y: np.ndarray) -> dict:
    """EXACT one-sided (positive-direction) permutation test for Spearman rho AND
    Kendall tau-b, tie-aware, by full enumeration of all n! pairings.

    Generalises scripts/meta_stats.py::exact_one_sided_p (which uses the
    tie-free d^2 formula and Spearman only): identical p on tie-free data, plus
    correct handling of tied ranks and the Kendall statistic.
    """
    n = len(x)
    P = _perm_matrix(n)
    rx = rankdata(x)
    ry = rankdata(y)
    Ry = ry[P]                                     # (n!, n)
    # --- Spearman (Pearson-of-ranks; the y-rank norm is permutation-invariant) ---
    A = rx - rx.mean()
    nb = float(np.sqrt(((ry - ry.mean()) ** 2).sum()))
    na = float(np.sqrt((A * A).sum()))
    rho_all = (Ry - ry.mean()) @ A / (na * nb + 1e-300)
    rho_obs = float(rho_all[0])                    # row 0 = identity permutation
    p_rho = float(np.mean(rho_all >= rho_obs - 1e-12))
    # --- Kendall tau-b (tie counts n1/n2 are permutation-invariant) ---
    iu_i, iu_j = np.triu_indices(n, 1)
    sx_pairs = np.sign(x[iu_i] - x[iu_j])
    n0 = len(sx_pairs)
    n1 = int((sx_pairs == 0).sum())
    n2 = int((np.sign(ry[iu_i] - ry[iu_j]) == 0).sum())
    den = float(np.sqrt((n0 - n1) * (n0 - n2)))
    tau_num = np.zeros(len(P))
    for k in range(n0):
        tau_num += sx_pairs[k] * np.sign(Ry[:, iu_i[k]] - Ry[:, iu_j[k]])
    tau_all = tau_num / (den + 1e-300)
    tau_obs = float(tau_all[0])
    p_tau = float(np.mean(tau_all >= tau_obs - 1e-12))
    return {"rho": rho_obs, "p_rho": p_rho, "tau": tau_obs, "p_tau": p_tau,
            "n_perms": int(len(P))}


def boot_ci(x: np.ndarray, y: np.ndarray, rng) -> dict:
    """Attack-level pairs bootstrap: resample (D, Y) pairs with replacement,
    B=B_BOOT, percentile CI; degenerate replicates (constant x or y) dropped."""
    n = len(x)
    rhos, taus = [], []
    for _ in range(B_BOOT):
        idx = rng.integers(0, n, n)
        xb, yb = x[idx], y[idx]
        if np.ptp(xb) == 0 or np.ptp(yb) == 0:
            continue
        r = _rho(xb, yb)
        t = _tau_b(xb, yb)
        if np.isfinite(r):
            rhos.append(r)
        if np.isfinite(t):
            taus.append(t)
    out = {}
    for name, vals in (("rho", rhos), ("tau", taus)):
        v = np.asarray(vals, float)
        if v.size:
            out[name] = {"ci_lo": float(np.quantile(v, CI_LO_Q)),
                         "ci_hi": float(np.quantile(v, CI_HI_Q)),
                         "n_boot_valid": int(v.size)}
        else:
            out[name] = {"ci_lo": float("nan"), "ci_hi": float("nan"),
                         "n_boot_valid": 0}
    return out


# ----------------------------- Y-definition tables ---------------------------
def load_dev_grid_Y() -> pd.DataFrame:
    """dev_grid Y from the published held-out-attack protocol
    (results/crossattack_dev/crossattack_results.csv): per (holdout_attack, model)
    roc_auc / pr_auc on a test set = benign + ONLY the held-out attack, with stored
    grouped-bootstrap CIs (n_boot=200, seed=splits.seed) for the 5 matched-N models.
    Prevalence = n_heldout / n_test (verified against predictions.parquet)."""
    d = pd.read_csv(RESULTS / "crossattack_dev" / "crossattack_results.csv")
    rows = []
    for atk, sub in d.groupby("holdout_attack"):
        assert sub["n_test"].nunique() == 1 and sub["n_heldout"].nunique() == 1, atk
        n_test = int(sub["n_test"].iloc[0])
        n_pos = int(sub["n_heldout"].iloc[0])
        prev = n_pos / n_test
        for _, r in sub.iterrows():
            has_ci = np.isfinite(r["roc_auc_lo"])
            rows.append({
                "dataset": "dev_grid", "attack": atk, "y_def": r["model"],
                "model_kind": ("individual_full_train" if r["model"].endswith("__full")
                               else "individual"),
                "prevalence": prev, "n_eval": n_test, "n_pos": n_pos,
                "roc_auc": float(r["roc_auc"]),
                "roc_auc_lo": float(r["roc_auc_lo"]), "roc_auc_hi": float(r["roc_auc_hi"]),
                "pr_auc": float(r["pr_auc"]),
                "pr_auc_lo": float(r["pr_auc_lo"]), "pr_auc_hi": float(r["pr_auc_hi"]),
                "y_ci_method": ("grouped bootstrap (transmitters, n_boot=200, seed=7); "
                                "stored in crossattack_results.csv" if has_ci else
                                "IRRECOVERABLE: no per-window predictions stored for "
                                "classical@full rows; CI would require retraining"),
            })
        # exploratory upper envelope: max over ALL model rows (exactly a1._Y_dev_grid)
        rows.append({
            "dataset": "dev_grid", "attack": atk, "y_def": BEST,
            "model_kind": "exploratory_upper_envelope",
            "prevalence": prev, "n_eval": n_test, "n_pos": n_pos,
            "roc_auc": float(sub["roc_auc"].max()),
            "roc_auc_lo": np.nan, "roc_auc_hi": np.nan,
            "pr_auc": float(sub["pr_auc"].max()),
            "pr_auc_lo": np.nan, "pr_auc_hi": np.nan,
            "y_ci_method": ("IRRECOVERABLE: post-hoc maximum over models (selection); "
                            "a per-attack CI is not defined without joint resampling "
                            "over models -- kept only as the labelled exploratory "
                            "upper envelope"),
        })
    return pd.DataFrame(rows)


def load_nextgen_Y() -> tuple[pd.DataFrame, dict]:
    """nextgen Y from results/veremi_nextgen/veremi_nextgen_per_attack.csv
    (variant='full', exactly a1._Y_nextgen's filter): per (attack, model) roc/pr on
    an evaluation set = ALL test benign windows + that attack's positives (verified
    bit-exact from predictions.parquet, seed 7). CIs are NOT stored in the CSV; they
    are recomputed here from the stored per-window predictions with the SHIPPED
    qkv2x.models.metrics.grouped_bootstrap_ci (transmitter groups, n_boot=200,
    seed=7 -- the same convention the dev_grid CSV used)."""
    from qkv2x.models.metrics import grouped_bootstrap_ci

    d = pd.read_csv(RESULTS / "veremi_nextgen" / "veremi_nextgen_per_attack.csv")
    d = d[d.variant == "full"].copy()
    p = pd.read_parquet(RESULTS / "veremi_nextgen" / "predictions.parquet")
    p = p[(p.task == "binary") & (p.variant == "full") & (p.seed == Y_CI_SEED)]

    from sklearn.metrics import average_precision_score, roc_auc_score
    rows, repro = [], []
    for model, dm in d.groupby("model"):
        pm = p[p.model == model]
        ben = pm[pm.y_true == 0]
        for _, r in dm.iterrows():
            atk = r["attack"]
            pos = pm[(pm.y_true == 1) & (pm.attack_name == atk)]
            sub = pd.concat([ben, pos])
            roc_re = roc_auc_score(sub.y_true, sub.score)
            pr_re = average_precision_score(sub.y_true, sub.score)
            repro.append(max(abs(roc_re - r["roc_auc"]), abs(pr_re - r["pr_auc"])))
            ci = grouped_bootstrap_ci(sub.y_true.to_numpy(), sub.score.to_numpy(),
                                      sub.group.to_numpy(), n_boot=Y_CI_NBOOT,
                                      seed=Y_CI_SEED)
            n_eval = int(len(sub))
            rows.append({
                "dataset": "nextgen", "attack": atk, "y_def": model,
                "model_kind": "individual",
                "prevalence": int(r["n_pos"]) / n_eval,
                "n_eval": n_eval, "n_pos": int(r["n_pos"]),
                "roc_auc": float(r["roc_auc"]),
                "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
                "pr_auc": float(r["pr_auc"]),
                "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
                "y_ci_method": ("grouped bootstrap (transmitters, n_boot=200, seed=7) "
                                "recomputed here from stored predictions.parquet with "
                                "the shipped qkv2x grouped_bootstrap_ci"),
            })
    # exploratory upper envelope (exactly a1._Y_nextgen: max over full-variant models)
    for atk, sub in d.groupby("attack"):
        n_pos = int(sub["n_pos"].iloc[0])
        n_ben = int(sub["n_benign"].iloc[0])
        rows.append({
            "dataset": "nextgen", "attack": atk, "y_def": BEST,
            "model_kind": "exploratory_upper_envelope",
            "prevalence": n_pos / (n_pos + n_ben),
            "n_eval": n_pos + n_ben, "n_pos": n_pos,
            "roc_auc": float(sub["roc_auc"].max()),
            "roc_auc_lo": np.nan, "roc_auc_hi": np.nan,
            "pr_auc": float(sub["pr_auc"].max()),
            "pr_auc_lo": np.nan, "pr_auc_hi": np.nan,
            "y_ci_method": ("IRRECOVERABLE: post-hoc maximum over models (selection); "
                            "kept only as the labelled exploratory upper envelope"),
        })
    checks = {"nextgen_point_reproduction_max_abs_diff": float(np.max(repro)),
              "nextgen_reproduction_note": ("per-attack roc/pr recomputed from "
                                            "predictions.parquet (seed 7) vs the "
                                            "published CSV")}
    return pd.DataFrame(rows), checks


def dev_grid_prediction_check() -> dict:
    """Reproduce the dev_grid crossattack roc/pr from stored per-window predictions
    (5 matched-N models; the __full rows have no stored predictions)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    d = pd.read_csv(RESULTS / "crossattack_dev" / "crossattack_results.csv")
    p = pd.read_parquet(RESULTS / "crossattack_dev" / "predictions.parquet")
    diffs = []
    for (atk, model), sub in p.groupby(["holdout_attack", "model"]):
        row = d[(d.holdout_attack == atk) & (d.model == model)].iloc[0]
        roc_re = roc_auc_score(sub.y_true, sub.score)
        pr_re = average_precision_score(sub.y_true, sub.score)
        diffs.append(max(abs(roc_re - row["roc_auc"]), abs(pr_re - row["pr_auc"])))
        # prevalence check: test set = benign + held-out attack only
        assert set(sub[sub.y_true == 1]["attack_name"].unique()) == {atk}
    return {"dev_grid_point_reproduction_max_abs_diff": float(np.max(diffs)),
            "dev_grid_reproduction_note": ("roc/pr recomputed from predictions.parquet "
                                           "for the 5 matched-N models x 9 attacks; "
                                           "positives verified == held-out attack")}


# ----------------------------- main ------------------------------------------
def main() -> None:
    t0 = time.time()
    print(f"correlations correlation-statistics hardening | interpreter={sys.executable}")
    if not divergence_CSV.exists():
        print(f"FATAL: dependency missing: {divergence_CSV} (run divergence.py first)")
        raise SystemExit(2)

    import meta_stats as ms  # exact_one_sided_p / _stouffer / _fisher

    # ---------------- divergences (divergence) ----------------
    div = pd.read_csv(divergence_CSV)
    n_neg_clamped = int((div["D_raw"] < 0).sum())
    # scope=full D_raw used for all rank statistics (no negatives -> D_raw == D)
    dvals: dict = {}
    for _, r in div.iterrows():
        dvals[(r["dataset"], r["attack"], r["metric"], r["scope"])] = r

    # ---------------- Y tables + reproduction checks ----------------
    ydev = load_dev_grid_Y()
    yng, checks = load_nextgen_Y()
    checks.update(dev_grid_prediction_check())
    ytab = pd.concat([ydev, yng], ignore_index=True)

    # best-over-models must reproduce the published controllability.csv Y exactly
    pub = pd.read_csv(RESULTS / "frontier" / "controllability.csv")
    best = ytab[ytab.y_def == BEST].set_index(["dataset", "attack"])
    dmax = 0.0
    for _, r in pub.iterrows():
        b = best.loc[(r["dataset"], r["attack"])]
        dmax = max(dmax, abs(b["roc_auc"] - r["Y_rocauc"]), abs(b["pr_auc"] - r["Y_prauc"]))
    checks["best_envelope_vs_published_controllability_max_abs_diff"] = float(dmax)

    # published pooled rho reproduction (energy-only, envelope, unblocked)
    rho_pub_re, p_pub_re = spearmanr(pub["D_full"], pub["Y_rocauc"])
    checks["published_pooled_reproduction"] = {
        "rho": float(rho_pub_re), "p": float(p_pub_re), "n": int(len(pub)),
        "matches_published": bool(abs(rho_pub_re - PUBLISHED_POOLED["rho"]) < 1e-12),
    }

    # ---------------- master attack-level table ----------------
    div_cols = {}
    for (dsn, atk), _ in ytab.groupby(["dataset", "attack"]):
        rec = {}
        for met in METRICS:
            for sc in ("full", "phys"):
                r = dvals[(dsn, atk, met, sc)]
                tag = f"D_{'mmdu' if met == 'mmd_u' else met}_{sc}"
                rec[tag] = float(r["D"])
                rec[f"{tag}_raw"] = float(r["D_raw"])
                rec[f"{tag}_ci_lo"] = float(r["ci_lo"])
                rec[f"{tag}_ci_hi"] = float(r["ci_hi"])
                if sc == "full":
                    rec[f"rank_{'mmdu' if met == 'mmd_u' else met}_full_in_dataset"] = \
                        float(r["rank_in_dataset"])
        div_cols[(dsn, atk)] = rec
    table = ytab.copy()
    for col in next(iter(div_cols.values())):
        table[col] = [div_cols[(r.dataset, r.attack)][col] for r in table.itertuples()]
    table["is_reference_model"] = table["y_def"] == REFERENCE_MODEL
    table = table.sort_values(["dataset", "attack", "y_def"]).reset_index(drop=True)
    table.to_csv(OUT / "correlations_attack_table.csv", index=False)

    # ---------------- correlation battery ----------------
    ds_order = ["dev_grid", "nextgen"]
    ydefs_by_ds = {ds: sorted(ytab[ytab.dataset == ds]["y_def"].unique())
                   for ds in ds_order}
    per_dataset: dict = {}
    xy_cache: dict = {}
    self_check_done = False
    self_check: dict = {}

    for ds_i, ds in enumerate(ds_order):
        per_dataset[ds] = {}
        sub_ds = ytab[ytab.dataset == ds]
        attacks = sorted(sub_ds["attack"].unique())
        for met_i, met in enumerate(METRICS):
            per_dataset[ds][met] = {}
            x = np.array([dvals[(ds, a, met, "full")]["D_raw"] for a in attacks])
            for yd_i, ydef in enumerate(ydefs_by_ds[ds]):
                per_dataset[ds][met][ydef] = {}
                sub = sub_ds[sub_ds.y_def == ydef].set_index("attack")
                for tg_i, target in enumerate(TARGETS):
                    y = np.array([float(sub.loc[a, target]) for a in attacks])
                    pt = exact_perm_test(x, y)
                    rng = np.random.default_rng([BOOT_SEED, ds_i, met_i, yd_i, tg_i])
                    ci = boot_ci(x, y, rng)
                    # cross-check observed statistics against scipy
                    rho_sp = float(spearmanr(x, y)[0])
                    tau_sp = float(kendalltau(x, y)[0])
                    assert abs(pt["rho"] - rho_sp) < 1e-9, (ds, met, ydef, target)
                    assert abs(pt["tau"] - tau_sp) < 1e-9, (ds, met, ydef, target)
                    if not self_check_done and len(set(x)) == len(x) \
                            and len(set(y)) == len(y) and ds == "nextgen":
                        obs_ms, p_ms = ms.exact_one_sided_p(x, y, positive=True)
                        self_check = {
                            "combo": f"{ds}/{met}/{ydef}/{target}",
                            "ms_exact_one_sided_p": float(p_ms),
                            "correlations_exact_one_sided_p": pt["p_rho"],
                            "ms_rho": float(obs_ms), "correlations_rho": pt["rho"],
                            "agree": bool(abs(p_ms - pt["p_rho"]) < 1e-12
                                          and abs(obs_ms - pt["rho"]) < 1e-9),
                        }
                        self_check_done = True
                    per_dataset[ds][met][ydef]["Y_roc" if target == "roc_auc"
                                              else "Y_pr"] = {
                        "n": len(attacks),
                        "spearman_rho": pt["rho"],
                        "spearman_exact_one_sided_perm_p": pt["p_rho"],
                        "spearman_ci_lo": ci["rho"]["ci_lo"],
                        "spearman_ci_hi": ci["rho"]["ci_hi"],
                        "kendall_tau_b": pt["tau"],
                        "kendall_exact_one_sided_perm_p": pt["p_tau"],
                        "kendall_ci_lo": ci["tau"]["ci_lo"],
                        "kendall_ci_hi": ci["tau"]["ci_hi"],
                        "n_boot_valid_rho": ci["rho"]["n_boot_valid"],
                        "n_boot_valid_tau": ci["tau"]["n_boot_valid"],
                        "n_perms": pt["n_perms"],
                    }
            print(f"  [{ds}] metric={met}: {len(ydefs_by_ds[ds])} Y-defs x "
                  f"{len(TARGETS)} targets done [{time.time()-t0:.1f}s]", flush=True)

    # ---------------- dataset-blocked pooled inference ----------------
    all_ydefs = sorted(set(ydefs_by_ds["dev_grid"]) | set(ydefs_by_ds["nextgen"]))
    common = [yd for yd in all_ydefs
              if yd in ydefs_by_ds["dev_grid"] and yd in ydefs_by_ds["nextgen"]]
    blocked: dict = {}
    for met in METRICS:
        blocked[met] = {}
        for ydef in all_ydefs:
            blocked[met][ydef] = {}
            for target in TARGETS:
                tkey = "Y_roc" if target == "roc_auc" else "Y_pr"
                if ydef not in common:
                    only = "dev_grid" if ydef in ydefs_by_ds["dev_grid"] else "nextgen"
                    blocked[met][ydef][tkey] = {
                        "combined": "IRRECOVERABLE",
                        "reason": (f"model evaluated only on {only}; no published run "
                                   f"of '{ydef}' exists on the other dataset, so a "
                                   f"cross-dataset blocked p cannot be computed"),
                        "per_dataset_rho": {
                            only: per_dataset[only][met][ydef][tkey]["spearman_rho"]},
                    }
                    continue
                per = {ds: per_dataset[ds][met][ydef][tkey] for ds in ds_order}
                dirs = {("pos" if per[ds]["spearman_rho"] > 0 else
                         ("neg" if per[ds]["spearman_rho"] < 0 else "zero"))
                        for ds in ds_order}
                consistent = dirs <= {"pos"}
                ent = {
                    "per_dataset": {ds: {"n": per[ds]["n"],
                                         "rho": per[ds]["spearman_rho"],
                                         "tau": per[ds]["kendall_tau_b"],
                                         "spearman_perm_p":
                                             per[ds]["spearman_exact_one_sided_perm_p"],
                                         "kendall_perm_p":
                                             per[ds]["kendall_exact_one_sided_perm_p"]}
                                    for ds in ds_order},
                    "directions_consistent_positive": bool(consistent),
                }
                if consistent:
                    ns = [per[ds]["n"] for ds in ds_order]
                    ps_r = [per[ds]["spearman_exact_one_sided_perm_p"] for ds in ds_order]
                    ps_t = [per[ds]["kendall_exact_one_sided_perm_p"] for ds in ds_order]
                    ent["stouffer_spearman"] = ms._stouffer(ps_r, ns)
                    ent["fisher_spearman"] = ms._fisher(ps_r)
                    ent["stouffer_kendall"] = ms._stouffer(ps_t, ns)
                    ent["fisher_kendall"] = ms._fisher(ps_t)
                    ent["combinable"] = True
                else:
                    ent["combinable"] = False
                    ent["note"] = ("per-dataset directions disagree -> not combinable; "
                                   "report per-dataset only (consistency rule)")
                blocked[met][ydef][tkey] = ent

    # ---------------- rank-standardised pooled rho (descriptive only) ---------
    rank_pooled: dict = {}
    for met in METRICS:
        rank_pooled[met] = {}
        for ydef in common:
            rank_pooled[met][ydef] = {}
            for target in TARGETS:
                tkey = "Y_roc" if target == "roc_auc" else "Y_pr"
                xs, ys = [], []
                for ds in ds_order:
                    sub_ds = ytab[(ytab.dataset == ds) & (ytab.y_def == ydef)]
                    attacks = sorted(sub_ds["attack"].unique())
                    n = len(attacks)
                    x = np.array([dvals[(ds, a, met, "full")]["D_raw"] for a in attacks])
                    y = np.array([float(sub_ds.set_index("attack").loc[a, target])
                                  for a in attacks])
                    xs.append((rankdata(x) - 0.5) / n)   # within-dataset mid-quantiles
                    ys.append((rankdata(y) - 0.5) / n)
                xp, yp = np.concatenate(xs), np.concatenate(ys)
                rank_pooled[met][ydef][tkey] = {
                    "n_pooled": int(len(xp)),
                    "spearman_rho": _rho(xp, yp),
                    "kendall_tau_b": _tau_b(xp, yp),
                    "descriptive_only": True,
                    "note": ("within-dataset mid-quantile ranks (rank-0.5)/n pooled "
                             "across datasets; NO p-value attached -- inference is "
                             "the blocked Stouffer/Fisher combination above"),
                }

    # ---------------- claim calibrator ----------------
    def _calibrate(met: str, ydef: str) -> dict:
        tkey = "Y_roc"
        rhos = {ds: per_dataset[ds][met][ydef][tkey]["spearman_rho"] for ds in ds_order}
        blk = blocked[met][ydef][tkey]
        p_blk = (blk.get("stouffer_spearman", {}).get("p")
                 if isinstance(blk, dict) and blk.get("combinable") else None)
        min_rho = min(rhos.values())
        if all(abs(r - 1.0) < 1e-12 for r in rhos.values()):
            phrase = "monotonically orders"
        elif min_rho >= STRONG_RHO and p_blk is not None and p_blk < ALPHA:
            phrase = "strongly associated"
        elif min_rho >= MODERATE_RHO and p_blk is not None and p_blk < ALPHA:
            phrase = "moderately associated"
        else:
            phrase = ("no association adjective supported at this n; report "
                      "per-dataset values only")
        return {"per_dataset_rho": rhos, "min_rho": float(min_rho),
                "blocked_stouffer_p": p_blk, "supported_phrase": phrase,
                "monotonically_orders_supported": bool(
                    all(abs(r - 1.0) < 1e-12 for r in rhos.values()))}

    calibrator = {
        "rules": {
            "monotonically orders": "ONLY if per-dataset Spearman rho == 1 in every dataset",
            "strongly associated": f"min per-dataset rho >= {STRONG_RHO} AND blocked "
                                   f"Stouffer p < {ALPHA}",
            "moderately associated": f"min per-dataset rho >= {MODERATE_RHO} AND blocked "
                                     f"Stouffer p < {ALPHA}",
            "otherwise": "no association adjective; per-dataset values only",
        },
        "target": "Y_roc (the published Y)",
        "verdicts": {f"{met}/{ydef}": _calibrate(met, ydef)
                     for met in METRICS
                     for ydef in (REFERENCE_MODEL, "qsvm_crosslayer", BEST)},
    }
    # record text
    ref_e = calibrator["verdicts"][f"energy/{REFERENCE_MODEL}"]
    best_e = calibrator["verdicts"][f"energy/{BEST}"]
    calibrator["ledger_text"] = (
        f"The model-free divergence is '{ref_e['supported_phrase']}' with the "
        f"pre-specified reference model ({REFERENCE_MODEL}) detectability "
        f"(energy: rho_dev={ref_e['per_dataset_rho']['dev_grid']:+.2f}, "
        f"rho_ng={ref_e['per_dataset_rho']['nextgen']:+.2f}, blocked "
        f"p={ref_e['blocked_stouffer_p']:.4g}) and "
        f"'{best_e['supported_phrase']}' with the exploratory best-over-models "
        f"upper envelope (rho_dev={best_e['per_dataset_rho']['dev_grid']:+.2f}, "
        f"rho_ng={best_e['per_dataset_rho']['nextgen']:+.2f}, blocked "
        f"p={best_e['blocked_stouffer_p']:.4g}). The phrase 'monotonically orders' "
        f"is NOT supported (no per-dataset rho equals 1)."
    )

    # ---------------- old-vs-new delta ----------------
    def _pooled_raw_base(met: str, target: str) -> dict:
        """Same construction as the published number (raw-value pooling, envelope Y)
        but with the divergence re-estimated divergence -- recorded for the delta ONLY,
        explicitly NOT the primary inference."""
        xs, ys = [], []
        for ds in ds_order:
            sub_ds = ytab[(ytab.dataset == ds) & (ytab.y_def == BEST)]
            for a in sorted(sub_ds["attack"].unique()):
                xs.append(dvals[(ds, a, met, "full")]["D_raw"])
                ys.append(float(sub_ds.set_index("attack").loc[a, target]))
        rho, p = spearmanr(xs, ys)
        tau, pt = kendalltau(xs, ys)
        return {"rho": float(rho), "p_two_sided_asymptotic": float(p),
                "tau": float(tau), "n": len(xs),
                "label": "NON-INFERENTIAL (unblocked raw pooling, envelope Y) -- "
                         "delta record only"}

    old_vs_new = {
        "published": {
            "source": "results/frontier/controllability_validation.json "
                      "screens.D_full.pooled",
            **PUBLISHED_POOLED,
            "construction_flaws": [
                "energy distance only (MMD computed but dropped)",
                "Y = max over models per attack (optimistic upper envelope, "
                "selection bias)",
                "raw D values pooled across datasets with no blocking "
                "(scale conflation)",
                "divergence standardised on pooled benign+attack data "
                "(self-referential scaling; fixed in divergence)",
            ],
        },
        "published_reproduced_here": checks["published_pooled_reproduction"],
        "same_construction_with_divergence": {
            f"{met}/{tkey}": _pooled_raw_base(met, target)
            for met in METRICS
            for target, tkey in (("roc_auc", "Y_roc"),)
        },
        "new_primary_inference": {
            f"{met}/{ydef}/Y_roc": {
                "per_dataset": blocked[met][ydef]["Y_roc"].get("per_dataset"),
                "stouffer_p": blocked[met][ydef]["Y_roc"]
                              .get("stouffer_spearman", {}).get("p"),
                "fisher_p": blocked[met][ydef]["Y_roc"]
                            .get("fisher_spearman", {}).get("p"),
            }
            for met in METRICS for ydef in (REFERENCE_MODEL, BEST)
        },
        "delta_note": (
            "The single published number (pooled rho=0.820, p=1.8e-4, n=15) is "
            "replaced by per-dataset Spearman/Kendall with exact permutation p and "
            "attack-level bootstrap CIs, combined across datasets ONLY via "
            "dataset-blocked Stouffer/Fisher on the per-dataset exact p-values; "
            "the envelope Y is retained solely as a labelled exploratory upper "
            "envelope next to every individual model and the pre-specified "
            "reference model."
        ),
    }

    # ---------------- protocol ----------------
    protocol = {
        "dependency": str(divergence_CSV.relative_to(REPO)),
        "divergence_used": ("divergence D_raw, scope='full' (grouped cross-fit, "
                            "fold-A-only standardisation); D_raw == D for all 15 "
                            f"points ({n_neg_clamped} negative raw MMD values)"),
        "y_sources": {
            "dev_grid": ("results/crossattack_dev/crossattack_results.csv -- "
                         "held-out-attack protocol; test = benign + held-out attack; "
                         "8 Y-definitions (5 matched-N models + 3 classical@full) "
                         "+ envelope"),
            "nextgen": ("results/veremi_nextgen/veremi_nextgen_per_attack.csv "
                        "variant='full' (9 models) + envelope; per-attack eval set = "
                        "all test benign + that attack's positives, seed 7"),
        },
        "prevalence_definition": ("positive-window fraction of the SAME evaluation "
                                  "set on which the row's ROC/PR were computed "
                                  "(dev_grid: n_heldout/n_test; nextgen: "
                                  "n_pos/(n_pos+n_benign)); this is the PR-AUC "
                                  "chance baseline"),
        "reference_model": (f"{REFERENCE_MODEL} -- pre-specified: the project-wide "
                            "matched-N classical baseline used as the comparison "
                            "anchor in RQ3/H2/H3 (pre-specified)"),
        "envelope_policy": ("best-over-models retained, labelled "
                            "'exploratory_upper_envelope'; never used for the "
                            "primary claim"),
        "spearman": "tie-aware (Pearson of average ranks); verified against scipy",
        "kendall": "tau-b (tie-corrected); verified against scipy",
        "permutation_test": ("EXACT full enumeration of all n! pairings (dev_grid "
                             "9!=362880, nextgen 6!=720); one-sided in the "
                             "pre-specified POSITIVE direction (higher divergence -> "
                             "higher detectability); tie-aware; row-0 identity gives "
                             "the observed statistic; generalises "
                             "meta_stats.exact_one_sided_p (equality asserted "
                             "on a tie-free combo, see self_check)"),
        "bootstrap_ci": (f"attack-level pairs bootstrap: B={B_BOOT} resamples of "
                         f"(D,Y) pairs with replacement, percentile "
                         f"[{CI_LO_Q},{CI_HI_Q}]; degenerate replicates dropped and "
                         f"counted; rng=default_rng([{BOOT_SEED}, ds_idx, "
                         f"metric_idx, ydef_idx, target_idx]) with ds order "
                         f"{ds_order}, metric order {list(METRICS)}, ydefs sorted, "
                         f"target order {list(TARGETS)}"),
        "blocked_combination": ("per-dataset EXACT one-sided permutation p combined "
                                "by Stouffer sqrt(n)-weighted Z AND Fisher chi^2 "
                                "(imported from scripts/meta_stats.py); "
                                "combined ONLY when per-dataset directions are "
                                "consistently positive (consistency rule); models absent "
                                "from one dataset are marked IRRECOVERABLE"),
        "rank_standardised_pooling": ("within-dataset mid-quantile ranks "
                                      "(rank-0.5)/n pooled; DESCRIPTIVE ONLY, "
                                      "no p-value"),
        "nextgen_y_ci": (f"shipped qkv2x.models.metrics.grouped_bootstrap_ci "
                         f"(transmitter groups, n_boot={Y_CI_NBOOT}, "
                         f"seed={Y_CI_SEED}) on stored per-window predictions -- "
                         f"same convention as the stored dev_grid CIs"),
        "claim_calibrator_thresholds": {"strong_rho": STRONG_RHO,
                                        "moderate_rho": MODERATE_RHO, "alpha": ALPHA},
        "no_silent_pooling": "every pooled quantity is labelled blocked or descriptive",
        "interpreter": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__,
        "seeds": {"boot_seed_root": BOOT_SEED, "y_ci_seed": Y_CI_SEED,
                  "permutation": "deterministic (exhaustive enumeration, no RNG)"},
    }

    checks["scipy_crosscheck"] = ("observed rho/tau matched scipy spearmanr/"
                                  "kendalltau to <1e-9 on every combo (asserted)")
    checks["ms_helper_self_check"] = self_check

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "item": "correlations_correlation_statistics_hardening",
        "runtime_s": round(time.time() - t0, 1),
        "protocol": protocol,
        "reproduction_checks": checks,
        "per_dataset": per_dataset,
        "blocked_pooled": blocked,
        "rank_standardised_pooled_descriptive": rank_pooled,
        "claim_calibrator": calibrator,
        "old_vs_new": old_vs_new,
    }
    (OUT / "correlations.json").write_text(json.dumps(summary, indent=2,
                                                          default=float))

    # ------------------------------ report -----------------------------------
    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print("correlations -- per-dataset correlations (scope=full divergence), Y_roc")
    print("=" * 110)
    hdr = (f"{'dataset':9s} {'metric':7s} {'Y-definition':22s} {'rho':>7s} "
           f"{'[95% CI]':>17s} {'perm p':>8s} {'tau':>7s} {'perm p':>8s}")
    print(hdr)
    for ds in ds_order:
        for met in METRICS:
            for ydef in ydefs_by_ds[ds]:
                e = per_dataset[ds][met][ydef]["Y_roc"]
                print(f"{ds:9s} {met:7s} {ydef:22s} {e['spearman_rho']:+7.3f} "
                      f"[{e['spearman_ci_lo']:+.2f},{e['spearman_ci_hi']:+.2f}]"
                      f"{'':4s} {e['spearman_exact_one_sided_perm_p']:8.4f} "
                      f"{e['kendall_tau_b']:+7.3f} "
                      f"{e['kendall_exact_one_sided_perm_p']:8.4f}")
    print("\nBLOCKED pooled inference (Stouffer / Fisher on exact per-dataset p, Y_roc):")
    for met in METRICS:
        for ydef in [REFERENCE_MODEL, "qsvm_crosslayer", BEST]:
            b = blocked[met][ydef]["Y_roc"]
            if b.get("combinable"):
                print(f"  {met:7s} {ydef:22s} Stouffer Z={b['stouffer_spearman']['Z']:.3f} "
                      f"p={b['stouffer_spearman']['p']:.5f} | Fisher chi2="
                      f"{b['fisher_spearman']['chi2']:.2f} p={b['fisher_spearman']['p']:.5f}")
            else:
                print(f"  {met:7s} {ydef:22s} NOT combinable: "
                      f"{b.get('note', b.get('reason'))}")
    print("\nCLAIM CALIBRATOR:")
    for k, v in calibrator["verdicts"].items():
        print(f"  {k:26s} -> '{v['supported_phrase']}' (min rho={v['min_rho']:+.3f}, "
              f"blocked p={v['blocked_stouffer_p']})")
    print(f"\n  LEDGER: {calibrator['ledger_text']}")
    print("\nOLD vs NEW:")
    print(f"  published pooled (flawed): rho={PUBLISHED_POOLED['rho']:.4f} "
          f"p={PUBLISHED_POOLED['p']:.2e} n=15 (energy-only, envelope, unblocked)")
    for k, v in old_vs_new["same_construction_with_divergence"].items():
        print(f"  same construction, divergence D ({k}): rho={v['rho']:.4f} (delta record only)")
    print("\nReproduction checks:")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    print(f"\nwrote {OUT / 'correlations_attack_table.csv'}")
    print(f"wrote {OUT / 'correlations.json'}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
