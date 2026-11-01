#!/usr/bin/env python3
"""divergence -- reproducible divergence re-estimation.

Reports energy distance and MMD separately, with the exact estimator, the biased or
unbiased MMD form, the kernel bandwidth rule, feature scaling, the grouped
cross-fitting method, fold count, fold aggregation, confidence-interval method and
dataset-wise normalisation all specified. Different divergences are never combined
into one number.

What the PUBLISHED pipeline did (verified against the shipped code):
  * scripts/controllability.py computes BOTH an RBF-MMD^2 (`_mmd2_rbf`, a BIASED
    V-statistic with a pooled-median bandwidth) and an energy distance (`_energy`,
    V-statistic) per attack, but only the ENERGY value reaches
    results/frontier/controllability.csv; the published pooled Spearman rho=0.82
    (results/frontier/controllability_validation.json) is therefore energy-only.
  * a1 standardises features on ALL data (benign+attack pooled, quantile-clipped
    z-score) BEFORE computing the divergence -> self-referential scaling.
  * scripts/separability_grouped.py already has the grouped machinery
    (transmitter-grouped subsampling, cross-fit standardisation on a disjoint
    benign fold, grouped bootstrap) but again reports ENERGY only.

This script re-estimates every attack x dataset divergence point used by the
published frontier analysis (dev_grid: 9 attacks, veremi_nextgen "nextgen":
6 attacks) with every estimator choice spelled out, energy and MMD reported
STRICTLY separately (never averaged or combined):

  1. Estimators (per standardised sample pair A=attack, B=benign):
       energy  : V-statistic (biased) form, identical to the published `_energy`:
                 E = 2*mean(d(A,B)) - mean(d(A,A)) - mean(d(B,B))
                 (means INCLUDE the zero diagonal; Euclidean distance d).
       mmd_u   : UNBIASED U-statistic RBF-MMD^2:
                 MMD^2_u = sum_{i!=j} k(a_i,a_j)/(m(m-1))
                         + sum_{i!=j} k(b_i,b_j)/(n(n-1))
                         - 2/(mn) sum_{ij} k(a_i,b_j),
                 k(x,y) = exp(-||x-y||^2 / (2 sigma^2)).
                 Raw (possibly negative) value stored as D_raw; D = max(D_raw, 0).
  2. MAIN PATH = transmitter-grouped 2-fold cross-fitting (mirrors a1b):
     benign transmitter groups are split 50/50 into folds; the standardiser
     (clip to the fit fold's [q0.005, q0.995] per feature, then mean/std --
     exactly a1b's `_xfit_standardize`) is fit on benign fold A ONLY; the
     divergence(attack vs benign) is computed on fold-B benign + attack windows,
     both standardised by fold-A stats; roles are swapped and the MEAN over the
     2 fold-orders is reported. The median-heuristic bandwidth sigma (median
     nonzero pairwise Euclidean distance) is computed ONLY from the fitting
     fold's standardised benign windows (<= N_BW rows of the fit draw).
  3. Bandwidth sensitivity: mmd_u recomputed at {0.5x, 1x, 2x} sigma ->
     divergence_bandwidth_sensitivity.csv + ordering-stability Spearman.
  4. Uncertainty: grouped bootstrap -- transmitter groups resampled WITH
     replacement (attack groups; and benign groups within each fold after a
     fresh fold split), B=200, seeded; CI = 2.5/97.5 percentiles of the
     replicate raw statistic.
  5. D_phys alongside D_full: identical protocol on the non-controllable
     "phys" feature subset from results/frontier/feature_controllability.csv,
     selected exactly as controllability.py::_controllability_for does
     (rows with network_mode == dataset mode and controllability == "phys");
     the SAME window draws are used for full and phys (columns sliced), and
     phys gets its OWN median-heuristic bandwidth on the sliced fit fold.
  6. rank_in_dataset: rank of D within (dataset, metric, scope), ascending
     (1 = smallest), scipy.stats.rankdata average method. POOLING across
     datasets is deliberately NOT done here (correlations's job).
  7. Subsampling scale mirrors a1b: N_PER_CLASS=300 windows per class drawn as
     WHOLE transmitter groups (draw stops at the first group crossing the
     target, so slight overshoot is possible), 3 seeds for the point estimate.

Loaders mirror separability_grouped.py::_loaders (raw, UN-standardised X +
transmitter groups); standardisation happens only inside the cross-fit, which
is the entire point. 

Usage:
    python scripts/studies/divergence.py \
        2>&1 | tee results/studies/divergence_run.log

Outputs (results/studies/):
    divergence.csv  long format, one row per
                                     dataset x attack x metric x scope
    divergence_bandwidth_sensitivity.csv  mmd_u at {0.5,1,2} x sigma
    divergence_summary.json  full protocol + sanity + headline numbers
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import rankdata, spearmanr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RESULTS = REPO / "results"
OUT = RESULTS / "studies"
OUT.mkdir(parents=True, exist_ok=True)

# ----------------------------- protocol constants ----------------------------
N_PER_CLASS = 300      # score-side windows per class (mirrors a1b N_PER_CLASS)
N_FIT = 1000           # benign fit-fold windows used for the standardiser
N_BW = 512             # <= this many standardised fit windows for the median
N_SEEDS = 3            # point-estimate repetitions (fresh grouped draws)
BASE_SEED = 20260807   # point-estimate seed root
BOOT_SEED = 913        # bootstrap seed root
B_BOOT = 200           # grouped-bootstrap replicates
N_FOLDS = 2            # benign cross-fitting folds
BW_MULTS = (0.5, 1.0, 2.0)
MIN_ATTACK_WINDOWS = 30  # a1's attack floor (keeps the published 15 points)
CI_LO_Q, CI_HI_Q = 0.025, 0.975


# ----------------------------- estimators ------------------------------------
def _energy_v(Daa: np.ndarray, Dbb: np.ndarray, Dab: np.ndarray) -> float:
    """Energy distance, V-statistic (biased) form -- IDENTICAL to the published
    controllability._energy / separability_grouped._energy:
        2*mean(Dab) - mean(Daa) - mean(Dbb), means including the zero diagonal."""
    return float(2.0 * Dab.mean() - Daa.mean() - Dbb.mean())


def _mmd2_u(Daa: np.ndarray, Dbb: np.ndarray, Dab: np.ndarray,
            sigma: float) -> float:
    """UNBIASED (U-statistic) RBF-MMD^2 with k(x,y)=exp(-||x-y||^2/(2 sigma^2)).
    Within-sample sums exclude the diagonal; raw value may be negative."""
    m, n = Daa.shape[0], Dbb.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    g = 1.0 / (2.0 * sigma * sigma + 1e-12)
    Kaa = np.exp(-g * Daa ** 2)
    Kbb = np.exp(-g * Dbb ** 2)
    Kab = np.exp(-g * Dab ** 2)
    t_aa = (Kaa.sum() - np.trace(Kaa)) / (m * (m - 1))
    t_bb = (Kbb.sum() - np.trace(Kbb)) / (n * (n - 1))
    return float(t_aa + t_bb - 2.0 * Kab.mean())


def _xfit_standardizer(fit: np.ndarray):
    """separability_grouped._xfit_standardize, refactored to return a transform:
    clip to the FIT fold's [q0.005, q0.995] per feature, then (x-mu)/sd with
    mu/sd of the clipped fit fold (sd floor 1e-9)."""
    lo = np.quantile(fit, 0.005, axis=0)
    hi = np.quantile(fit, 0.995, axis=0)
    fc = np.clip(fit, lo, hi)
    mu = fc.mean(axis=0)
    sd = fc.std(axis=0) + 1e-9

    def transform(x: np.ndarray) -> np.ndarray:
        return (np.clip(x, lo, hi) - mu) / sd

    return transform


def _median_bandwidth(F: np.ndarray) -> float:
    """Median heuristic: median NONZERO pairwise Euclidean distance (same
    zero-exclusion rule as the published a1._mmd2_rbf); fallback 1.0."""
    d = cdist(F, F)
    nz = d[d > 0]
    return float(np.median(nz)) if nz.size else 1.0


# ----------------------------- grouped draws ---------------------------------
def _group_rows(groups: np.ndarray, rows: np.ndarray) -> dict:
    """Map group id -> absolute row indices, for the given subset of rows."""
    out: dict = {}
    df = pd.DataFrame({"g": groups[rows]})
    for g, pos in df.groupby("g").indices.items():
        out[g] = rows[pos]
    return out


def _draw(group_rows: dict, ids: np.ndarray, target: int, rng,
          replace: bool) -> np.ndarray:
    """Draw WHOLE transmitter groups until >= target windows.
    replace=False: permute unique groups, accumulate (a1b._grouped_sample).
    replace=True : i.i.d. group draws with replacement (cluster bootstrap);
    duplicated groups contribute duplicated rows. Slight overshoot possible."""
    if len(ids) == 0:
        return np.array([], dtype=int)
    out, tot = [], 0
    if not replace:
        for g in rng.permutation(ids):
            r = group_rows[g]
            out.append(r)
            tot += len(r)
            if tot >= target:
                break
    else:
        cap = 50 * max(1, len(ids))
        draws = 0
        while tot < target and draws < cap:
            g = ids[rng.integers(0, len(ids))]
            r = group_rows[g]
            out.append(r)
            tot += len(r)
            draws += 1
    return np.concatenate(out)


# ----------------------------- one cross-fit evaluation ----------------------
def eval_crossfit(X: np.ndarray, atk_state: dict, ben_state: dict,
                  scopes: dict, rng, replace: bool, bw_mults=(1.0,)) -> dict:
    """One transmitter-grouped 2-fold cross-fit evaluation.

    Returns raw divergences averaged over the 2 fold-orders:
      {"energy": {scope: raw}, "mmd_u": {scope: {mult: raw}},
       "sigma": {scope: mean sigma}, "n_attack": .., "n_benign": ..}
    """
    # attack draw (shared across the two fold-orders)
    a_target = min(N_PER_CLASS, atk_state["n_total"])
    a_rows = _draw(atk_state["group_rows"], atk_state["ids"], a_target, rng, replace)
    # benign fold split on the ORIGINAL unique groups
    perm = rng.permutation(ben_state["ids"])
    half = max(1, len(perm) // 2)
    folds = (perm[:half], perm[half:])
    acc = {"energy": {s: [] for s in scopes},
           "mmd_u": {s: {m: [] for m in bw_mults} for s in scopes},
           "sigma": {s: [] for s in scopes},
           "n_benign": []}
    for fit_ids, score_ids in ((folds[0], folds[1]), (folds[1], folds[0])):
        n_fit_avail = sum(len(ben_state["group_rows"][g]) for g in fit_ids)
        n_sc_avail = sum(len(ben_state["group_rows"][g]) for g in score_ids)
        fit_rows = _draw(ben_state["group_rows"], fit_ids,
                         min(N_FIT, n_fit_avail), rng, replace)
        score_rows = _draw(ben_state["group_rows"], score_ids,
                           min(N_PER_CLASS, n_sc_avail), rng, replace)
        if len(fit_rows) < 10 or len(score_rows) < 10 or len(a_rows) < 2:
            continue
        transform = _xfit_standardizer(X[fit_rows])
        A = transform(X[a_rows])
        B = transform(X[score_rows])
        F = transform(X[fit_rows])[:N_BW]
        acc["n_benign"].append(len(score_rows))
        for scope, cols in scopes.items():
            Fa, Fb, Ff = A[:, cols], B[:, cols], F[:, cols]
            sigma = _median_bandwidth(Ff)
            Daa = cdist(Fa, Fa)
            Dbb = cdist(Fb, Fb)
            Dab = cdist(Fa, Fb)
            acc["sigma"][scope].append(sigma)
            acc["energy"][scope].append(_energy_v(Daa, Dbb, Dab))
            for m in bw_mults:
                acc["mmd_u"][scope][m].append(_mmd2_u(Daa, Dbb, Dab, sigma * m))
    if not acc["n_benign"]:
        return None
    return {
        "energy": {s: float(np.mean(v)) for s, v in acc["energy"].items()},
        "mmd_u": {s: {m: float(np.mean(v)) for m, v in ms.items()}
                  for s, ms in acc["mmd_u"].items()},
        "sigma": {s: float(np.mean(v)) for s, v in acc["sigma"].items()},
        "n_attack": int(len(a_rows)),
        "n_benign": float(np.mean(acc["n_benign"])),
        "n_folds_used": len(acc["n_benign"]),
    }


# ----------------------------- dataset loaders (mirror a1b) ------------------
def load_datasets() -> list:
    """RAW (un-standardised) windows + transmitter groups + feature names.
    Mirrors separability_grouped._loaders; feature names added for the phys
    subset; network_mode as in a1 (dev_grid=rssi, nextgen=timing)."""
    from qkv2x.config import load_yaml
    from qkv2x.experiments.common import load_experiment_data

    cfg = load_yaml(REPO / "configs" / "features.yaml")
    d = load_experiment_data(REPO / "runs" / "dev_grid", cfg)
    dev = {"name": "dev_grid", "network_mode": "rssi",
           "X": np.asarray(d.X, float), "y": np.asarray(d.y, int),
           "attack": np.asarray(d.attack_name),
           "groups": np.asarray(d.groups, int),
           "features": list(d.feature_names)}
    out = [dev]
    f = RESULTS / "veremi_nextgen" / "checkpoints" / "assembled.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        out.append({"name": "nextgen", "network_mode": "timing",
                    "X": np.asarray(z["X"], float),
                    "y": np.asarray(z["y"], int),
                    "attack": np.asarray([str(s) for s in z["attack_name"]]),
                    "groups": np.asarray(z["groups"], int),
                    "features": [str(s) for s in z["feature_names"]]})
    return out


def phys_scope(ds: dict) -> list:
    """EXACTLY controllability._controllability_for's phys selection."""
    ctab = pd.read_csv(RESULTS / "frontier" / "feature_controllability.csv")
    phys = set(ctab[(ctab.network_mode == ds["network_mode"])
                    & (ctab.controllability == "phys")].feature)
    return [i for i, f in enumerate(ds["features"]) if f in phys]


# ----------------------------- main ------------------------------------------
def main() -> None:
    t0 = time.time()
    print(f"divergence divergence re-estimation | interpreter={sys.executable}")
    print(f"protocol: N_PER_CLASS={N_PER_CLASS} N_FIT={N_FIT} N_BW={N_BW} "
          f"seeds={N_SEEDS} (base {BASE_SEED}) B_boot={B_BOOT} (base {BOOT_SEED}) "
          f"folds={N_FOLDS} bw_mults={BW_MULTS}", flush=True)

    pub = pd.read_csv(RESULTS / "frontier" / "controllability.csv")
    pub_val = json.loads(
        (RESULTS / "frontier" / "controllability_validation.json").read_text())
    pub_Y = {(r["dataset"], r["attack"]): float(r["Y_rocauc"])
             for _, r in pub.iterrows()}
    pub_energy_D = {(r["dataset"], r["attack"]): float(r["D_full"])
                    for _, r in pub.iterrows()}

    datasets = load_datasets()
    rows, sens_rows, point_details = [], [], {}

    for ds_idx, ds in enumerate(datasets):
        name = ds["name"]
        X, y, atk, grp = ds["X"], ds["y"], ds["attack"], ds["groups"]
        d_all = X.shape[1]
        p_idx = phys_scope(ds)
        scopes = {"full": list(range(d_all)), "phys": p_idx}
        ben_rows = np.where(y == 0)[0]
        ben_state = {"group_rows": _group_rows(grp, ben_rows),
                     "ids": np.unique(grp[ben_rows])}
        print(f"\n=== {name}: N={len(y)} d={d_all} phys_d={len(p_idx)} "
              f"benign={len(ben_rows)} windows / {len(ben_state['ids'])} groups ===",
              flush=True)

        attacks = [a for a in sorted(set(atk[y == 1]))
                   if a not in ("none", "benign", "")
                   and int(((atk == a) & (y == 1)).sum()) >= MIN_ATTACK_WINDOWS]
        for atk_idx, a in enumerate(attacks):
            a_rows_all = np.where((atk == a) & (y == 1))[0]
            atk_state = {"group_rows": _group_rows(grp, a_rows_all),
                         "ids": np.unique(grp[a_rows_all]),
                         "n_total": int(len(a_rows_all))}

            # ---- point estimate: N_SEEDS grouped draws WITHOUT replacement --
            evs = []
            for s in range(N_SEEDS):
                rng = np.random.default_rng([BASE_SEED, ds_idx, atk_idx, s])
                ev = eval_crossfit(X, atk_state, ben_state, scopes, rng,
                                   replace=False, bw_mults=BW_MULTS)
                if ev is not None:
                    evs.append(ev)
            if not evs:
                print(f"  {a}: IRRECOVERABLE (no valid cross-fit evaluation)")
                continue
            point = {
                "energy": {sc: float(np.mean([e["energy"][sc] for e in evs]))
                           for sc in scopes},
                "mmd_u": {sc: {m: float(np.mean([e["mmd_u"][sc][m] for e in evs]))
                          for m in BW_MULTS} for sc in scopes},
                "sigma": {sc: float(np.mean([e["sigma"][sc] for e in evs]))
                          for sc in scopes},
                "n_attack": float(np.mean([e["n_attack"] for e in evs])),
                "n_benign": float(np.mean([e["n_benign"] for e in evs])),
            }

            # ---- grouped bootstrap: groups WITH replacement, B=200 ----------
            boots = {("energy", sc): [] for sc in scopes}
            boots.update({("mmd_u", sc): [] for sc in scopes})
            for b in range(B_BOOT):
                rng = np.random.default_rng([BOOT_SEED, ds_idx, atk_idx, b])
                ev = eval_crossfit(X, atk_state, ben_state, scopes, rng,
                                   replace=True, bw_mults=(1.0,))
                if ev is None:
                    continue
                for sc in scopes:
                    boots[("energy", sc)].append(ev["energy"][sc])
                    boots[("mmd_u", sc)].append(ev["mmd_u"][sc][1.0])

            seeds_str = f"point:[{BASE_SEED},{ds_idx},{atk_idx},0..{N_SEEDS-1}];" \
                        f"boot:[{BOOT_SEED},{ds_idx},{atk_idx},0..{B_BOOT-1}]"
            for metric in ("energy", "mmd_u"):
                for sc in scopes:
                    raw = (point["energy"][sc] if metric == "energy"
                           else point["mmd_u"][sc][1.0])
                    bv = np.asarray(boots[(metric, sc)], float)
                    rows.append({
                        "dataset": name, "attack": a, "metric": metric,
                        "scope": sc,
                        "D": float(max(raw, 0.0)), "D_raw": float(raw),
                        "ci_lo": float(np.quantile(bv, CI_LO_Q)) if bv.size else np.nan,
                        "ci_hi": float(np.quantile(bv, CI_HI_Q)) if bv.size else np.nan,
                        "n_boot_valid": int(bv.size),
                        "rank_in_dataset": np.nan,  # filled after the loop
                        "bandwidth": (point["sigma"][sc] if metric == "mmd_u"
                                      else np.nan),
                        "n_folds": N_FOLDS,
                        "n_attack_windows": int(round(point["n_attack"])),
                        "n_benign_windows": int(round(point["n_benign"])),
                        "n_groups": int(len(atk_state["ids"])),
                        "n_benign_groups": int(len(ben_state["ids"])),
                        "seeds": seeds_str,
                    })
                    # bandwidth sensitivity rows (mmd_u only)
                    if metric == "mmd_u":
                        for m in BW_MULTS:
                            r_m = point["mmd_u"][sc][m]
                            sens_rows.append({
                                "dataset": name, "attack": a, "scope": sc,
                                "bw_multiplier": m,
                                "sigma_used": point["sigma"][sc] * m,
                                "mmd_u_raw": float(r_m),
                                "mmd_u": float(max(r_m, 0.0)),
                            })
            point_details[f"{name}/{a}"] = {
                "n_point_evals": len(evs),
                "sigma_full": point["sigma"]["full"],
                "sigma_phys": point["sigma"]["phys"],
            }
            e_f = point["energy"]["full"]
            m_f = point["mmd_u"]["full"][1.0]
            print(f"  {a:26s} energy_full={e_f:9.4f}  mmd_u_full={m_f:9.6f} "
                  f"(sigma={point['sigma']['full']:.3f})  "
                  f"nA={int(point['n_attack'])} nB={int(point['n_benign'])} "
                  f"[{time.time()-t0:6.1f}s]", flush=True)

    df = pd.DataFrame(rows)
    # within-dataset ranks (ascending: 1 = smallest D) per metric x scope
    for (dsn, met, sc), sub in df.groupby(["dataset", "metric", "scope"]):
        df.loc[sub.index, "rank_in_dataset"] = rankdata(sub["D"], method="average")
    df.to_csv(OUT / "divergence.csv", index=False)
    sens = pd.DataFrame(sens_rows)
    sens.to_csv(OUT / "divergence_bandwidth_sensitivity.csv", index=False)

    # ------------------------- sanity + headline -----------------------------
    def _spear(x, yv):
        if len(x) < 4:
            return {"n": int(len(x)), "note": "too few points"}
        rho, p = spearmanr(x, yv)
        return {"n": int(len(x)), "rho": float(rho), "p": float(p)}

    published_energy_rho = {
        dsn: pub_val["screens"]["D_full"]["per_dataset"].get(dsn, {})
        for dsn in df.dataset.unique()
    }

    sanity, consistency, ci_widths, bw_stability = {}, {}, {}, {}
    for dsn, sub_ds in df.groupby("dataset"):
        sanity[dsn] = {}
        full = sub_ds[sub_ds.scope == "full"]
        for met, sub in full.groupby("metric"):
            yv = [pub_Y.get((dsn, a), np.nan) for a in sub["attack"]]
            ok = np.isfinite(yv)
            sanity[dsn][met] = {
                "spearman_Dfull_vs_publishedY": _spear(
                    sub["D"].to_numpy()[ok], np.asarray(yv)[ok]),
                "spearman_Dphys_vs_publishedY": _spear(
                    sub_ds[(sub_ds.scope == "phys")
                           & (sub_ds.metric == met)]["D"].to_numpy()[ok],
                    np.asarray(yv)[ok]),
            }
        # new grouped energy vs published (ungrouped, all-data-standardised) energy
        e_full = full[full.metric == "energy"]
        old = [pub_energy_D.get((dsn, a), np.nan) for a in e_full["attack"]]
        sanity[dsn]["spearman_new_energy_vs_published_energy_Dfull"] = _spear(
            e_full["D"].to_numpy(), np.asarray(old))
        # energy vs mmd_u ordering consistency (separately per scope)
        consistency[dsn] = {}
        for sc in ("full", "phys"):
            e = df[(df.dataset == dsn) & (df.scope == sc)
                   & (df.metric == "energy")].set_index("attack")["D"]
            m = df[(df.dataset == dsn) & (df.scope == sc)
                   & (df.metric == "mmd_u")].set_index("attack")["D"]
            common = e.index.intersection(m.index)
            consistency[dsn][sc] = _spear(e[common].to_numpy(), m[common].to_numpy())
        # bandwidth ordering stability on mmd_u D vectors (raw, per scope)
        bw_stability[dsn] = {}
        for sc in ("full", "phys"):
            piv = sens[(sens.dataset == dsn) & (sens.scope == sc)].pivot(
                index="attack", columns="bw_multiplier", values="mmd_u_raw")
            bw_stability[dsn][sc] = {
                f"{a}x_vs_{b}x": float(spearmanr(piv[a], piv[b])[0])
                for a, b in ((0.5, 1.0), (1.0, 2.0), (0.5, 2.0))
            }
    for met, sub in df.groupby("metric"):
        w = (sub["ci_hi"] - sub["ci_lo"]).to_numpy()
        by_scope = {}
        for sc, s2 in sub.groupby("scope"):
            w2 = (s2["ci_hi"] - s2["ci_lo"]).to_numpy()
            by_scope[sc] = {"min": float(np.nanmin(w2)), "max": float(np.nanmax(w2))}
        ci_widths[met] = {"min": float(np.nanmin(w)), "max": float(np.nanmax(w)),
                          "by_scope": by_scope}

    neg_raw = df[(df.metric == "mmd_u") & (df.D_raw < 0)][
        ["dataset", "attack", "scope", "D_raw"]].to_dict("records")

    # point-set match vs the published frontier table
    got = {(r["dataset"], r["attack"]) for r in rows}
    expected = set(pub_Y.keys())
    match = {"points_computed": len(got),  # unique (dataset, attack) points
             "published_points": len(expected),
             "missing_vs_published": sorted(map(str, expected - got)),
             "extra_vs_published": sorted(map(str, got - expected))}

    protocol = {
        "divergences_reported_separately": True,
        "never_combined_note": ("energy and mmd_u are NEVER averaged or combined "
                                "into one number; each is its own CSV row."),
        "energy_estimator": ("V-statistic (biased) energy distance, identical to "
                             "published a1._energy: 2*mean(d(A,B)) - mean(d(A,A)) "
                             "- mean(d(B,B)); Euclidean d; means include the zero "
                             "diagonal; non-negative by construction."),
        "mmd_estimator": ("UNBIASED U-statistic RBF-MMD^2: "
                          "sum_{i!=j}k(a_i,a_j)/(m(m-1)) + "
                          "sum_{i!=j}k(b_i,b_j)/(n(n-1)) - 2/(mn) sum k(a_i,b_j); "
                          "k(x,y)=exp(-||x-y||^2/(2 sigma^2)); raw value kept in "
                          "D_raw (may be negative), D = max(D_raw, 0)."),
        "bandwidth_rule": ("median heuristic: sigma = median NONZERO pairwise "
                           "Euclidean distance among <=512 standardised benign "
                           "windows of the FITTING fold only (never the scored "
                           "fold, never attack windows); fallback sigma=1.0; "
                           "recomputed per scope (full/phys) on sliced columns."),
        "feature_scaling": ("per-feature clip to the benign FIT fold's "
                            "[q0.005, q0.995], then (x - mu)/sd with mu/sd of the "
                            "clipped fit fold (sd floor 1e-9) -- a1b's "
                            "_xfit_standardize refit on fold A only; attack and "
                            "scored-benign windows never influence the scaler."),
        "grouped_cross_fitting": ("benign transmitter groups split 50/50 into 2 "
                                  "folds; scaler+bandwidth fit on fold A benign "
                                  "only; divergence computed on fold-B benign + "
                                  "attack windows (both transformed by fold-A "
                                  "stats); roles swapped; attack draw shared by "
                                  "the two fold-orders."),
        "fold_count": N_FOLDS,
        "fold_aggregation": "arithmetic mean of the raw statistic over the 2 fold-orders",
        "point_estimate": (f"mean over {N_SEEDS} seeded repetitions (whole-group "
                           f"subsampling WITHOUT replacement, seed sequence "
                           f"[{BASE_SEED}, ds_idx, atk_idx, s])"),
        "subsampling": {"n_per_class_target": N_PER_CLASS, "n_fit_target": N_FIT,
                        "n_bandwidth_rows": N_BW,
                        "rule": ("whole transmitter groups accumulated until the "
                                 "target is reached (overshoot by at most one "
                                 "group; a1b._grouped_sample rule)")},
        "ci_method": (f"grouped percentile bootstrap: transmitter groups resampled "
                      f"WITH replacement (attack groups from the full pool; benign "
                      f"groups within each fold of a fresh 50/50 split), B={B_BOOT}, "
                      f"seed sequence [{BOOT_SEED}, ds_idx, atk_idx, b]; CI = "
                      f"[{CI_LO_Q}, {CI_HI_Q}] percentiles of the replicate RAW "
                      f"statistic (mmd_u CIs are on the raw, un-clamped value)."),
        "dataset_wise_normalisation": ("NONE across datasets: divergences are "
                                       "reported per dataset; rank_in_dataset "
                                       "(scipy rankdata, ascending, average ties) "
                                       "is provided for downstream pooled analyses "
                                       "(pooling itself deferred to correlations)."),
        "phys_scope": ("feature subset with controllability == 'phys' in "
                       "results/frontier/feature_controllability.csv for the "
                       "dataset's network_mode (dev_grid=rssi, nextgen=timing), "
                       "selected exactly as a1._controllability_for; same window "
                       "draws as full, columns sliced; own bandwidth."),
        "loaders": ("raw un-standardised windows, mirroring a1b._loaders: "
                    "dev_grid via qkv2x load_experiment_data(runs/dev_grid, "
                    "configs/features.yaml); nextgen via "
                    "results/veremi_nextgen/checkpoints/assembled.npz"),
        "attack_selection": (f"sorted attacks with y==1, excluding "
                             f"none/benign/empty, >= {MIN_ATTACK_WINDOWS} windows "
                             f"(a1's rule; reproduces the published 15 points)"),
        "published_baseline_context": {
            "published_D_source": "results/frontier/controllability.csv (energy only)",
            "published_pooled_rho": pub_val["screens"]["D_full"]["pooled"],
            "published_scaling_caveat": ("a1 standardised on ALL data (benign+attack "
                                         "pooled) BEFORE the divergence and subsampled "
                                         "random windows, not groups"),
        },
        "interpreter": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__,
    }

    headline = {
        "per_dataset_per_metric_spearman_Dfull_vs_Y": {
            dsn: {met: sanity[dsn][met]["spearman_Dfull_vs_publishedY"]
                  for met in ("energy", "mmd_u")}
            for dsn in sanity
        },
        "published_energy_only_spearman": published_energy_rho,
        "energy_vs_mmdu_rank_consistency": consistency,
        "ci_widths": ci_widths,
        "bandwidth_ordering_stability_spearman": bw_stability,
        "negative_raw_mmd_points": neg_raw,
        "point_set_vs_published": match,
    }

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - t0, 1),
        "item": "divergence_reestimation",
        "protocol": protocol,
        "sanity": sanity,
        "headline": headline,
        "point_details": point_details,
    }
    (OUT / "divergence_summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=float))

    # ------------------------------ report -----------------------------------
    pd.set_option("display.width", 200)
    print("\n" + "=" * 100)
    print("divergence -- energy and MMD_u reported SEPARATELY (full scope, point estimates + grouped-bootstrap CI)")
    print("=" * 100)
    show = df[df.scope == "full"][["dataset", "attack", "metric", "D", "D_raw",
                                   "ci_lo", "ci_hi", "rank_in_dataset",
                                   "bandwidth", "n_attack_windows",
                                   "n_benign_windows", "n_groups"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.5g}"))
    print("\nSanity: Spearman(D_full, published Y_rocauc) per dataset per metric")
    for dsn in sanity:
        old = published_energy_rho.get(dsn, {})
        old_rho = old.get("spearman_rho")
        for met in ("energy", "mmd_u"):
            sp = sanity[dsn][met]["spearman_Dfull_vs_publishedY"]
            print(f"  {dsn:9s} {met:7s} rho={sp.get('rho', float('nan')):+.3f} "
                  f"(p={sp.get('p', float('nan')):.4f}, n={sp.get('n')})"
                  + (f"   [published energy-only rho={old_rho:+.3f}]"
                     if met == "energy" and old_rho is not None else ""))
        cons = consistency[dsn]["full"]
        print(f"  {dsn:9s} energy-vs-mmd_u rank consistency (full): "
              f"rho={cons.get('rho', float('nan')):+.3f}")
    print("\nBandwidth ordering stability (Spearman of raw mmd_u vectors):")
    for dsn, sc_d in bw_stability.items():
        print(f"  {dsn:9s} full: {sc_d['full']}")
    print(f"\nnegative raw MMD_u points: {len(neg_raw)}")
    print(f"wrote {OUT / 'divergence.csv'}")
    print(f"wrote {OUT / 'divergence_bandwidth_sensitivity.csv'}")
    print(f"wrote {OUT / 'divergence_summary.json'}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
