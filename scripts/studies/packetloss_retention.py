#!/usr/bin/env python3
"""packetloss_retention -- packet-loss retention + failures variant.

The packet-loss study has a survivorship bias: a window needs at least five received
messages, so windows disappear as loss grows. This script reports the retention rate at
every loss level, separately for positive and negative windows, the attack-family
composition and the retained transmitter count; adds a variant in which missing windows
count as failures; and provides both ROC-AUC and PR-AUC loss curves.

Source data (READ-ONLY, never modified):
    results/packetloss_dev/predictions.parquet   392,880 rows x 9 cols
        [model, packet_loss, i, y_true, score, y_pred, attack_name, group, t_end]
        = per-window TEST-set predictions for the 5 matched-N models at each loss
        level p in {0.0, 0.2, 0.4, 0.6, 0.8}.
    results/packetloss_dev/packetloss_results.csv  per-(model, p) metrics incl. BOTH
        pr_auc and roc_auc (+ grouped-bootstrap CIs for the matched models).

VERIFIED DATA FACTS THAT SHAPE THE DESIGN (all re-checked in this script; the run
aborts if any stops holding):

  (V1) Within a loss level the window population is MODEL-INVARIANT: for every level,
       all 5 models carry identical (i -> y_true/attack_name/group/t_end) metadata,
       so one model's rows describe the level's test population. (Also: the
       (group, t_end) key has 1 accidental collision at p=0.0 and 1 at p=0.2 --
       two distinct windows from different RSU receiver streams sharing a sender and
       a last-message time; rx_node is not stored in the parquet. All counting here
       is therefore ROW-based, never key-set-based.)

  (V2) Across loss levels the window populations are NOT nested. Two reasons, both
       verified quantitatively (numbers recorded in packetloss_retention_summary.json):
         (a) windows are W consecutive MESSAGES of an (rx, sender) stream
             (src/qkv2x/features/dataset.py); dropping messages re-windows the
             stream, so t_end shifts and window IDENTITY changes -- ~96% of p>0
             (group, t_end) keys do not exist at p=0;
         (b) the pipeline re-runs GroupShuffleSplit on each level's re-built window
             array (src/qkv2x/experiments/packetloss.py + common.py::group_split),
             so the TEST TRANSMITTER SET is re-randomised per level -- only ~30-37%
             of test groups at p>0 are also test groups at p=0.
       Consequently a fixed population defined as "windows present at
       p=0 keyed by (group, t_end)" is INFEASIBLE on this artefact. Exact
       window-level cross-level matching is IRRECOVERABLE from predictions.parquet
       (no t_start / rx_node columns; and window identity is message-index-based,
       not time-grid-based). Scores for level-p windows of transmitters that fell
       into the level-p train/val split are also IRRECOVERABLE (predictions cover
       the test split only).

  REPLACEMENT DESIGN (failures variant): for each p>0 restrict to the
  INTERSECTION FLEET G_cap(p) = {test transmitters at p=0} n {test transmitters at p}
  (model-invariant by V1). Fixed population = the p=0 test windows of G_cap(p).
  Failure accounting is PER-GROUP and COUNT-BASED:
      absent_pos(p) = sum_g max(0, n_pos_p0(g) - n_pos_p(g))
  i.e. every positive window a transmitter had at p=0 but no longer has at p is a
  MISS (it never formed, so it can never raise an alarm). Verified: per-group TOTAL
  window counts never increase under thinning; positive counts increase for at most
  2 groups by 1 window each (label re-composition; excess counted and reported).
  Absent NEGATIVES are dropped, not scored: a window that never formed cannot raise
  a false alarm, so for PR/recall purposes dropping it is equivalent to giving it a
  score of -infinity (it never crosses any positive threshold); counting it as a
  "correct rejection" would be an unsupported claim in the other direction.

  Operating point: threshold at 1% FPR on the PRESENT negatives of the population
  being scored (rule: allowed = floor(0.01 * n_neg); thr = (allowed+1)-th largest
  negative score; detect iff score > thr -- guarantees realised FPR <= 1% under ties;
  deterministic). Reported per (model, p>0):
    recall_fixedpop        = detected present positives of G_cap / ALL p=0 positives
                             of G_cap  (absent positives count as misses)
    recall_survivors_common= detected present positives of G_cap / present positives
                             of G_cap  (same numerator; survivors-only denominator ->
                             the bias magnitude is the gap between the two)
    recall_survivors_fulltest = survivors-only recall@1%FPR on the level's FULL test
                             set (published-style comparator, own threshold)
    ap_conservative_fixedpop = average precision with the absent positives appended
                             at a worst-score sentinel (min present score - 1, i.e.
                             ranked below everything; finite stand-in for -inf)
    ap_survivors_common / ap_fulltest = survivors-only APs for comparison.
  Grouped bootstrap CIs (resample G_cap transmitters with replacement, threshold
  re-derived inside every replicate) for the three headline quantities.

RETENTION (part 1) is reported in three tiers, clearly labelled:
    retention_full  n_windows(p)/n_windows(0) from packetloss_results.csv --
                        the FULL windowed dataset (train+val+test), overall only;
    retention_test  n_test(p)/n_test(0) etc. from predictions.parquet -- a
                        TEST-SPLIT SIZE RATIO (splits re-drawn per level, see V2b),
                        overall / positives / negatives + attack composition +
                        retained transmitter counts;
    retention_nested  windows_p(G_cap)/windows_p0(G_cap) -- TRUE retention on the
                        fixed intersection fleet, overall / positives / negatives.

PR-AUC EXISTENCE CHECK (part 3): pr_auc (+CI) per level DOES exist in
packetloss_results.csv, and results/packetloss_dev/pr_auc_vs_packetloss.{png,pdf}
were already rendered by the original run, so a PR-vs-loss figure needs no new
data. packetloss_curves.csv carries both curves (CSV values cross-checked
against AUCs recomputed from predictions.parquet).

Determinism: the only stochastic step is the bootstrap (numpy default_rng(20260807)).
Everything else is exact arithmetic on the frozen artefact.

Usage:
    python scripts/studies/packetloss_retention.py

Outputs (results/studies/):
    packetloss_retention.csv  per-level retention tiers + attack composition + groups
    packetloss_failures_variant.csv  per (model, p) failures-variant vs survivors-only
    packetloss_curves.csv  ROC-AUC + PR-AUC vs loss per model (survivors-only)
    packetloss_retention_summary.json  protocol, verified checks, headline numbers
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

SRC = REPO / "results" / "packetloss_dev"
OUT = REPO / "results" / "studies"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20260807
N_BOOT = 1000
FPR_TARGET = 0.01
REF_MODEL = "qsvm_crosslayer"          # any model works after the V1 check passes
MODELS = ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest", "mlp"]
P0 = 0.0

ATTACKS = ["const_offset", "const_pos", "data_falsification", "drift",
           "eventual_stop", "random_offset", "random_pos", "replay", "sybil"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def thr_at_fpr(neg_scores: np.ndarray, target: float = FPR_TARGET) -> float:
    """Deterministic threshold: detect iff score > thr, guaranteeing FPR <= target.

    allowed = floor(target * n_neg); thr = the (allowed+1)-th largest negative
    score. Then #(neg > thr) <= allowed even under ties (ties AT thr don't fire).
    """
    n = len(neg_scores)
    if n == 0:
        return float("nan")
    allowed = int(np.floor(target * n))
    desc = np.sort(np.asarray(neg_scores, dtype=float))[::-1]
    idx = min(allowed, n - 1)
    return float(desc[idx])


def ap_with_absent_pos(pos_scores: np.ndarray, neg_scores: np.ndarray,
                       n_absent_pos: int) -> float:
    """Average precision with n_absent_pos synthetic positives at a worst-score
    sentinel (min present score - 1 => ranked strictly below every real score;
    finite stand-in for -inf). Absent negatives are ALREADY excluded upstream --
    a never-formed window cannot false-positive (equivalent to scoring it -inf)."""
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    if len(pos) + n_absent_pos == 0:
        return float("nan")
    if n_absent_pos > 0:
        present = np.concatenate([pos, neg]) if len(pos) + len(neg) else np.array([0.0])
        sentinel = float(np.min(present)) - 1.0
        pos = np.concatenate([pos, np.full(n_absent_pos, sentinel)])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    s = np.concatenate([pos, neg])
    return float(average_precision_score(y, s))


def recall_pair(pos_scores: np.ndarray, neg_scores: np.ndarray,
                n_pos_fixed: int) -> tuple[float, float, float, float]:
    """(recall_fixedpop, recall_survivors, thr, realised_fpr) at 1% FPR on the
    given present negatives. Numerator identical for both recalls."""
    thr = thr_at_fpr(neg_scores)
    if not np.isfinite(thr):
        return float("nan"), float("nan"), thr, float("nan")
    det = int(np.sum(np.asarray(pos_scores, dtype=float) > thr))
    r_fix = det / n_pos_fixed if n_pos_fixed > 0 else float("nan")
    r_sur = det / len(pos_scores) if len(pos_scores) > 0 else float("nan")
    fpr = float(np.mean(np.asarray(neg_scores, dtype=float) > thr))
    return r_fix, r_sur, thr, fpr


def grouped_bootstrap(groups: list, per_group: dict, n_boot: int, rng) -> dict:
    """Percentile CIs, resampling transmitters with replacement. per_group maps
    g -> (pos_scores, neg_scores, n_pos_fixed, n_absent_pos). The 1%-FPR threshold
    is re-derived inside every replicate."""
    stats = {"recall_fixedpop": [], "recall_survivors_common": [],
             "ap_conservative_fixedpop": []}
    garr = np.asarray(groups, dtype=object)
    for _ in range(n_boot):
        take = rng.integers(0, len(garr), size=len(garr))
        pos_l, neg_l, nfix, nabs = [], [], 0, 0
        for gi in take:
            ps, ns, nf, na = per_group[garr[gi]]
            pos_l.append(ps)
            neg_l.append(ns)
            nfix += nf
            nabs += na
        pos = np.concatenate(pos_l) if pos_l else np.array([])
        neg = np.concatenate(neg_l) if neg_l else np.array([])
        r_fix, r_sur, _, _ = recall_pair(pos, neg, nfix)
        stats["recall_fixedpop"].append(r_fix)
        stats["recall_survivors_common"].append(r_sur)
        stats["ap_conservative_fixedpop"].append(ap_with_absent_pos(pos, neg, nabs))
    out = {}
    for k, v in stats.items():
        arr = np.asarray(v, dtype=float)
        arr = arr[np.isfinite(arr)]
        out[k] = ({"lo": float(np.percentile(arr, 2.5)),
                   "hi": float(np.percentile(arr, 97.5))}
                  if len(arr) else {"lo": float("nan"), "hi": float("nan")})
    return out


# --------------------------------------------------------------------------- #
# verification (V1/V2) -- abort on failure
# --------------------------------------------------------------------------- #
def verify_facts(df: pd.DataFrame) -> dict:
    checks = {}
    levels = sorted(df["packet_loss"].unique())
    checks["levels"] = [float(p) for p in levels]

    # V1: (p, i)-aligned metadata identical across models
    ref = df[df.model == REF_MODEL].set_index(["packet_loss", "i"]).sort_index()
    inv = True
    for m in MODELS:
        if m == REF_MODEL:
            continue
        o = df[df.model == m].set_index(["packet_loss", "i"]).sort_index()
        inv &= (o["y_true"].equals(ref["y_true"])
                and o["group"].equals(ref["group"])
                and o["attack_name"].equals(ref["attack_name"])
                and bool(np.allclose(o["t_end"].to_numpy(),
                                     ref["t_end"].to_numpy())))
    checks["v1_window_population_model_invariant"] = bool(inv)
    if not inv:
        raise SystemExit("FATAL: window populations differ across models (V1).")

    # label semantics: y==0 <=> attack_name=='none'
    d = df[df.model == REF_MODEL]
    bad = int((((d.y_true == 0) & (d.attack_name != "none"))
               | ((d.y_true == 1) & (d.attack_name == "none"))).sum())
    checks["label_iff_attackname_violations"] = bad
    if bad:
        raise SystemExit("FATAL: y_true<->attack_name mismatch.")

    # key-collision census (informational: counting is row-based)
    coll = {}
    for p in levels:
        sub = d[d.packet_loss == p]
        coll[str(p)] = int(sub.duplicated(subset=["group", "t_end"]).sum())
    checks["group_tend_key_collisions_per_level"] = coll

    # V2a: cross-level key non-nesting
    keys = {p: set(zip(d[d.packet_loss == p]["group"],
                       d[d.packet_loss == p]["t_end"])) for p in levels}
    checks["v2a_keys_at_p_not_in_p0_frac"] = {
        str(p): round(len(keys[p] - keys[P0]) / len(keys[p]), 4)
        for p in levels if p > P0}

    # V2b: test-transmitter turnover
    gsets = {p: set(d[d.packet_loss == p]["group"]) for p in levels}
    checks["v2b_test_group_overlap_with_p0"] = {
        str(p): {"n_groups": len(gsets[p]),
                 "n_common_with_p0": len(gsets[p] & gsets[P0])}
        for p in levels}

    # per-group monotonicity on the intersection fleet
    mono = {}
    a = d[d.packet_loss == P0]
    for p in levels:
        if p <= P0:
            continue
        b = d[d.packet_loss == p]
        common = gsets[p] & gsets[P0]
        j = (a[a.group.isin(common)].groupby("group")["y_true"]
             .agg(pos0="sum", tot0="count")
             .join(b[b.group.isin(common)].groupby("group")["y_true"]
                   .agg(posp="sum", totp="count")))
        mono[str(p)] = {
            "groups_with_more_total_windows_at_p": int((j.totp > j.tot0).sum()),
            "groups_with_more_pos_windows_at_p": int((j.posp > j.pos0).sum()),
            "excess_pos_windows": int((j.posp - j.pos0).clip(lower=0).sum()),
        }
        if mono[str(p)]["groups_with_more_total_windows_at_p"] > 0:
            raise SystemExit("FATAL: per-group total window count grew under "
                             "thinning -- design assumption broken.")
    checks["pergroup_monotonicity_on_intersection"] = mono
    return checks


# --------------------------------------------------------------------------- #
# part 1: retention
# --------------------------------------------------------------------------- #
def retention_table(df: pd.DataFrame, res_csv: pd.DataFrame):
    d = df[df.model == REF_MODEL]
    levels = sorted(d["packet_loss"].unique())
    base = d[d.packet_loss == P0]
    g0 = set(base["group"])
    n_full = {float(r.packet_loss): int(r.n_windows)
              for r in res_csv[res_csv.model == REF_MODEL].itertuples()}
    rows, absent_by_attack = [], {}
    for p in levels:
        sub = d[d.packet_loss == p]
        npos, nneg = int((sub.y_true == 1).sum()), int((sub.y_true == 0).sum())
        gp = set(sub["group"])
        common = gp & g0
        b0 = base[base.group.isin(common)]
        bp = sub[sub.group.isin(common)]
        row = {
            "packet_loss": float(p),
            # tier 1: full windowed dataset (train+val+test), from results csv
            "n_windows_full": n_full[float(p)],
            "retention_full": round(n_full[float(p)] / n_full[P0], 4),
            # tier 2: test split (re-drawn per level -> size ratio, not nested)
            "n_test": len(sub), "n_test_pos": npos, "n_test_neg": nneg,
            "ratio_test": round(len(sub) / len(base), 4),
            "ratio_test_pos": round(npos / int((base.y_true == 1).sum()), 4),
            "ratio_test_neg": round(nneg / int((base.y_true == 0).sum()), 4),
            "n_groups": len(gp),
            "n_groups_common_p0": len(common),
            # tier 3: TRUE retention on the fixed intersection fleet
            "retention_nested": round(len(bp) / len(b0), 4) if len(b0) else np.nan,
            "retention_nested_pos": (round(int((bp.y_true == 1).sum())
                                           / int((b0.y_true == 1).sum()), 4)
                                     if int((b0.y_true == 1).sum()) else np.nan),
            "retention_nested_neg": (round(int((bp.y_true == 0).sum())
                                           / int((b0.y_true == 0).sum()), 4)
                                     if int((b0.y_true == 0).sum()) else np.nan),
        }
        # attack-family composition of the level's test population
        vc = sub["attack_name"].value_counts()
        row["n_win_none"] = int(vc.get("none", 0))
        for a_ in ATTACKS:
            row[f"n_win_{a_}"] = int(vc.get(a_, 0))
            row[f"posshare_{a_}"] = round(int(vc.get(a_, 0)) / npos, 4) if npos else np.nan
        rows.append(row)

        # absent-positive composition on the intersection fleet (per-group clipped)
        if p > P0:
            j0 = (b0[b0.y_true == 1].groupby(["group", "attack_name"]).size()
                  .rename("n0"))
            jp = (bp[bp.y_true == 1].groupby(["group", "attack_name"]).size()
                  .rename("np_"))
            jj = pd.concat([j0, jp], axis=1).fillna(0)
            jj["absent"] = (jj.n0 - jj.np_).clip(lower=0)
            absent_by_attack[str(float(p))] = {
                a_: int(v) for a_, v in
                jj.groupby(level="attack_name")["absent"].sum().items() if v > 0}
    return pd.DataFrame(rows), absent_by_attack


# --------------------------------------------------------------------------- #
# part 2: failures variant
# --------------------------------------------------------------------------- #
def failures_variant(df: pd.DataFrame, rng) -> pd.DataFrame:
    d0meta = df[(df.model == REF_MODEL) & (df.packet_loss == P0)]
    g0 = set(d0meta["group"])
    levels = sorted(df["packet_loss"].unique())
    rows = []
    for p in levels:
        meta_p = df[(df.model == REF_MODEL) & (df.packet_loss == p)]
        common = sorted(set(meta_p["group"]) & g0)
        pos0_by_g = (d0meta[d0meta.group.isin(common)]
                     .groupby("group")["y_true"].sum().astype(int))
        for model in MODELS:
            dm = df[(df.model == model) & (df.packet_loss == p)]
            dc = dm[dm.group.isin(common)]
            # per-group present scores + fixed/absent counts
            per_group = {}
            n_pos_fixed = n_absent = n_excess = 0
            for g in common:
                sub = dc[dc.group == g]
                ps = sub.loc[sub.y_true == 1, "score"].to_numpy(dtype=float)
                ns = sub.loc[sub.y_true == 0, "score"].to_numpy(dtype=float)
                nf = int(pos0_by_g.get(g, 0))
                na = max(0, nf - len(ps))
                n_excess += max(0, len(ps) - nf)
                per_group[g] = (ps, ns, nf, na)
                n_pos_fixed += nf
                n_absent += na
            pos = np.concatenate([per_group[g][0] for g in common])
            neg = np.concatenate([per_group[g][1] for g in common])
            r_fix, r_sur, thr, fpr = recall_pair(pos, neg, n_pos_fixed)
            ap_cons = ap_with_absent_pos(pos, neg, n_absent)
            ap_sur = ap_with_absent_pos(pos, neg, 0)
            # published-style comparator on the FULL level-p test set
            posF = dm.loc[dm.y_true == 1, "score"].to_numpy(dtype=float)
            negF = dm.loc[dm.y_true == 0, "score"].to_numpy(dtype=float)
            _, r_full, thr_full, _ = recall_pair(posF, negF, len(posF))
            ap_full = float(average_precision_score(
                dm["y_true"].to_numpy(), dm["score"].to_numpy()))
            ci = grouped_bootstrap(common, per_group, N_BOOT, rng)
            rows.append({
                "model": model, "packet_loss": float(p),
                "n_common_groups": len(common),
                "n_pos_fixed": n_pos_fixed, "n_pos_present": int(len(pos)),
                "n_absent_pos": n_absent, "n_excess_pos": n_excess,
                "n_neg_present": int(len(neg)),
                "thr_1fpr_common": thr, "realized_fpr_common": round(fpr, 5),
                "detected_present_pos": int(np.sum(pos > thr)),
                "recall_fixedpop": round(r_fix, 4),
                "recall_fixedpop_lo": round(ci["recall_fixedpop"]["lo"], 4),
                "recall_fixedpop_hi": round(ci["recall_fixedpop"]["hi"], 4),
                "recall_survivors_common": round(r_sur, 4),
                "recall_survivors_common_lo":
                    round(ci["recall_survivors_common"]["lo"], 4),
                "recall_survivors_common_hi":
                    round(ci["recall_survivors_common"]["hi"], 4),
                "recall_bias_gap": round(r_sur - r_fix, 4),
                "recall_survivors_fulltest": round(r_full, 4),
                "ap_conservative_fixedpop": round(ap_cons, 4),
                "ap_conservative_fixedpop_lo":
                    round(ci["ap_conservative_fixedpop"]["lo"], 4),
                "ap_conservative_fixedpop_hi":
                    round(ci["ap_conservative_fixedpop"]["hi"], 4),
                "ap_survivors_common": round(ap_sur, 4),
                "ap_bias_gap": round(ap_sur - ap_cons, 4),
                "ap_fulltest": round(ap_full, 4),
            })
            print(f"[packetloss_retention] p={p:.1f} {model:16s} fleet={len(common):3d} "
                  f"fix={n_pos_fixed:5d} present={len(pos):5d} absent={n_absent:5d} "
                  f"R_fix={r_fix:.3f} R_sur={r_sur:.3f} AP_cons={ap_cons:.3f}",
                  file=sys.stderr)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# part 3: curves (survivors-only) + cross-check
# --------------------------------------------------------------------------- #
def curves_table(df: pd.DataFrame, res_csv: pd.DataFrame):
    rows = []
    max_dr = max_dp = 0.0
    for r in res_csv.itertuples():
        row = {"model": r.model, "packet_loss": float(r.packet_loss),
               "roc_auc": r.roc_auc, "roc_auc_lo": r.roc_auc_lo,
               "roc_auc_hi": r.roc_auc_hi,
               "pr_auc": r.pr_auc, "pr_auc_lo": r.pr_auc_lo,
               "pr_auc_hi": r.pr_auc_hi,
               "n_test": int(r.n_test), "n_pos": int(r.n_pos),
               "source": "packetloss_results.csv",
               "roc_auc_recomputed": np.nan, "pr_auc_recomputed": np.nan}
        if r.model in MODELS:
            dm = df[(df.model == r.model) & (df.packet_loss == r.packet_loss)]
            roc = float(roc_auc_score(dm["y_true"], dm["score"]))
            ap = float(average_precision_score(dm["y_true"], dm["score"]))
            row["roc_auc_recomputed"] = round(roc, 6)
            row["pr_auc_recomputed"] = round(ap, 6)
            max_dr = max(max_dr, abs(roc - r.roc_auc))
            max_dp = max(max_dp, abs(ap - r.pr_auc))
        rows.append(row)
    return pd.DataFrame(rows), max_dr, max_dp


# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    df = pd.read_parquet(SRC / "predictions.parquet")
    res_csv = pd.read_csv(SRC / "packetloss_results.csv")

    print("[packetloss_retention] verifying data facts (V1/V2)...", file=sys.stderr)
    checks = verify_facts(df)

    print("[packetloss_retention] part 1: retention...", file=sys.stderr)
    ret, absent_by_attack = retention_table(df, res_csv)
    ret.to_csv(OUT / "packetloss_retention.csv", index=False)

    print("[packetloss_retention] part 2: failures variant (grouped bootstrap "
          f"B={N_BOOT}, seed={SEED})...", file=sys.stderr)
    fv = failures_variant(df, rng)
    fv.to_csv(OUT / "packetloss_failures_variant.csv", index=False)

    print("[packetloss_retention] part 3: curves + PR existence check...", file=sys.stderr)
    curves, max_dr, max_dp = curves_table(df, res_csv)
    curves.to_csv(OUT / "packetloss_curves.csv", index=False)
    pr_figs = sorted(p.name for p in SRC.glob("pr_auc_vs_packetloss.*"))

    def _cell(model, p):
        r = fv[(fv.model == model) & (fv.packet_loss == p)].iloc[0]
        return {"recall_fixedpop": float(r.recall_fixedpop),
                "recall_fixedpop_ci": [float(r.recall_fixedpop_lo),
                                       float(r.recall_fixedpop_hi)],
                "recall_survivors_common": float(r.recall_survivors_common),
                "recall_survivors_common_ci": [float(r.recall_survivors_common_lo),
                                              float(r.recall_survivors_common_hi)],
                "recall_survivors_fulltest": float(r.recall_survivors_fulltest),
                "bias_gap": float(r.recall_bias_gap),
                "ap_conservative_fixedpop": float(r.ap_conservative_fixedpop),
                "ap_survivors_common": float(r.ap_survivors_common),
                "n_pos_fixed": int(r.n_pos_fixed),
                "n_absent_pos": int(r.n_absent_pos)}

    def _ret(p):
        r = ret[ret.packet_loss == p].iloc[0]
        return {"retention_full_alllevels_windows": float(r.retention_full),
                "ratio_test_pos": float(r.ratio_test_pos),
                "ratio_test_neg": float(r.ratio_test_neg),
                "retention_nested_overall": float(r.retention_nested),
                "retention_nested_pos": float(r.retention_nested_pos),
                "retention_nested_neg": float(r.retention_nested_neg),
                "n_groups": int(r.n_groups),
                "n_groups_common_p0": int(r.n_groups_common_p0)}

    summary = {
        "item": "packetloss_retention", "script": "scripts/studies/packetloss_retention.py",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_time_s": None,  # filled below
        "protocol": {
            "source": {"predictions": str(SRC / "predictions.parquet"),
                       "results_csv": str(SRC / "packetloss_results.csv"),
                       "source_run_provenance": "results/packetloss_dev/provenance.json "
                                                "(2026-06-09, commit c8e6892, split seed 7, "
                                                "GroupShuffleSplit re-run per level)"},
            "bootstrap": {"n_boot": N_BOOT, "seed": SEED,
                          "unit": "transmitter (group), resampled with replacement; "
                                  "1%-FPR threshold re-derived inside every replicate",
                          "ci": "percentile [2.5, 97.5]"},
            "operating_point": {"rule": "allowed = floor(0.01*n_neg); thr = "
                                        "(allowed+1)-th largest present-negative score; "
                                        "detect iff score > thr (FPR <= 1% under ties)",
                                "fpr_target": FPR_TARGET},
            "fixed_population": "p=0 TEST windows of the per-level intersection fleet "
                                "G_cap(p) = test groups at p=0 AND at p. A "
                                "'(group,t_end) windows present at p=0' key is "
                                "infeasible: windows are message-index entities that "
                                "change identity under thinning (V2a) and the test "
                                "split is re-randomised per level (V2b).",
            "failure_accounting": "per-group count-based: absent_pos = "
                                  "sum_g max(0, n_pos_p0(g) - n_pos_p(g)); "
                                  "absent positives = misses (scored below every "
                                  "present window via sentinel min(present)-1 for the "
                                  "conservative AP).",
            "absent_negatives": "dropped, because a never-formed window cannot raise "
                                "a false alarm; for PR/recall purposes identical to "
                                "scoring it -inf (never crosses a positive threshold). "
                                "Counting it as a correct rejection would overclaim.",
            "population_stats_model": REF_MODEL + " (valid for all models by check V1)",
            "irrecoverable": {
                "exact_crosslevel_window_matching":
                    "IRRECOVERABLE -- predictions.parquet lacks t_start/rx_node and "
                    "window identity is message-index-based; re-windowed thinned "
                    "streams do not map 1:1 to p=0 windows.",
                "scores_for_p0_test_groups_resplit_into_train":
                    "IRRECOVERABLE -- predictions cover each level's own test split "
                    "only; ~63-70% of p=0 test transmitters land outside the level-p "
                    "test split (V2b), hence the intersection-fleet restriction."},
        },
        "verified_checks": checks,
        "retention": {"table": "packetloss_retention.csv",
                      "absent_pos_by_attack_on_intersection_fleet": absent_by_attack},
        "pr_auc_existence": {
            "pr_auc_in_packetloss_results_csv": True,
            "pr_auc_ci_columns_present": True,
            "pr_figure_files_already_rendered": pr_figs,
            "verdict": "PR-vs-loss data (and a rendered PR figure) already exist; "
                       "packetloss_curves.csv carries both curves for a 2-panel "
                       "ROC/PR figure."},
        "curves_crosscheck": {"max_abs_diff_roc_auc": round(max_dr, 6),
                              "max_abs_diff_pr_auc": round(max_dp, 6)},
        "headline": {
            "retention": {str(p): _ret(p) for p in (0.4, 0.6, 0.8)},
            "failures_vs_survivors": {
                m: {str(p): _cell(m, p) for p in (0.6, 0.8)}
                for m in ("qsvm_crosslayer", "rbf_svm")},
        },
    }
    summary["wall_time_s"] = round(time.time() - t0, 1)
    (OUT / "packetloss_retention_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[packetloss_retention] done in {summary['wall_time_s']}s -> "
          f"{OUT}/packetloss_retention_{{retention,failures_variant,curves}}.csv + packetloss_retention_summary.json",
          file=sys.stderr)


if __name__ == "__main__":
    main()
