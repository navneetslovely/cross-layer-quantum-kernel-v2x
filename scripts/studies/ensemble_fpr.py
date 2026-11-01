"""ensemble_fpr.py -- ensembles calibrated to a matched OVERALL false-positive rate.

Receiver-fusion and ensemble comparisons need a matched overall false-positive rate:
the union of six detectors, each set to 1% FPR, will usually exceed 1% FPR overall.
This script calibrates the full ensemble to 1% overall FPR before comparing it with a
single QSVM, and reports the communication overhead, latency, trust assumptions and
coordination cost of 20-second cross-receiver fusion.

Consumes results/studies/threshold_protocol_scores.parquet (validation + test scores per
dataset x model, produced by threshold_protocol.py). All calibration happens
on VALIDATION only; thresholds are applied unchanged to TEST; realised test FPR is
reported next to every recall.

Ensembles per dataset (over all models with scores):
  naive_union_1pc : per-model thresholds each at 1% FPR (validation) -- the
                    uncalibrated setting; reported to show the
                    realised ensemble FPR inflation.
  union_matched   : one common per-model FPR level alpha found by bisection on
                    validation so the UNION FPR = 1%; applied to test.
  rank_mean       : mean of per-model rank-transformed scores (rank calibrated on
                    validation benign; test scores ranked against the validation
                    reference), threshold at 1% FPR on validation fused scores.
  single QSVM     : qsvm_crosslayer at validation-calibrated 1% FPR (comparator).

Run: python scripts/studies/ensemble_fpr.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "studies"
TARGET_FPR = 0.01


def _thr_at_fpr(benign_scores: np.ndarray, fpr: float) -> float:
    """Positive iff score > thr; benign quantile convention (matches threshold_protocol)."""
    return float(np.quantile(benign_scores, 1.0 - fpr))


def _rates(y, flags):
    y = np.asarray(y).astype(int)
    flags = np.asarray(flags).astype(bool)
    fpr = float(flags[y == 0].mean()) if (y == 0).any() else np.nan
    rec = float(flags[y == 1].mean()) if (y == 1).any() else np.nan
    prec = float(y[flags].mean()) if flags.any() else np.nan
    return fpr, rec, prec


def _union_flags(S, thr_vec):
    return (S > thr_vec[None, :]).any(axis=1)


def main() -> None:
    sc = pd.read_parquet(OUT / "threshold_protocol_scores.parquet")
    rows, acct = [], {}
    for ds, g in sc.groupby("dataset"):
        models = sorted(g.model.unique())
        va = g[g.split == "va"].pivot_table(index=["group", "t_end"],
                                            columns="model", values="score")
        te = g[g.split == "te"].pivot_table(index=["group", "t_end"],
                                            columns="model", values="score")
        yva = (g[g.split == "va"].groupby(["group", "t_end"]).y.first()
               .reindex(va.index).to_numpy().astype(int))
        yte = (g[g.split == "te"].groupby(["group", "t_end"]).y.first()
               .reindex(te.index).to_numpy().astype(int))
        va, te = va[models], te[models]
        assert not va.isna().any().any() and not te.isna().any().any(), \
            f"{ds}: model score matrix has holes"
        Sva, Ste = va.to_numpy(), te.to_numpy()

        # per-model validation thresholds at target FPR
        thr1 = np.array([_thr_at_fpr(Sva[yva == 0, j], TARGET_FPR)
                         for j in range(len(models))])

        # (a) naive union of per-model 1% thresholds
        for split, S, y in [("va", Sva, yva), ("te", Ste, yte)]:
            fpr, rec, prec = _rates(y, _union_flags(S, thr1))
            rows.append({"dataset": ds, "ensemble": "naive_union_1pc",
                         "split": split, "n_models": len(models),
                         "fpr": fpr, "recall": rec, "precision": prec})

        # (b) union calibrated to 1% ENSEMBLE FPR on validation (bisection on alpha)
        lo, hi = 1e-5, TARGET_FPR
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            thr = np.array([_thr_at_fpr(Sva[yva == 0, j], mid)
                            for j in range(len(models))])
            f, _, _ = _rates(yva, _union_flags(Sva, thr))
            if f > TARGET_FPR:
                hi = mid
            else:
                lo = mid
        alpha = lo
        thr_m = np.array([_thr_at_fpr(Sva[yva == 0, j], alpha)
                          for j in range(len(models))])
        for split, S, y in [("va", Sva, yva), ("te", Ste, yte)]:
            fpr, rec, prec = _rates(y, _union_flags(S, thr_m))
            rows.append({"dataset": ds, "ensemble": "union_matched_1pc",
                         "split": split, "n_models": len(models),
                         "per_model_alpha": alpha,
                         "fpr": fpr, "recall": rec, "precision": prec})

        # (c) rank-mean fusion, calibrated on validation
        ref = [np.sort(Sva[:, j]) for j in range(len(models))]
        def _fuse(S):
            R = np.stack([np.searchsorted(ref[j], S[:, j]) / len(ref[j])
                          for j in range(len(models))], axis=1)
            return R.mean(axis=1)
        fva, fte = _fuse(Sva), _fuse(Ste)
        thr_f = _thr_at_fpr(fva[yva == 0], TARGET_FPR)
        for split, f_, y in [("va", fva, yva), ("te", fte, yte)]:
            fpr, rec, prec = _rates(y, f_ > thr_f)
            rows.append({"dataset": ds, "ensemble": "rank_mean_1pc",
                         "split": split, "n_models": len(models),
                         "fpr": fpr, "recall": rec, "precision": prec})

        # (d) single-QSVM comparator
        if "qsvm_crosslayer" in models:
            j = models.index("qsvm_crosslayer")
            thr_q = _thr_at_fpr(Sva[yva == 0, j], TARGET_FPR)
            for split, S, y in [("va", Sva, yva), ("te", Ste, yte)]:
                fpr, rec, prec = _rates(y, S[:, j] > thr_q)
                rows.append({"dataset": ds, "ensemble": "single_qsvm_1pc",
                             "split": split, "n_models": 1,
                             "fpr": fpr, "recall": rec, "precision": prec})

        acct[ds] = {"models": models, "n_va": int(len(yva)), "n_te": int(len(yte)),
                    "alpha_matched": alpha}

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ensemble_fpr.csv", index=False)
    print(df[df.split == "te"].round(4).to_string(index=False))

    # cross-receiver fusion cost accounting (overhead, latency, trust, coordination)
    fusion_cost = {
        "setting": "A5 cross-receiver fusion (scripts/crossreceiver_fusion.py): windows "
                   "of one claimed sender pooled across receivers per 20 s time bin",
        "communication_overhead": "each receiver reports one 19-float window "
                                  "feature vector (+ ids) per observed sender per "
                                  "20 s bin ~= 19*4 B + 16 B ids ~= 92 B; at 50 "
                                  "senders/receiver that is ~4.6 kB per receiver "
                                  "per bin (~0.23 kB/s) -- negligible vs 10 Hz BSM "
                                  "traffic (~300 B/beacon ~= 3 kB/s per sender)",
        "latency": "decision latency is bounded by the bin close: up to 20 s + "
                   "backhaul; unsuitable for pre-crash reaction, aimed at "
                   "revocation/misbehaviour-authority reporting timescales",
        "trust_assumptions": "the aggregator (RSU/edge) is honest; receiver "
                             "reports are authenticated; a compromised reporting "
                             "receiver can poison fused features (not modelled)",
        "coordination_cost": "receivers need ~1 s-level time sync for bin "
                             "alignment (GNSS time suffices); no inter-receiver "
                             "channel needed (star topology to the aggregator)",
    }
    summary = {"target_fpr": TARGET_FPR, "per_dataset": acct,
               "fusion_cost_accounting": fusion_cost,
               "table": rows}
    (OUT / "ensemble_fpr_summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=float))
    print(json.dumps(acct, indent=1))


if __name__ == "__main__":
    main()
