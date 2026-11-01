"""mimicry_ci.py -- grouped-bootstrap confidence intervals on mimicry recall.

Adds uncertainty intervals to the adaptive-mimicry results: mimicry_ksweep.py reports
attack-wise outcomes, sample counts, the K-sweep and the transfer control as point
estimates of recall; this script attaches grouped-bootstrap CIs to them.

Input: results/evasion/scores_<envelope>_<model>_dump32.npz, produced by
`scripts/mimicry_evasion.py --k-cand 32 --out-suffix _dump32 --dump-scores` (the published
K=32 per-model white-box setting, re-run only to persist per-window scores; the point
estimates it reproduces are asserted against results/evasion/evasion_results.csv).

Resampling unit: the attacker's physical transmitter group, NOT the window -- windows
of one attacker are not independent (same trajectory, same channel), so a window-level
bootstrap would understate the interval. Paired by construction: before/after scores
come from the same windows, so we also report the paired recall DROP with its own CI.

Run: python scripts/studies/mimicry_ci.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
EV = REPO / "results" / "evasion"
OUT = REPO / "results" / "studies"
N_BOOT, SEED, TOL = 2000, 7, 5e-3


def _boot(flags_b, flags_a, groups, n_boot=N_BOOT, seed=SEED):
    """Grouped percentile bootstrap over attacker groups -> CIs for recall_before,
    recall_after and their paired difference."""
    uniq = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in uniq}
    rng = np.random.default_rng(seed)
    rb, ra, rd = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(len(uniq), size=len(uniq), replace=True)
        sel = np.concatenate([idx_by_g[uniq[i]] for i in pick])
        b, a = float(flags_b[sel].mean()), float(flags_a[sel].mean())
        rb.append(b)
        ra.append(a)
        rd.append(b - a)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return q(rb), q(ra), q(rd)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(EV.glob("scores_*_dump32.npz"))
    if not files:
        raise SystemExit("no scores_*_dump32.npz -- run mimicry_evasion --dump-scores first")

    pub = pd.read_csv(EV / "evasion_results.csv")
    rows, mismatches = [], []
    for f in files:
        stem = f.stem[len("scores_"):-len("_dump32")]
        env, model = stem.split("_", 1)
        z = np.load(f, allow_pickle=True)
        sb, sa = z["s_before"], z["s_after"]
        thr = float(z["thr"])
        atk = z["attack"].astype(str)
        grp = np.asarray(z["groups"])
        fb, fa = (sb > thr).astype(float), (sa > thr).astype(float)

        for a in list(sorted(set(atk))) + ["ALL"]:
            m = np.ones(len(atk), bool) if a == "ALL" else (atk == a)
            if m.sum() == 0:
                continue
            cb, ca, cd = _boot(fb[m], fa[m], grp[m])
            r = {"envelope": env, "model": model, "attack": a,
                 "n_windows": int(m.sum()),
                 "n_attacker_groups": int(len(np.unique(grp[m]))),
                 "recall_before": float(fb[m].mean()),
                 "recall_before_lo": cb[0], "recall_before_hi": cb[1],
                 "recall_after": float(fa[m].mean()),
                 "recall_after_lo": ca[0], "recall_after_hi": ca[1],
                 "recall_drop": float(fb[m].mean() - fa[m].mean()),
                 "recall_drop_lo": cd[0], "recall_drop_hi": cd[1],
                 "drop_significant": bool(cd[0] > 0)}
            # reproduction gate vs the published point estimates
            p = pub[(pub.envelope == env) & (pub.model == model) & (pub.attack == a)]
            if len(p):
                for col in ("recall_before", "recall_after"):
                    d = abs(float(p.iloc[0][col]) - r[col])
                    r[f"published_{col}"] = float(p.iloc[0][col])
                    r[f"delta_{col}"] = d
                    if d > TOL:
                        mismatches.append(
                            {"envelope": env, "model": model, "attack": a,
                             "col": col, "published": float(p.iloc[0][col]),
                             "recomputed": r[col], "delta": d})
            rows.append(r)

    df = pd.DataFrame(rows).sort_values(["envelope", "model", "attack"])
    df.to_csv(OUT / "mimicry_ci.csv", index=False)
    show = df[df.attack.isin(["ALL", "dosAttack"])][
        ["envelope", "model", "attack", "n_windows", "n_attacker_groups",
         "recall_after", "recall_after_lo", "recall_after_hi"]]
    print(show.round(3).to_string(index=False))

    # headline: does the QSVM-vs-RF DoS robustness gap survive uncertainty?
    gap = {}
    for env in df.envelope.unique():
        d = df[(df.envelope == env) & (df.attack == "dosAttack")].set_index("model")
        if {"qsvm_crosslayer", "random_forest"} <= set(d.index):
            q, r_ = d.loc["qsvm_crosslayer"], d.loc["random_forest"]
            gap[env] = {
                "qsvm_recall_after": q.recall_after,
                "qsvm_ci": [q.recall_after_lo, q.recall_after_hi],
                "rf_recall_after": r_.recall_after,
                "rf_ci": [r_.recall_after_lo, r_.recall_after_hi],
                "intervals_disjoint": bool(q.recall_after_lo > r_.recall_after_hi),
            }

    summary = {
        "protocol": {
            "source": "mimicry_evasion.py --k-cand 32 --dump-scores (published white-box "
                      "per-model best-of-K setting)",
            "resampling_unit": "attacker physical transmitter group (windows of one "
                               "attacker are not independent)",
            "n_boot": N_BOOT, "seed": SEED, "interval": "percentile 2.5/97.5",
            "paired": "before/after share the same windows, so the recall DROP is "
                      "bootstrapped as a paired quantity",
            "reproduction_tolerance": TOL,
        },
        "reproduction_mismatches": mismatches,
        "dos_robustness_gap": gap,
        "n_rows": len(df),
    }
    (OUT / "mimicry_ci_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nDoS gap:", json.dumps(gap, indent=1, default=float))
    if mismatches:
        print(f"\nWARNING: {len(mismatches)} point estimates differ from published "
              f"beyond {TOL}")


if __name__ == "__main__":
    main()
