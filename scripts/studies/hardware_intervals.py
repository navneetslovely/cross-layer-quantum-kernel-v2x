"""hardware_intervals.py -- per-condition intervals for the paper hardware claims.

Reports, per hardware condition, the test-window count, the number of repeat runs and
95% confidence intervals, and tests whether the hardware evidence is layout sensitive.

Data: NINE repeat runs on ibm_marrakesh (342 s QPU in total, 2026-08-07), submitted as
separate jobs (IBM Quantum open plan; QPU time is metered over a 28-day rolling window).
Final sample sizes: NextGen pinned n=5, NextGen default n=4 (a fifth default run predates
the detection block and contributes only its estimability r), dev_grid pinned n=3.

What the repeats show:
 1. The default-layout estimability COLLAPSE DOES NOT REPRODUCE. The published r=0.481
    is a single run; four fresh default-layout runs give r = 0.980, 0.981, 0.982, 0.988,
    inside the pinned range (0.958-0.991). It was a device/calibration state, not a
    property of the layout.
 2. The published layout comparison (default ROC 0.48 vs pinned 0.92) compared the
    EXTREMES of two noisy samples. At n=5/n=4 the samples overlap heavily and the mean
    gap (+0.17) is NOT significant (Welch p=0.15, Mann-Whitney p=0.32).
 3. Hardware ROC-AUC here is computed on only 10 test windows, so it is quantised in
    steps of 0.04 and high-variance by construction. The stable quantity is r, the
    correlation between hardware and statevector kernel entries (120 entries per run),
    tight at 0.98+/-0.01 across every run and BOTH layouts. r is therefore the primary
    hardware quantity; hardware ROC is illustrative.

Run: python scripts/studies/hardware_intervals.py
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "studies"
NG = REPO / "results" / "veremi_nextgen" / "ibm_hardware"
DG = REPO / "results" / "dev_grid_multiseed" / "ibm_hardware"
PINNED = [33, 34, 35, 39, 51, 52, 53, 58]


def _t_ci(v):
    """Two-sided 95% t interval for a small sample; None when n < 2."""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    n = len(v)
    if n < 2:
        return (float(v[0]) if n else np.nan, np.nan, np.nan, n)
    from scipy import stats
    m, sd = float(v.mean()), float(v.std(ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return m, sd, h, n


def _load(pat, dataset):
    rows = []
    for f in sorted(glob.glob(pat)):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        bk = str(d.get("backend", ""))
        if "marrakesh" not in bk:
            continue
        il = d.get("initial_layout")
        det = d.get("detection") or {}
        est = d.get("estimability") or {}
        # ROC is taken ONLY from the runner's own detection block. The earliest
        # default run predates that block; we deliberately do NOT substitute an
        # offline reconstruction, because any other scorer produces a number that
        # is not comparable with the rest of the sample. That run still
        # contributes its estimability r, which IS directly comparable.
        hw_roc, recon = d.get("roc_auc_hw", det.get("roc_auc_hw")), False
        rows.append({
            "dataset": dataset, "file": Path(f).name, "backend": bk,
            "layout": "pinned" if il == PINNED else ("default" if il in (None, [])
                                                     else "other"),
            "shots": d.get("shots"),
            "n_test_windows": d.get("n_test") or det.get("n_test"),
            "hw_roc": hw_roc, "hw_roc_reconstructed": recon,
            "sim_roc": d.get("roc_auc_sim", det.get("roc_auc_sim")),
            "r_hw_vs_sim": est.get("r") or est.get("pearson_r") or d.get("r"),
            "contraction_a": est.get("a") or est.get("contraction_slope_a"),
            "qpu_s": d.get("qpu_usage_s"),
        })
    return rows


def _roc_from_kernels(d):
    """ROC of the hardware kernel for runs saved before the detection block existed.
    Mirrors the runner: SVM-free rank score = mean hw kernel row against train."""
    try:
        K = np.asarray(d["K_hw"], float)
        z = np.load(REPO / "results" / "veremi_nextgen" / "checkpoints" /
                    "assembled.npz", allow_pickle=True)
        y = np.asarray(z["y"], int)[np.asarray(d["test_idx"], int)]
        ytr = np.asarray(z["y"], int)[np.asarray(d["train_idx"], int)]
        from sklearn.metrics import roc_auc_score
        s = K[:, ytr == 1].mean(1) - K[:, ytr == 0].mean(1)
        return float(roc_auc_score(y, s))
    except Exception:
        return None



def _welch(a, b):
    """Welch t-test on the two layout samples (unequal variance, small n)."""
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return {"available": False}
    r = stats.ttest_ind(a, b, equal_var=False)
    return {"available": True, "t": float(r.statistic), "p_two_sided": float(r.pvalue),
            "significant_at_05": bool(r.pvalue < 0.05)}


def _mwu(a, b):
    """Distribution-free fallback -- ROC on 10 windows is quantised, not normal."""
    from scipy import stats
    if len(a) < 2 or len(b) < 2:
        return {"available": False}
    r = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"available": True, "U": float(r.statistic),
            "p_two_sided": float(r.pvalue),
            "significant_at_05": bool(r.pvalue < 0.05)}


def main() -> None:
    rows = _load(str(NG / "*.json"), "veremi_nextgen") + \
        _load(str(DG / "*.json"), "dev_grid")
    df = pd.DataFrame(rows)
    # crosslayer conditions only (zz rows are a comparison feature map and are excluded)
    df = df[~df.file.str.contains("_zz_")]
    df.to_csv(OUT / "hardware_runs.csv", index=False)
    print(df[["dataset", "layout", "hw_roc", "sim_roc", "r_hw_vs_sim",
              "n_test_windows", "file"]].to_string(index=False))

    conds, summary = [], {}
    for (ds, lay), g in df.groupby(["dataset", "layout"]):
        m, sd, h, n = _t_ci(g.hw_roc.tolist())
        rm, rsd, rh, rn = _t_ci(g.r_hw_vs_sim.tolist())
        conds.append({
            "dataset": ds, "layout": lay, "n_runs": n,
            "hw_roc_values": sorted(round(float(x), 3)
                                    for x in g.hw_roc.dropna()),
            "hw_roc_mean": m, "hw_roc_sd": sd, "hw_roc_ci95_halfwidth": h,
            "sim_roc": float(g.sim_roc.iloc[0]),
            "r_mean": rm, "r_sd": rsd, "r_ci95_halfwidth": rh, "r_n": rn,
            "n_test_windows": int(g.n_test_windows.dropna().iloc[0])
            if g.n_test_windows.notna().any() else None,
        })
    cdf = pd.DataFrame(conds)
    cdf.to_csv(OUT / "hardware_conditions.csv", index=False)
    print("\n" + cdf.round(4).to_string(index=False))

    # the layout comparison, restated with intervals
    ng = cdf[cdf.dataset == "veremi_nextgen"].set_index("layout")
    layout = {}
    if {"pinned", "default"} <= set(ng.index):
        p, d0 = ng.loc["pinned"], ng.loc["default"]
        overlap = bool(p.hw_roc_values and d0.hw_roc_values
                       and min(p.hw_roc_values) <= max(d0.hw_roc_values))
        layout = {
            "pinned_values": p.hw_roc_values, "default_values": d0.hw_roc_values,
            "pinned_mean": p.hw_roc_mean, "default_mean": d0.hw_roc_mean,
            "mean_gap": p.hw_roc_mean - d0.hw_roc_mean,
            "samples_overlap": bool(overlap),
            "welch_t_test": _welch(p.hw_roc_values, d0.hw_roc_values),
            "mannwhitney_u": _mwu(p.hw_roc_values, d0.hw_roc_values),
            "published_comparison": {"default": 0.48, "pinned": 0.92,
                                     "note": "these are the MIN of the default sample "
                                             "and (near) the MAX of the pinned sample; "
                                             "with n=3 each the comparison is far less "
                                             "clean than a 0.48-vs-0.92 headline implies"},
        }
    # Did the published marrakesh DEFAULT-layout estimability collapse reproduce?
    ngd = df[(df.dataset == "veremi_nextgen") & (df.layout == "default")]
    rr = ngd[["file", "r_hw_vs_sim"]].dropna().sort_values("file")
    collapse = {
        "published_default_r": float(rr.r_hw_vs_sim.iloc[0]) if len(rr) else None,
        "repeat_default_r": [float(x) for x in rr.r_hw_vs_sim.iloc[1:]],
        "pinned_r": [float(x) for x in df[(df.dataset == "veremi_nextgen")
                                          & (df.layout == "pinned")]
                     .r_hw_vs_sim.dropna()],
        "reproduced": bool(len(rr) > 1 and max(rr.r_hw_vs_sim.iloc[1:]) < 0.9),
        "reading": f"the default-layout estimability collapse (r=0.48) that "
                   f"motivates the layout-sensitivity claim did NOT reproduce: "
                   f"{len(rr) - 1} fresh default-layout runs on the same device "
                   f"return r~0.98, "
                   "indistinguishable from the pinned runs. The original collapse "
                   "therefore looks like a device/calibration state at that time "
                   "rather than a property of the default layout.",
    }

    r_all = df.r_hw_vs_sim.dropna()
    stability = {
        "r_min": float(r_all.min()), "r_max": float(r_all.max()),
        "r_mean": float(r_all.mean()), "r_sd": float(r_all.std(ddof=1)),
        "n_runs": int(len(r_all)),
        "hw_roc_min": float(df.hw_roc.min()), "hw_roc_max": float(df.hw_roc.max()),
        "reading": "r (120 kernel entries per run) is stable to ~0.01 across every run "
                   "and device condition, while hardware ROC-AUC (10 test windows) "
                   "spans a wide range; the estimability claim is the robust one and "
                   "should lead, with hardware ROC reported as illustrative.",
    }
    summary = {
        "protocol": {
            "added_runs": 9, "qpu_seconds_spent": 342,
            "date_utc": "2026-08-07", "backend": "ibm_marrakesh", "shots": 1024,
            "pinned_layout": PINNED,
            "accounts": "IBM Quantum open plan; QPU time is metered over a 28-day "
                        "rolling window, so the runs were submitted as separate jobs",
            "interval": "two-sided 95% t interval; n=5 NextGen pinned, n=4 NextGen "
                        "default, n=3 dev_grid pinned",
        },
        "conditions": conds,
        "layout_claim_restated": layout,
        "stability_r_vs_roc": stability,
        "default_layout_collapse_reproduction": collapse,
    }
    (OUT / "hardware_intervals_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print("\nlayout claim:", json.dumps(layout, indent=1, default=float))
    print("\nstability:", json.dumps(stability, indent=1, default=float))
    print("\ncollapse reproduction:", json.dumps(collapse, indent=1, default=float))


if __name__ == "__main__":
    main()
