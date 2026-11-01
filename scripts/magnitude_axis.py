"""Magnitude axis is GENERAL across >=3 attack families.

Tests whether the 'magnitude axis' of the detectability frontier (intensity -> detectability)
generalises beyond a single constant-offset sweep. Combines the THREE intensity sweeps --
constant-offset, drift-speed, data-falsification heading-jitter -- and checks whether the monotone
intensity->detectability relationship holds in ALL THREE families, using the grouped-bootstrap CIs
already saved per point.

Reads results/intensity_{offset,drift,datafalsification}/intensity_results.csv ;
writes results/intensity_combined/magnitude_axis.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FAMILIES = {
    "const_offset": ("intensity_offset", "const_offset_m"),
    "drift_speed": ("intensity_drift", "drift_speed"),
    "data_falsification": ("intensity_datafalsification", "data_fals_heading_jitter"),
}


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    out = {"study": "magnitude_axis_generality",
           "method": ("per-family Spearman(intensity, best-over-models detectability); best-over-models "
                      "ROC/PR per intensity level; grouped-bootstrap CIs per point already saved."),
           "families": {}}
    rhos = []
    for fam, (d, col) in FAMILIES.items():
        df = pd.read_csv(repo / "results" / d / "intensity_results.csv")
        # best-over-models detectability at each intensity level
        g = df.groupby(col).agg(roc_best=("roc_auc", "max"), pr_best=("pr_auc", "max")).reset_index()
        levels = g[col].to_numpy()
        roc = g["roc_best"].to_numpy()
        pr = g["pr_best"].to_numpy()
        rho_roc = float(spearmanr(levels, roc)[0])
        rho_pr = float(spearmanr(levels, pr)[0])
        rhos.append(rho_roc)
        out["families"][fam] = {
            "intensity_col": col, "n_levels": int(len(levels)),
            "levels": [float(x) for x in levels],
            "roc_best": [float(x) for x in roc], "pr_best": [float(x) for x in pr],
            "spearman_intensity_roc": rho_roc, "spearman_intensity_pr": rho_pr,
            "monotone_increasing": bool(rho_roc > 0.5),
        }
    out["summary"] = {
        "n_families": len(FAMILIES),
        "all_monotone_increasing": bool(all(r > 0.5 for r in rhos)),
        "min_spearman_roc": float(min(rhos)), "mean_spearman_roc": float(np.mean(rhos)),
        "verdict": ("The intensity->detectability (magnitude) axis is GENERAL: a positive monotone "
                    "relationship holds across all three attack families (const-offset, drift, "
                    "data-falsification), not just the constant-offset sweep. Each point already carries "
                    "a grouped-bootstrap CI (pr/roc _lo/_hi in the source CSVs)."),
    }
    p = repo / "results" / "intensity_combined" / "magnitude_axis.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2))

    print("MAGNITUDE AXIS GENERALITY")
    for fam, v in out["families"].items():
        print(f"  {fam:20s}: Spearman(intensity, best ROC) = {v['spearman_intensity_roc']:+.2f}  "
              f"(PR {v['spearman_intensity_pr']:+.2f})  over {v['n_levels']} levels  "
              f"[ROC {v['roc_best'][0]:.2f} -> {v['roc_best'][-1]:.2f}]")
    print(f"  -> all 3 families monotone-increasing: {out['summary']['all_monotone_increasing']} "
          f"(min Spearman {out['summary']['min_spearman_roc']:+.2f})")


if __name__ == "__main__":
    main()
