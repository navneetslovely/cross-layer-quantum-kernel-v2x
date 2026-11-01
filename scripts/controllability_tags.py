"""Per-feature controllability tagging (ctrl/phys).

Tags every cross-layer window feature as attacker-CONTROLLABLE (`ctrl`) or
NON-controllable (`phys`) with the following rules:

  * Defining-lever rule: if an attack must perturb an observable to exist (e.g. the
    DoS rate, the Sybil co-location/RSSI signature), that observable is non-controllable.
  * Residual rule: claim-vs-physics residuals (kinematic residual e_kin, cross-layer
    residual e_xl, RSSI/overlap/off-road/delay) are non-controllable.
  * Controllable: claimed kinematics alone (speed/heading/accel consistency), which an
    adaptive adversary can shape to look benign without losing... see caveat below.

Covers both network observable modes: dev_grid (RSSI-anchored) and VeReMi NextGen
(timing/map/overlap-anchored). Output: results/frontier/feature_controllability.csv.

NOTE (transparency): the spec tags the kinematic residual e_kin as non-controllable
(it is the claim-vs-motion-physics residual; making it small forces a fully consistent
fake trajectory, which costs attack utility). The 4 controllable features are the
within-claim consistency/accel terms. This matches the paper's "strong envelope"
adaptive adversary, which overwrites exactly the controllable kinematics. The CSV records
the rule per feature so the assignment is auditable/adjustable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _study_lib import RESULTS, now_iso, save_table, write_metadata  # noqa: E402

from qkv2x.features.crosslayer import feature_spec  # noqa: E402

# Non-controllable (phys): residual + defining-lever observables.
PHYS_PREFIXES = (
    "e_kin", "e_xl", "rssi", "sybil", "rate", "pdr",
    "interarrival", "aoi", "delay", "offroad", "overlap",
)
# Controllable (ctrl): claimed-kinematics-only (payload the attacker fully controls).
CTRL_PREFIXES = ("speed_consistency", "heading_consistency", "accel")


def tag(name: str) -> tuple[str, str]:
    for p in CTRL_PREFIXES:
        if name.startswith(p):
            return "ctrl", "claimed-kinematics payload (attacker-controllable)"
    for p in PHYS_PREFIXES:
        if name.startswith(p):
            if name.startswith(("rate", "pdr", "interarrival")):
                return "phys", "defining-lever: rate/PDR (DoS cannot flood without it)"
            if name.startswith("sybil"):
                return "phys", "defining-lever: Sybil co-location/RSSI signature"
            return "phys", "residual: claim-vs-physics (e_kin/e_xl/RSSI/delay/map)"
    return "ctrl", "default: payload-only, no cross-check"


def main() -> None:
    t0 = time.time()
    started = now_iso()
    out_dir = RESULTS / "frontier"
    rows = []
    for mode in ("rssi", "timing"):
        for name, _src, _agg, block in feature_spec(mode):
            ctrl, rule = tag(name)
            rows.append(
                {
                    "network_mode": mode,
                    "feature": name,
                    "block": block,
                    "controllability": ctrl,
                    "rule": rule,
                }
            )
    df = pd.DataFrame(rows).drop_duplicates(subset=["network_mode", "feature"])
    save_table(df, out_dir, "feature_controllability.csv")

    summary = {}
    for mode in ("rssi", "timing"):
        sub = df[df.network_mode == mode]
        summary[mode] = {
            "n_features": int(len(sub)),
            "n_ctrl": int((sub.controllability == "ctrl").sum()),
            "n_phys": int((sub.controllability == "phys").sum()),
            "ctrl_features": sorted(sub[sub.controllability == "ctrl"].feature.tolist()),
        }

    write_metadata(
        out_dir,
        study="feature_controllability",
        started_iso=started,
        t0=t0,
        hyperparams={
            "phys_prefixes": list(PHYS_PREFIXES),
            "ctrl_prefixes": list(CTRL_PREFIXES),
        },
        results_summary=summary,
        notes="Per-feature ctrl/phys tags; feeds the controllability index C(a).",
    )
    print("done -> results/frontier/feature_controllability.csv")
    for mode, s in summary.items():
        print(f"  {mode}: {s['n_features']} features, {s['n_ctrl']} ctrl, {s['n_phys']} phys; ctrl={s['ctrl_features']}")


if __name__ == "__main__":
    main()
