#!/usr/bin/env python3
"""can_seeds — CAN cross-domain detection over three seeds.

Re-runs the UNCHANGED scripts/can_ids.py (HCRL, CIC-IoV-2024) and scripts/road_ids.py
(ROAD) pipelines with seeds {7, 17, 27}. Nothing in those scripts is modified; only
their output root is redirected so the published seed-7 results in results/{hcrl,cic,
road}_ids/ are left untouched. The seed drives the window subsampling, the
class-balanced cap, the family-aware split and the classical model seeds (KTA batch
seed stays at quantum.yaml::train.seed, as in every published run).

Outputs: results/studies/can_seeds/seed{N}/... (per-seed, same layout as the
originals) and results/studies/can_seeds.csv + can_seeds_summary.json (mean +/- sd).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import can_ids   # noqa: E402
import road_ids  # noqa: E402

OUT = REPO / "results" / "studies"
ROOT = OUT / "can_seeds"
SEEDS = [7, 17, 27]


def main():
    t0 = time.time()
    for seed in SEEDS:
        out = ROOT / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        can_ids.RESULTS = out
        road_ids.RESULTS = out
        if not (out / "hcrl_ids" / "hcrl_detection.csv").exists():
            can_ids.run_detection(can_ids.load_hcrl(seed=seed), "hcrl",
                                  "can_seeds seed replication; pipeline unchanged", seed=seed)
        if not (out / "cic_ids" / "cic_detection.csv").exists():
            can_ids.run_detection(can_ids.load_cic(seed=seed), "cic",
                                  "can_seeds seed replication; pipeline unchanged", seed=seed)
        if not (out / "road_ids" / "road_detection.csv").exists():
            road_ids.run(seed=seed)
    rows = []
    for seed in SEEDS:
        for tag, fn in (("hcrl", "hcrl_ids/hcrl_detection.csv"), ("cic", "cic_ids/cic_detection.csv"),
                        ("road", "road_ids/road_detection.csv")):
            df = pd.read_csv(ROOT / f"seed{seed}" / fn)
            for _, r in df.iterrows():
                rows.append({"dataset": tag, "seed": seed, "model": r["model"],
                             "roc_auc": r.get("roc_auc"), "pr_auc": r.get("pr_auc"),
                             "n_test": r.get("n_test"), "n_pos": r.get("n_pos")})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "can_seeds.csv", index=False)
    agg = (df.groupby(["dataset", "model"])
             .agg(roc_mean=("roc_auc", "mean"), roc_sd=("roc_auc", lambda s: s.std(ddof=1)),
                  pr_mean=("pr_auc", "mean"), pr_sd=("pr_auc", lambda s: s.std(ddof=1)),
                  n_seeds=("seed", "nunique")).reset_index())
    agg.to_csv(OUT / "can_seeds_per_dataset.csv", index=False)
    # published seed-7 anchors
    pub = {}
    for tag, fn in (("hcrl", "hcrl_ids/hcrl_detection.csv"), ("cic", "cic_ids/cic_detection.csv"),
                    ("road", "road_ids/road_detection.csv")):
        p = REPO / "results" / fn
        if p.exists():
            d = pd.read_csv(p)
            pub[tag] = {r["model"]: float(r["roc_auc"]) for _, r in d.iterrows() if pd.notna(r.get("roc_auc"))}
    verdict = {}
    for tag, g in agg.groupby("dataset"):
        m = {r.model: (r.roc_mean, r.roc_sd) for r in g.itertuples()}
        cl, zz = m.get("qsvm_crosslayer", (None, None)), m.get("qsvm_zz", (None, None))
        best_c = max((k for k in m if k not in ("qsvm_crosslayer", "qsvm_zz")), key=lambda k: m[k][0])
        verdict[tag] = {"crosslayer": cl, "zz": zz, "best_classical": (best_c, m[best_c]),
                        "crosslayer_beats_zz_mean": bool(cl[0] is not None and zz[0] is not None and cl[0] > zz[0]),
                        "no_advantage_mean": bool(m[best_c][0] >= cl[0])}
    summ = {"seeds": SEEDS, "published_seed7_roc": pub, "verdict_mean": verdict,
            "runtime_s": time.time() - t0, "interpreter": sys.executable,
            "note": "scripts/can_ids.py and scripts/road_ids.py unchanged; output root redirected"}
    (OUT / "can_seeds_summary.json").write_text(json.dumps(summ, indent=1, default=float))
    print(agg.round(3).to_string(index=False))
    print(json.dumps({"status": "done", "runtime_s": summ["runtime_s"]}))


if __name__ == "__main__":
    main()
