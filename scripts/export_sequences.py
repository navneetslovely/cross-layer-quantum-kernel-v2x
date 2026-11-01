"""Export each dataset's windowed/sequence feature table WITH labels to CSV, so the
full pipeline (raw data -> sequences -> results) is independently auditable and plots can
be regenerated WITHOUT re-running the heavy loaders / co-simulation.

Writes to full_paper_data/sequences/ (big ones gzipped):
  dev_grid_sequences.csv.gz        90,553 windows x (19 cross-layer features + y + attack_name + group + ...)
  veremi_nextgen_sequences.csv      5,999 windows x (23 features + y + attack_name + group + aoi + split)
  veremi_original_windows.csv  per-window labels + per-model scores (input features need the loader; see manifest)

Raw datasets are not redistributed (see README.md, Datasets). Run:
  python scripts/export_sequences.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "full_paper_data" / "sequences"
manifest = {}


def record(name, df, label_col):
    OUT.mkdir(parents=True, exist_ok=True)
    gz = len(df) > 20000
    fn = f"{name}.csv.gz" if gz else f"{name}.csv"
    df.to_csv(OUT / fn, index=False, compression="gzip" if gz else None)
    dist = df[label_col].value_counts().to_dict() if label_col in df else {}
    manifest[name] = {"file": fn, "rows": int(len(df)), "cols": list(df.columns),
                      "label_col": label_col,
                      "label_dist": {str(k): int(v) for k, v in dist.items()}}
    print(f"  {fn:48s} rows={len(df):7d} cols={len(df.columns)}")


# 1. dev_grid (primary co-sim) — full windowed feature table
p = REPO / "runs" / "dev_grid" / "features.parquet"
if p.exists():
    record("dev_grid_sequences", pd.read_parquet(p), "y")

# 2. VeReMi NextGen — windowed features from the assembled checkpoint
z = REPO / "results" / "veremi_nextgen" / "checkpoints" / "assembled.npz"
if z.exists():
    d = np.load(z, allow_pickle=True)
    cols = [str(c) for c in d["feature_names"]]
    df = pd.DataFrame(d["X"], columns=cols)
    df["y"] = d["y"]; df["attack_name"] = d["attack_name"].astype(str)
    df["group"] = d["groups"]; df["aoi"] = d["aoi"]; df["split"] = d["split"].astype(str)
    record("veremi_nextgen_sequences", df, "y")

# 3. VeReMi original — per-window labels + per-model scores (input features need the loader)
pp = REPO / "results" / "veremi" / "predictions.parquet"
if pp.exists():
    pr = pd.read_parquet(pp)
    binr = pr[pr["task"] == "binary"]
    wide = binr.pivot_table(index=["i", "y_true", "attack_name", "group", "t_end"],
                            columns="model", values="score").reset_index()
    record("veremi_original_windows", wide, "y_true")

(OUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
print(f"\n[done] {len(manifest)} sequence files -> {OUT.relative_to(REPO)}/  (+ MANIFEST.json)")
for n, m in manifest.items():
    print(f"   {n}: {m['rows']} rows, label '{m['label_col']}'")
