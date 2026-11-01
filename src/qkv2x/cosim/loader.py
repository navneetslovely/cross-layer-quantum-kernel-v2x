"""Load the raw CSVs produced by one co-simulation run.

A "run" directory contains tx.csv, rx.csv, truth.csv and meta.json. The feature
pipeline consumes the reception log (rx.csv): each row is a real BSM decoded at a
real receiver, carrying the claimed payload, the PHY-measured RSSI, and the
receiver's own true position. Ground-truth labels (is_attack, attack_type) are
logged by the transmitter and joined onto the same row by the simulator.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .schema import ATTACK_CODE_TO_NAME, RX_DTYPES


@dataclass
class Run:
    path: Path
    rx: pd.DataFrame
    meta: dict

    @property
    def name(self) -> str:
        return self.path.name


def load_rx(run_dir: str | Path) -> pd.DataFrame:
    run_dir = Path(run_dir)
    rx_path = run_dir / "rx.csv"
    if not rx_path.exists():
        raise FileNotFoundError(f"rx.csv not found in {run_dir}")
    df = pd.read_csv(rx_path)
    if df.empty:
        raise ValueError(
            f"{rx_path} is empty — the co-simulation produced no receptions. "
            "Check that SUMO inserted vehicles and that the channel is not fully lossy."
        )
    # Backward compatibility: rx.csv logged before delay_s was added has no
    # such column. Reconstruct it from the timestamps so old and new datasets are
    # interchangeable (delay_s == one-way latency t_rx - t_gen by construction).
    if "delay_s" not in df.columns and {"t_rx", "t_gen"}.issubset(df.columns):
        df["delay_s"] = (df["t_rx"] - df["t_gen"]).clip(lower=0.0)
    for col, dt in RX_DTYPES.items():
        if col in df.columns:
            df[col] = df[col].astype(dt)
    df["attack_name"] = df["attack_type"].map(ATTACK_CODE_TO_NAME).fillna("none")
    return df


def load_meta(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {run_dir}")
    with meta_path.open() as fh:
        return json.load(fh)


def load_run(run_dir: str | Path) -> Run:
    run_dir = Path(run_dir)
    return Run(path=run_dir, rx=load_rx(run_dir), meta=load_meta(run_dir))


def iter_runs(runs_root: str | Path):
    """Yield every leaf run directory (one containing rx.csv) under runs_root."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs root not found: {runs_root}")
    found = False
    for rx_path in sorted(runs_root.rglob("rx.csv")):
        found = True
        yield rx_path.parent
    if not found:
        raise FileNotFoundError(
            f"no rx.csv under {runs_root} — run the co-simulation first (qkv2x cosim …)."
        )
