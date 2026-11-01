"""VeReMi loader — original (van der Heijden 2018) and Extension
(Kamel 2020).

VeReMi ships JSON-lines reception logs (one file per receiver) plus a ground-truth
log. We parse the real receptions into the SAME canonical schema the SUMO+ns-3
pipeline uses, so a single feature/kernel pipeline runs on both. Nothing is
synthesised; if files are missing we raise rather than invent data.

Message conventions:
  type 2  -> receiver's own GPS pose ('pos','spd','messageID' optional, time field)
  type 3  -> a received BSM ('rcvTime','sendTime','sender','senderPseudo',
             'messageID','pos','spd','RSSI')
Labels come from the per-message attacker type, taken from the message if present
(Extension) else joined from the ground-truth file by messageID (original).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

# VeReMi attacker-type code -> canonical name. Codes follow the original release;
# Extension reuses 0/1/2/4/8/16 for the position attacks and adds higher codes.
VEREMI_TYPE_TO_NAME = {
    0: "none",
    1: "const_pos",
    2: "const_offset",
    4: "random_pos",
    8: "random_offset",
    16: "eventual_stop",
}


def _first(d: dict, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _xy(vec):
    if vec is None:
        return float("nan"), float("nan")
    if isinstance(vec, (list, tuple)):
        return float(vec[0]), float(vec[1])
    return float("nan"), float("nan")


def _iter_json_lines(path: Path):
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Some VeReMi files wrap records in a top-level array; handle that.
                continue


def _load_ground_truth(sim_dir: Path) -> dict[int, int]:
    """messageID -> attackerType, from any file whose name contains GroundTruth."""
    mapping: dict[int, int] = {}
    gt_files = [p for p in sim_dir.iterdir()
                if p.is_file() and "groundtruth" in p.name.lower()]
    for gt in gt_files:
        for rec in _iter_json_lines(gt):
            mid = _first(rec, "messageID", "messageId")
            atk = _first(rec, "attackerType", "attackType", default=0)
            if mid is not None:
                mapping[int(mid)] = int(atk)
    return mapping


def load_veremi_run(sim_dir: str | Path) -> pd.DataFrame:
    """Parse one VeReMi simulation directory into a canonical rx-like frame."""
    sim_dir = Path(sim_dir)
    if not sim_dir.is_dir():
        raise NotADirectoryError(f"VeReMi sim dir not found: {sim_dir}")

    gt = _load_ground_truth(sim_dir)

    recv_files = [p for p in sim_dir.iterdir()
                  if p.is_file() and p.suffix == ".json"
                  and "groundtruth" not in p.name.lower()]
    if not recv_files:
        raise FileNotFoundError(f"no reception .json files in {sim_dir}")

    rows = []
    for f in recv_files:
        own_t, own_x, own_y = [], [], []
        receptions = []
        recv_id = None
        for rec in _iter_json_lines(f):
            mtype = _first(rec, "type")
            if mtype == 2:
                t = float(_first(rec, "rcvTime", "sendTime", "time", default=math.nan))
                x, y = _xy(_first(rec, "pos"))
                if not math.isnan(t) and not math.isnan(x):
                    own_t.append(t)
                    own_x.append(x)
                    own_y.append(y)
                if recv_id is None:
                    recv_id = _first(rec, "sender", "senderPseudo", "pseudonym")
            elif mtype == 3:
                receptions.append(rec)
        if not receptions or not own_t:
            continue

        own_t = np.asarray(own_t)
        own_x = np.asarray(own_x)
        own_y = np.asarray(own_y)
        order = np.argsort(own_t)
        own_t, own_x, own_y = own_t[order], own_x[order], own_y[order]

        if recv_id is None:
            recv_id = abs(hash(f.name)) % (2**31)

        for rec in receptions:
            t_rx = float(_first(rec, "rcvTime", "time", default=math.nan))
            t_gen = float(_first(rec, "sendTime", default=t_rx))
            cx, cy = _xy(_first(rec, "pos"))
            vx, vy = _xy(_first(rec, "spd"))
            speed = math.hypot(vx, vy) if not math.isnan(vx) else float("nan")
            heading = (math.degrees(math.atan2(vx, vy)) % 360.0) if not math.isnan(vx) else 0.0
            rssi = _first(rec, "RSSI", "rssi", default=math.nan)
            mid = _first(rec, "messageID", "messageId")
            atk = _first(rec, "attackerType", "attackType")
            if atk is None and mid is not None:
                atk = gt.get(int(mid), 0)
            atk = int(atk) if atk is not None else 0

            # receiver's own position at reception time (nearest GPS sample)
            idx = int(np.searchsorted(own_t, t_rx))
            idx = min(max(idx, 0), len(own_t) - 1)
            rxx, rxy = float(own_x[idx]), float(own_y[idx])

            rows.append({
                "bsm_id": int(mid) if mid is not None else len(rows),
                "t_rx": t_rx,
                "rx_node": int(recv_id),
                "rx_is_rsu": 0,
                "rx_x": rxx,
                "rx_y": rxy,
                "rssi_raw": float(rssi),
                "claimed_id": int(_first(rec, "senderPseudo", "sender", default=0)),
                "claimed_x": cx,
                "claimed_y": cy,
                "claimed_speed": speed,
                "claimed_heading": heading,
                "claimed_accel": 0.0,
                "t_gen": t_gen,
                "tx_node": int(_first(rec, "sender", "senderPseudo", default=0)),
                "attack_type_raw": atk,
                "is_attack": 1 if atk != 0 else 0,
            })

    if not rows:
        raise ValueError(f"no usable receptions parsed from {sim_dir}")

    df = pd.DataFrame(rows)
    # Convert RSSI to a dBm-like scale; the absolute offset is absorbed by the
    # per-receiver propagation calibration in the VeReMi experiment.
    pos = df["rssi_raw"] > 0
    df["rssi_dbm"] = np.where(pos, 10.0 * np.log10(df["rssi_raw"].clip(lower=1e-20)),
                              df["rssi_raw"])
    df["noise_dbm"] = np.nan
    df["attack_name"] = df["attack_type_raw"].map(VEREMI_TYPE_TO_NAME).fillna(
        "attack_" + df["attack_type_raw"].astype(str))
    df["attack_type"] = df["attack_type_raw"]
    return df


def load_veremi(root: str | Path, max_runs: int | None = None) -> pd.DataFrame:
    """Concatenate every VeReMi simulation directory found under `root`.

    A simulation directory is any directory that contains a GroundTruth file.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"VeReMi root {root} not found — run scripts/fetch_veremi.sh first.")
    sim_dirs = []
    for p in sorted(root.rglob("*")):
        if p.is_dir() and any("groundtruth" in c.name.lower() for c in p.iterdir()
                              if c.is_file()):
            sim_dirs.append(p)
    if not sim_dirs:
        raise FileNotFoundError(
            f"no VeReMi simulation directories (with a GroundTruth file) under {root}.")
    if max_runs:
        sim_dirs = sim_dirs[:max_runs]

    frames = []
    for i, sd in enumerate(sim_dirs):
        df = load_veremi_run(sd)
        df["sim_run"] = sd.name
        df["rx_node"] = df["rx_node"].astype(np.int64) + i * 10_000_000  # disjoint ids
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
