"""VeReMi NextGen loader — Hermann et al., IEEE VNC 2026 (MOSAIC + SUMO InTAS).

NextGen ships, per (scenario, attack-subset), an official Train/Validation/Test
tree of per-receiver JSON files (`veh_<id>.json`), each an ARRAY of received CAMs:

    {rcvTime, sendTime (ns), sender_id ("veh_X"), sender_alias (pseudonym),
     messageID, attacker (0/1; absent in the no-attack baseline subset),
     receiver:{pos,pos_noise,spd,spd_noise,acl,acl_noise,hed,hed_noise,...},
     sender:  {same fields + distance_to_road_edge}}

Values are encoded either as JSON numbers or as strings ("x,y,z" for vectors)
depending on the subset — both are handled. We map every reception onto the SAME
canonical rx schema the SUMO+ns-3 pipeline uses. Differences vs classic VeReMi:

  * NO RSSI is logged. `rssi_dbm` is set to NaN and a `delay_us` column
    (one-way delay, sendTime->rcvTime, microseconds) is provided instead; the
    feature pipeline must run with `network_mode: timing` (crosslayer.py).
  * The CAM payload a receiver actually saw is the noisy measurement, so the
    claimed kinematics are `value + value_noise` (the ground-truth file confirms
    `pos` is attack-shifted while `pos_noise` is the sensor-error sample).
  * `attacker` is BINARY; the attack TYPE is the subset the file came from.

Nothing is synthesised; missing files raise rather than invent data.
"""
from __future__ import annotations

import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

# Subset directory names as published (Zenodo record 19665762, v1.0.0).
NEXTGEN_ATTACKS = [
    "accelerationMultiplication", "constantPositionOffset", "constantSpeedOffset",
    "dataReplay", "dosAttack", "feignedBraking", "positionMirroring",
    "randomPositionOffset", "randomSpeedOffset", "reversedHeading",
    "suddenConstantSpeed", "suddenStop", "timeDelayAttack",
    "trafficCongestionSybil", "zeroSpeedReport",
]
NEXTGEN_SPLITS = ("Train", "Validation", "Test")

_NS = 1e9   # NextGen timestamps are integer nanoseconds


def _f(v, default=np.nan) -> float:
    """Float from a NextGen value that may be a number, a numeric string, or None."""
    if v is None:
        return float(default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(default)


def _vec2(v) -> tuple[float, float]:
    """(x, y) from 'x,y,z' string or [x, y, z] list."""
    if v is None:
        return np.nan, np.nan
    if isinstance(v, str):
        parts = v.split(",")
        if len(parts) < 2:
            return np.nan, np.nan
        return _f(parts[0]), _f(parts[1])
    if isinstance(v, (list, tuple)) and len(v) >= 2:
        return _f(v[0]), _f(v[1])
    return np.nan, np.nan


def _stable_id(s: str) -> int:
    """Deterministic 31-bit int id for a NextGen string id (e.g. 'veh_2489').

    Plain digits are used directly; anything else is crc32-hashed (stable across
    processes, unlike Python's salted hash())."""
    s = str(s)
    tail = s.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return zlib.crc32(s.encode()) & 0x7FFFFFFF


def subset_dir(root: str | Path, scenario: str, subset: str, split: str) -> Path:
    """data/veremi_nextgen/extracted/<scenario>/<subset>/<Split> (see fetch script)."""
    return Path(root) / "extracted" / scenario / subset / split


def load_nextgen_split(split_dir: str | Path, attack_name: str) -> pd.DataFrame:
    """Parse one extracted (scenario, subset, split) directory of veh_*.json files
    into the canonical rx frame. `attack_name` labels attacker messages ('none'
    for the baseline subset, where no `attacker` field exists)."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise NotADirectoryError(f"NextGen split dir not found: {split_dir}")
    files = sorted(split_dir.rglob("veh_*.json"))
    if not files:
        raise FileNotFoundError(f"no veh_*.json receiver files under {split_dir}")
    return load_nextgen_files(files, attack_name)


def load_nextgen_files(files: list, attack_name: str) -> pd.DataFrame:
    """Parse an explicit list of per-receiver veh_*.json files (canonical frame).

    Split out of load_nextgen_split so the experiment driver can shard a large
    split across checkpointed batches."""
    rows = []
    for fp in files:
        rx_node = _stable_id(fp.stem)
        try:
            msgs = json.loads(fp.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"unparseable NextGen file {fp}: {exc}") from exc
        if not isinstance(msgs, list):
            raise ValueError(f"{fp}: expected a JSON array of CAMs")
        for m in msgs:
            snd = m.get("sender") or {}
            rcv = m.get("receiver") or {}
            t_rx = _f(m.get("rcvTime")) / _NS
            t_gen = _f(m.get("sendTime")) / _NS
            sx, sy = _vec2(snd.get("pos"))
            snx, sny = _vec2(snd.get("pos_noise"))
            rxx, rxy = _vec2(rcv.get("pos"))
            rnx, rny = _vec2(rcv.get("pos_noise"))
            atk = int(_f(m.get("attacker"), default=0) or 0)
            rows.append({
                "bsm_id": int(_f(m.get("messageID"), default=-1)),
                "t_rx": t_rx,
                "t_gen": t_gen,
                "rx_node": rx_node,
                "rx_is_rsu": 0,
                # what the receiver believes about itself = its noisy GPS fix
                "rx_x": rxx + (rnx if np.isfinite(rnx) else 0.0),
                "rx_y": rxy + (rny if np.isfinite(rny) else 0.0),
                # the claim as transmitted = measured (noisy / attacked) state
                "claimed_id": int(_f(m.get("sender_alias"), default=0)),
                "claimed_x": sx + (snx if np.isfinite(snx) else 0.0),
                "claimed_y": sy + (sny if np.isfinite(sny) else 0.0),
                "claimed_speed": _f(snd.get("spd"), 0.0) + _f(snd.get("spd_noise"), 0.0),
                "claimed_heading": (_f(snd.get("hed"), 0.0)
                                    + _f(snd.get("hed_noise"), 0.0)) % 360.0,
                "claimed_accel": _f(snd.get("acl"), 0.0) + _f(snd.get("acl_noise"), 0.0),
                "tx_node": _stable_id(m.get("sender_id", "veh_0")),
                # signed distance of the CLAIMED position to the road edge
                # (NextGen enrichment; receiver-computable from claim + map).
                # Present on every message of the attack subsets; absent in the
                # no-attack baselines (use include_baseline: false there).
                "road_edge_dist": _f(snd.get("distance_to_road_edge")),
                "attack_type_raw": atk,
                "is_attack": 1 if atk else 0,
            })

    if not rows:
        raise ValueError(f"no receptions parsed from {len(files)} receiver files")
    df = pd.DataFrame(rows)
    df = df[np.isfinite(df["t_rx"]) & np.isfinite(df["claimed_x"])].reset_index(drop=True)

    # network-layer observables: NextGen logs ns-resolution timing, not RSSI
    df["rssi_dbm"] = np.nan
    df["noise_dbm"] = np.nan
    df["delay_us"] = (df["t_rx"] - df["t_gen"]) * 1e6
    df["attack_name"] = np.where(df["is_attack"] == 1, attack_name, "none")
    df["attack_type"] = df["attack_type_raw"]
    return df


def estimate_delay_model(rx: pd.DataFrame, min_n: int = 50) -> dict:
    """Fit delay_us = a + b * claimed_distance on BENIGN receptions (the timing
    analogue of veremi_exp.estimate_propagation): the fitted line is the expected
    MAC+propagation delay at a claimed distance, and the timing cross-layer
    residual is e_xl = |delay_us - (a + b*d_claim)| (crosslayer.py, mode=timing).
    Returns a propagation-style dict for build_dataset."""
    benign = rx[rx["is_attack"] == 0]
    d = np.hypot(benign["claimed_x"] - benign["rx_x"],
                 benign["claimed_y"] - benign["rx_y"]).to_numpy()
    delay = benign["delay_us"].to_numpy()
    good = np.isfinite(d) & np.isfinite(delay) & (delay > 0)
    if int(good.sum()) < min_n:
        raise ValueError(
            f"estimate_delay_model: only {int(good.sum())} usable benign receptions (<{min_n}).")
    b, a = np.polyfit(d[good], delay[good], 1)
    return {
        "network_mode": "timing",
        "delay_intercept_us": float(a),
        "delay_slope_us_per_m": float(b),
        "_fit": {"slope_us_per_m": float(b), "intercept_us": float(a),
                 "n": int(good.sum()),
                 "delay_med_us": float(np.median(delay[good]))},
    }
