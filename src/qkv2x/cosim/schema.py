"""Canonical column schemas for the three raw co-simulation log streams.

These MUST match the headers written by the ns-3 CrossLayerLogger
(cosim/ns3/contrib/qkv2x/model/cross-layer-logger.cc).
"""
from __future__ import annotations

import numpy as np

TX_COLUMNS = [
    "bsm_id", "t_gen", "tx_node", "sumo_id", "claimed_id",
    "claimed_x", "claimed_y", "claimed_speed", "claimed_heading", "claimed_accel",
    "attack_type", "is_attack",
]

RX_COLUMNS = [
    "bsm_id", "t_rx", "rx_node", "rx_is_rsu", "rx_x", "rx_y",
    "rssi_dbm", "noise_dbm", "claimed_id",
    "claimed_x", "claimed_y", "claimed_speed", "claimed_heading", "claimed_accel",
    "t_gen", "tx_node", "attack_type", "is_attack",
    "delay_s",   # one-way latency t_rx - t_gen; appended, backfilled by the loader
]

TRUTH_COLUMNS = [
    "t", "sumo_id", "node", "true_x", "true_y", "true_speed", "true_heading", "true_accel",
]

RX_DTYPES = {
    "bsm_id": np.uint64,
    "t_rx": np.float64,
    "rx_node": np.uint32,
    "rx_is_rsu": np.int8,
    "rx_x": np.float64,
    "rx_y": np.float64,
    "rssi_dbm": np.float64,
    "noise_dbm": np.float64,
    "claimed_id": np.uint32,
    "claimed_x": np.float64,
    "claimed_y": np.float64,
    "claimed_speed": np.float64,
    "claimed_heading": np.float64,
    "claimed_accel": np.float64,
    "t_gen": np.float64,
    "tx_node": np.uint32,
    "attack_type": np.int16,
    "is_attack": np.int8,
    "delay_s": np.float64,
}

# integer code -> attack name (mirrors QkAttackType in attack-engine.h)
ATTACK_CODE_TO_NAME = {
    0: "none",
    1: "const_pos",
    2: "const_offset",
    3: "random_pos",
    4: "random_offset",
    5: "eventual_stop",
    6: "drift",
    7: "replay",
    8: "data_falsification",
    9: "sybil",
    10: "flooding",
}
ATTACK_NAME_TO_CODE = {v: k for k, v in ATTACK_CODE_TO_NAME.items()}
