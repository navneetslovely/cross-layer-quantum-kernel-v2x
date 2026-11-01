"""ROAD CAN-bus loader, window features and family-aware split for the CAN-IDS study.

Shared by scripts/road_ids.py (ROAD) and scripts/can_ids.py (HCRL Car-Hacking, CIC-IoV-2024).

CAN two-view feature split (the in-vehicle analogue of mobility x network):
    register A -- "WHEN" (timing / arrival / rate): message rate, inter-arrival-time
                 statistics, burstiness, window message count.
    register B -- "WHAT" (payload / signal content / ID diversity): number of distinct
                 arbitration IDs, top-ID dominance, ID entropy, new-ID count, decoded
                 signal value mean/std/range/jump.
Masquerade and fabrication attacks perturb the two views differently (a timing-only
flood vs. a content-only signal spoof vs. a coupled masquerade), so the cross-layer map
couples "when" x "what". CAN has no PHY/RSSI channel, so this is a two-view analogue,
not a literal V2X cross-layer deployment.

Pipeline:
  1. Load the signal-translated, per-message-labelled ROAD CSVs
     (data/road/road/signal_extractions/{attacks,ambient}).
  2. Build per-(capture, time-window) numeric features. Window label = 1 iff the window
     contains any attack-labelled message.
  3. Assemble an ExperimentData (groups = capture id for leakage-safe group splits;
     attack_name = ROAD attack family) and split with family_aware_split.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from _study_lib import REPO  # noqa: E402

# --------------------------------------------------------------------------- #
# ROAD layout + the labelled attack families
# --------------------------------------------------------------------------- #
ROAD_ROOT = REPO / "data" / "road" / "road" / "signal_extractions"
ATTACK_DIR = ROAD_ROOT / "attacks"
AMBIENT_DIR = ROAD_ROOT / "ambient"

# Attack signal-extraction captures whose Label column actually marks injected messages.
# (The accelerator_attack_* captures have NO labeled injections — physical attack, all
#  Label==0 in the signal extraction — so they cannot define a benign-vs-attack ROC and
#  are excluded; verified empirically by the loader, which drops any 0-positive capture.)
ATTACK_FILES = [
    "correlated_signal_attack_1_masquerade.csv",
    "correlated_signal_attack_2_masquerade.csv",
    "correlated_signal_attack_3_masquerade.csv",
    "max_engine_coolant_temp_attack_masquerade.csv",
    "max_speedometer_attack_1_masquerade.csv",
    "max_speedometer_attack_2_masquerade.csv",
    "max_speedometer_attack_3_masquerade.csv",
    "reverse_light_off_attack_1_masquerade.csv",
    "reverse_light_off_attack_2_masquerade.csv",
    "reverse_light_off_attack_3_masquerade.csv",
    "reverse_light_on_attack_1_masquerade.csv",
    "reverse_light_on_attack_2_masquerade.csv",
    "reverse_light_on_attack_3_masquerade.csv",
]

# Pure-benign ambient drives (no attack rows) used as the benign window pool. We sample a
# handful of structurally varied drives rather than the multi-GB long captures.
AMBIENT_FILES = [
    "ambient_dyno_drive_basic_short.csv",
    "ambient_dyno_drive_extended_short.csv",
    "ambient_dyno_drive_radio_infotainment.csv",
    "ambient_dyno_reverse.csv",
    "ambient_highway_street_driving_diagnostics.csv",
    "ambient_dyno_drive_winter.csv",
]


def _attack_family(fname: str) -> str:
    """ROAD attack-family name (collapse the per-capture numeric suffix + 'masquerade')."""
    s = fname.replace(".csv", "").replace("_masquerade", "")
    for fam in ("correlated_signal_attack", "max_engine_coolant_temp_attack",
                "max_speedometer_attack", "reverse_light_off_attack",
                "reverse_light_on_attack"):
        if s.startswith(fam):
            return fam
    return s


# --------------------------------------------------------------------------- #
# Windowed CAN-bus feature builder
# --------------------------------------------------------------------------- #
# 16 numeric window features split into the PRINCIPLED CAN two-view (see module docstring)
# so the reduced vector maps onto the (n_mobility=4, n_network=4) registers the cross-layer
# map expects. The split is the defensible information decomposition of a CAN window:
#   block A = "WHEN"  (timing / arrival / rate)         -> "mobility" register (temporal view)
#   block B = "WHAT"  (payload / signal-content / IDs)  -> "network"  register (content view)
# This is the in-vehicle analog of the V2X mobility x network split — a motivated two-view,
# NOT a random partition. (CAN has no PHY/RSSI to invert, so it is not a literal cross-layer
# deployment; it is the same machine on a principled CAN information split.) 8 feats/block -> PCA->4.
FEATURE_NAMES = [
    # --- block A ("mobility"/WHEN register): timing / arrival / message-rate dynamics (8) ---
    "msg_rate", "iat_mean", "iat_std", "iat_max", "iat_min", "iat_cv",
    "burstiness", "n_messages",
    # --- block B ("network"/WHAT register): ID diversity + signal content (8) ---
    "n_distinct_ids", "top_id_frac", "id_entropy", "new_id_count",
    "sig1_mean", "sig1_std", "sig1_range", "sig1_jump_max",
]
BLOCKS = (["mobility"] * 8) + ["network"] * 8
N_FEATURES = len(FEATURE_NAMES)


def _window_features(g: pd.DataFrame, prev_ids: set) -> tuple[np.ndarray, set]:
    """Aggregate one time-window of CAN messages into the FEATURE_NAMES vector.

    `prev_ids` is the running set of IDs seen so far in this capture, so `new_id_count`
    flags IDs that appear for the first time (a masquerade/injection signature)."""
    t = g["Time"].to_numpy()
    ids = g["ID"].to_numpy()
    n = len(g)
    dt = np.diff(t) if n > 1 else np.array([0.0])
    span = max(t[-1] - t[0], 1e-6)

    msg_rate = n / span
    iat_mean = float(dt.mean())
    iat_std = float(dt.std())
    iat_max = float(dt.max())
    iat_min = float(dt.min())
    iat_cv = iat_std / (iat_mean + 1e-9)
    # burstiness coefficient (Goh & Barabasi): (std-mean)/(std+mean), in [-1, 1]
    burstiness = (iat_std - iat_mean) / (iat_std + iat_mean + 1e-9)

    uniq, counts = np.unique(ids, return_counts=True)
    n_distinct = len(uniq)
    top_id_frac = float(counts.max() / n)
    p = counts / counts.sum()
    id_entropy = float(-(p * np.log2(p + 1e-12)).sum())
    new_ids = set(uniq.tolist()) - prev_ids
    new_id_count = float(len(new_ids))

    # Signal_1_of_ID: first decoded signal of each message — a generic content channel.
    s1 = pd.to_numeric(g["Signal_1_of_ID"], errors="coerce").to_numpy()
    s1 = s1[np.isfinite(s1)]
    if s1.size:
        sig1_mean = float(s1.mean())
        sig1_std = float(s1.std())
        sig1_range = float(s1.max() - s1.min())
        sig1_jump_max = float(np.abs(np.diff(s1)).max()) if s1.size > 1 else 0.0
    else:
        sig1_mean = sig1_std = sig1_range = sig1_jump_max = 0.0

    vec = np.array([
        msg_rate, iat_mean, iat_std, iat_max, iat_min, iat_cv, burstiness, float(n),
        float(n_distinct), top_id_frac, id_entropy, new_id_count,
        sig1_mean, sig1_std, sig1_range, sig1_jump_max,
    ], dtype=float)
    return vec, prev_ids | set(uniq.tolist())


def _build_capture(path: Path, window_s: float, min_msgs: int,
                   max_windows_per_capture: int | None, rng) -> tuple[np.ndarray, np.ndarray]:
    """Window one capture into (X[n_win, N_FEATURES], y[n_win]); y=1 iff window has any
    attack-labeled message. Returns empty arrays if the capture yields no usable windows."""
    df = pd.read_csv(path, usecols=lambda c: c in ("Label", "Time", "ID", "Signal_1_of_ID"))
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    bins = np.floor(df["Time"].to_numpy() / window_s).astype(np.int64)
    df["_w"] = bins

    feats, labels = [], []
    prev_ids: set = set()
    for _, g in df.groupby("_w", sort=True):
        if len(g) < min_msgs:
            # still advance the seen-ID memory so new_id_count stays meaningful
            prev_ids |= set(g["ID"].to_numpy().tolist())
            continue
        vec, prev_ids = _window_features(g, prev_ids)
        feats.append(vec)
        labels.append(int((g["Label"].to_numpy() == 1).any()))
    if not feats:
        return np.empty((0, N_FEATURES)), np.empty((0,), dtype=int)
    X = np.vstack(feats)
    y = np.asarray(labels, dtype=int)
    if max_windows_per_capture and len(y) > max_windows_per_capture:
        sel = np.sort(rng.choice(len(y), max_windows_per_capture, replace=False))
        X, y = X[sel], y[sel]
    return X, y


def build_road_dataset(window_s: float = 0.5, min_msgs: int = 20,
                       max_attack_win_per_capture: int = 400,
                       max_benign_win_per_capture: int = 350, seed: int = 7):
    """Assemble the full ROAD windowed dataset across attack + ambient captures.

    Returns an ExperimentData. groups = integer capture id (leakage-safe group split);
    attack_name = ROAD attack family for attack windows, 'none' for benign windows; aoi =
    zeros (no Age-of-Information channel exists on CAN)."""
    from qkv2x.experiments.common import ExperimentData

    rng = np.random.default_rng(seed)
    Xs, ys, atks, grps = [], [], [], []
    cap_info = []
    gid = 0

    # --- attack captures: contribute attack windows (and their own benign windows) ---
    for fname in ATTACK_FILES:
        path = ATTACK_DIR / fname
        if not path.exists():
            continue
        X, y = _build_capture(path, window_s, min_msgs, max_attack_win_per_capture, rng)
        if len(y) == 0 or int(y.sum()) == 0:
            cap_info.append({"capture": fname, "kind": "attack", "n_win": int(len(y)),
                             "n_pos": int(y.sum()), "used": False,
                             "reason": "no attack-labeled windows"})
            continue
        fam = _attack_family(fname)
        an = np.where(y == 1, fam, "none")
        Xs.append(X); ys.append(y); atks.append(an)
        grps.append(np.full(len(y), gid))
        cap_info.append({"capture": fname, "kind": "attack", "family": fam,
                         "group_id": gid, "n_win": int(len(y)), "n_pos": int(y.sum()),
                         "used": True})
        gid += 1

    # --- ambient captures: pure-benign window pool ---
    for fname in AMBIENT_FILES:
        path = AMBIENT_DIR / fname
        if not path.exists():
            continue
        X, y = _build_capture(path, window_s, min_msgs, max_benign_win_per_capture, rng)
        if len(y) == 0:
            continue
        y = np.zeros(len(y), dtype=int)  # ambient drives are benign by construction
        Xs.append(X); ys.append(y); atks.append(np.full(len(y), "none"))
        grps.append(np.full(len(y), gid))
        cap_info.append({"capture": fname, "kind": "ambient", "group_id": gid,
                         "n_win": int(len(y)), "n_pos": 0, "used": True})
        gid += 1

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    attack_name = np.concatenate(atks).astype(str)
    groups = np.concatenate(grps).astype(int)
    aoi = np.zeros(len(y), dtype=float)            # no AoI channel on CAN (zero)
    t_end = np.arange(len(y), dtype=float)         # ordinal placeholder

    data = ExperimentData(
        X=X, y=y, attack_name=attack_name, groups=groups, aoi=aoi, t_end=t_end,
        feature_names=list(FEATURE_NAMES), blocks=list(BLOCKS),
        # per-row `points` (one row per window) so ExperimentData.subset (used by the
        # detection study to cap windows for the O(N^2) Gram) can index it positionally.
        # `meta` stays empty (subset guards len==0).
        points=pd.DataFrame({"dataset": ["road"] * len(y)}), meta=pd.DataFrame(),
    )
    return data, cap_info


# --------------------------------------------------------------------------- #
# Family-aware split
# --------------------------------------------------------------------------- #
def family_aware_split(data, test_frac=0.30, val_frac=0.15, seed=7):
    """Leakage-conscious split that guarantees EVERY attack family appears in BOTH the
    train and test partitions, so every family has training windows and a held-out
    benign-vs-attack ROC.

    The repo's generic group_split only guarantees both CLASSES per side, not each
    family; with only 1-3 ROAD captures per family, whole families otherwise land on a
    single side. Strategy here:
      * Benign windows: split by CAPTURE (group) — fully leakage-safe.
      * Each attack family with >=2 captures: hold out whole captures for test (no
        capture spans train+test -> leakage-safe).
      * Each attack family with a single capture (e.g. max_engine_coolant): split that
        family's windows by TIME within the capture (train = earlier, test = later
        contiguous block) — windows are disjoint and non-overlapping, but share a
        capture; this is an explicit, documented relaxation (see metadata caveat).
    Returns (tr, va, te) index arrays. Validation is carved from train (whole captures
    where possible) so it is benign+attack and usable for SVM C-selection.
    """
    rng = np.random.default_rng(seed)
    y, an, gr = data.y, data.attack_name, data.groups
    n = len(y)
    role = np.array(["tr"] * n, dtype=object)

    # --- benign: split benign CAPTURES into test / val / train ---
    benign_caps = sorted(set(gr[y == 0].tolist()))
    rng.shuffle(benign_caps)
    n_te = max(1, int(round(len(benign_caps) * test_frac)))
    n_va = max(1, int(round(len(benign_caps) * val_frac)))
    te_caps = set(benign_caps[:n_te])
    va_caps = set(benign_caps[n_te:n_te + n_va])
    for c in te_caps:
        role[(gr == c)] = "te"
    for c in va_caps:
        role[(gr == c)] = "va"

    # --- attacks: per family, ensure presence in both train and test ---
    fams = sorted(set(an[y == 1].tolist()) - {"none"})
    for fam in fams:
        fam_mask = (an == fam) & (y == 1)
        caps = sorted(set(gr[fam_mask].tolist()))
        if len(caps) >= 2:
            # hold out ~test_frac of this family's captures (>=1) for test, the rest train
            rng.shuffle(caps)
            k = max(1, int(round(len(caps) * test_frac)))
            te_fc = set(caps[:k])
            for c in caps:
                idx_c = np.where(fam_mask & (gr == c))[0]
                role[idx_c] = "te" if c in te_fc else "tr"
        else:
            # single-capture family: contiguous temporal split of its windows
            idx_f = np.where(fam_mask)[0]
            order = idx_f[np.argsort(data.t_end[idx_f])]
            k = max(1, int(round(len(order) * test_frac)))
            role[order[-k:]] = "te"
            role[order[:-k]] = "tr"

    tr = np.where(role == "tr")[0]
    va = np.where(role == "va")[0]
    te = np.where(role == "te")[0]
    return tr, va, te


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #
# Quantum/baseline configs + the in-script features cfg (built once, reused per run).
def _build_cfgs(seed: int):
    from qkv2x.config import load_yaml
    quantum_cfg = load_yaml(REPO / "configs" / "quantum.yaml")
    baselines_cfg = load_yaml(REPO / "configs" / "baselines.yaml")
    features_cfg = {
        "reduction": {"method": "per_block_pca", "n_mobility": 4, "n_network": 4,
                      "scale_range": [0.0, float(np.pi)], "clip_quantiles": [0.005, 0.995]},
        "splits": {"test_frac": 0.30, "val_frac": 0.15, "group_by": "capture", "seed": seed},
        "aoi": {"kappa": 0.5},
    }
    return {"quantum": quantum_cfg, "baselines": baselines_cfg, "features": features_cfg}
