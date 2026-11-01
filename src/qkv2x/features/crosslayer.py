"""Per-message cross-layer features.

All features are computed from what a receiver legitimately observes: the claimed
payload (position/speed/heading/accel), the PHY-measured RSSI, and the receiver's
own true position. Ground truth is used only for labels, never as a feature.

Key signals:
  e_kin  (F1) kinematic plausibility residual of the claimed motion
  speed/heading  internal consistency of claimed kinematics vs displacement
  e_xl  (F2) |claimed distance - RSSI-implied distance|  <-- the
                     cross-layer feature that needs the network simulator
  sybil_neighbors  (F3) co-located distinct identities at the same receiver
  sybil_rssi_absdiff (F3) RSSI closeness of those identities (one source -> small)
  sybil_rssi_corr  (F3) RSSI-series correlation across co-located IDs (one source -> ~1)
  rate / pdr  (F4) transmission-rate and delivery anomalies
  aoi  Age of Information = t_rx - t_gen  (drives quantum modulation)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# (window_feature_name, source_message_column, aggregation, block)
FEATURE_SPEC = [
    ("e_kin_mean", "e_kin", "mean", "mobility"),
    ("e_kin_max", "e_kin", "max", "mobility"),
    ("e_kin_std", "e_kin", "std", "mobility"),
    ("speed_consistency_mean", "speed_consistency", "mean", "mobility"),
    ("speed_consistency_max", "speed_consistency", "max", "mobility"),
    ("heading_consistency_mean", "heading_consistency", "mean", "mobility"),
    ("accel_abs_mean", "claimed_accel_abs", "mean", "mobility"),
    ("e_xl_mean", "e_xl", "mean", "network"),
    ("e_xl_max", "e_xl", "max", "network"),
    ("e_xl_std", "e_xl", "std", "network"),
    ("rate_ratio", "__rate_ratio__", "special", "network"),
    ("interarrival_std", "interarrival", "std", "network"),
    ("pdr_proxy", "__pdr__", "special", "network"),
    ("sybil_neighbors_mean", "sybil_neighbors", "mean", "network"),
    ("sybil_neighbors_max", "sybil_neighbors", "max", "network"),
    ("sybil_rssi_absdiff_min", "sybil_rssi_absdiff", "min", "network"),
    ("sybil_rssi_corr_max", "sybil_rssi_corr", "max", "network"),
    ("rssi_std", "rssi_dbm", "std", "network"),
    ("aoi_mean", "aoi", "mean", "network"),
]
FEATURE_NAMES = [s[0] for s in FEATURE_SPEC]
FEATURE_BLOCKS = {s[0]: s[3] for s in FEATURE_SPEC}


def feature_spec(network_mode: str = "rssi") -> list[tuple]:
    """FEATURE_SPEC for a given network-observable mode.

    'rssi'   — the SUMO+ns-3 / classic-VeReMi spec above (default; unchanged).
    'timing' — for datasets that log no RSSI but do log ns-resolution timing
               (VeReMi NextGen): the PHY signature column becomes the one-way
               delay (`delay_us`), so `rssi_std` -> `delay_std` and the F3 Sybil
               signature features correlate per-identity DELAY series instead of
               RSSI series (same physical rationale: one radio, one queue/channel
               -> shared timing signature). e_xl is the delay-model residual.
    """
    if network_mode == "rssi":
        return FEATURE_SPEC
    if network_mode == "timing":
        spec = [(("delay_std", "delay_us", how, blk) if name == "rssi_std"
                 else (name, src, how, blk))
                for name, src, how, blk in FEATURE_SPEC]
        # Map/context plausibility features available in timing-mode datasets
        # (VeReMi NextGen; earlier follow-up): without RSSI nothing anchors the
        # CLAIMED position, so position falsification needs map plausibility —
        # `offroad_m` = metres the claimed position lies OFF the drivable road
        # (receiver-computable from claim + map; logged as
        # sender.distance_to_road_edge) — and cross-sender physical-overlap
        # plausibility — `claim_overlap` = # of OTHER identities claiming a
        # position within a vehicle footprint (~3 m) at the same instant
        # (benign traffic is collision-free, so stale/delayed/replayed/offset
        # claims that land on live traffic are anomalous).
        spec += [
            ("offroad_max", "offroad_m", "max", "mobility"),
            ("offroad_mean", "offroad_m", "mean", "mobility"),
            ("overlap_3m_max", "claim_overlap", "max", "network"),
            ("overlap_3m_mean", "claim_overlap", "mean", "network"),
        ]
        return spec
    raise ValueError(f"network_mode must be rssi|timing, got {network_mode!r}")


def _angular_diff_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs((a - b + 180.0) % 360.0 - 180.0)
    return d


def compute_message_features(
    df: pd.DataFrame,
    propagation: dict,
    nominal_rate_hz: float,
    sybil_radius: float,
    sybil_time_win: float = 0.5,
    sybil_min_overlap: int = 6,
    network_mode: str = "rssi",
    overlap_radius: float = 3.0,
) -> pd.DataFrame:
    """Add per-message feature columns to a copy of the reception frame.

    `network_mode` selects the network-layer observable that anchors F2/F3: 'rssi' (default — received power, the ns-3/classic-VeReMi path,
    bit-identical to the original behaviour) or 'timing' (one-way delay, for
    datasets like VeReMi NextGen that log no RSSI; `propagation` then carries the
    benign-calibrated delay model from data.veremi_nextgen.estimate_delay_model).
    """
    sig_col = "rssi_dbm" if network_mode == "rssi" else "delay_us"
    required = {"rx_node", "claimed_id", "t_rx", "t_gen", "rx_x", "rx_y",
                "claimed_x", "claimed_y", "claimed_speed", "claimed_heading",
                "claimed_accel", sig_col}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"reception frame missing columns: {sorted(missing)}")

    out = df.copy()
    out = out.sort_values(["rx_node", "claimed_id", "t_gen"]).reset_index(drop=True)

    # --- F2: cross-layer residual (per message) ------------------------------
    d_claim = np.hypot(out["claimed_x"] - out["rx_x"], out["claimed_y"] - out["rx_y"])
    if network_mode == "rssi":
        # claimed distance vs RSSI-implied distance
        ptx = propagation["tx_power_dbm"]
        pl0 = propagation["ref_loss_db"]
        eta = propagation["path_loss_exponent"]
        d0 = propagation["ref_distance"]
        d_rssi = d0 * np.power(10.0, (ptx - pl0 - out["rssi_dbm"]) / (10.0 * eta))
        out["e_xl"] = np.abs(d_claim.to_numpy() - d_rssi.to_numpy())
    elif network_mode == "timing":
        # observed one-way delay vs the benign delay expected at the CLAIMED
        # distance (delay model fitted on benign receptions only)
        a = propagation["delay_intercept_us"]
        b = propagation["delay_slope_us_per_m"]
        expected = a + b * d_claim.to_numpy()
        out["e_xl"] = np.abs(out["delay_us"].to_numpy() - expected)
    else:
        raise ValueError(f"network_mode must be rssi|timing, got {network_mode!r}")
    out["aoi"] = np.maximum(out["t_rx"] - out["t_gen"], 0.0)
    out["claimed_accel_abs"] = np.abs(out["claimed_accel"])

    # --- F1 + kinematic-consistency (sequential within sender stream) --------
    g = out.groupby(["rx_node", "claimed_id"], sort=False)
    dt = g["t_gen"].diff()
    px = g["claimed_x"].shift()
    py = g["claimed_y"].shift()
    pspd = g["claimed_speed"].shift()
    phdg = g["claimed_heading"].shift()
    pacc = g["claimed_accel"].shift()

    hr = np.radians(phdg.to_numpy())
    pvx = pspd.to_numpy() * np.sin(hr)
    pvy = pspd.to_numpy() * np.cos(hr)
    pax = pacc.to_numpy() * np.sin(hr)
    pay = pacc.to_numpy() * np.cos(hr)
    dt_np = dt.to_numpy()
    pred_x = px.to_numpy() + pvx * dt_np + 0.5 * pax * dt_np**2
    pred_y = py.to_numpy() + pvy * dt_np + 0.5 * pay * dt_np**2
    e_kin = np.hypot(out["claimed_x"].to_numpy() - pred_x,
                     out["claimed_y"].to_numpy() - pred_y)
    out["e_kin"] = np.where(np.isfinite(e_kin), e_kin, 0.0)

    dx = out["claimed_x"].to_numpy() - px.to_numpy()
    dy = out["claimed_y"].to_numpy() - py.to_numpy()
    disp = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        implied_speed = np.where(dt_np > 0, disp / dt_np, np.nan)
    speed_cons = np.abs(out["claimed_speed"].to_numpy() - implied_speed)
    out["speed_consistency"] = np.where(np.isfinite(speed_cons), speed_cons, 0.0)

    disp_heading = np.degrees(np.arctan2(dx, dy)) % 360.0
    hc = _angular_diff_deg(out["claimed_heading"].to_numpy(), disp_heading)
    # only meaningful when actually moving
    hc = np.where(disp > 1.0, hc, 0.0)
    out["heading_consistency"] = np.where(np.isfinite(hc), hc, 0.0)

    out["interarrival"] = np.where(np.isfinite(dt_np), dt_np, 0.0)

    # --- F3: Sybil co-location at a receiver (cross-sender) -------------------
    # The PHY-signature column is RSSI in 'rssi' mode and one-way delay in
    # 'timing' mode; the Sybil rationale is identical (one physical radio -> one
    # channel/queue -> shared signature across its pseudonyms).
    sybil_neighbors, sybil_absdiff = _sybil_features(
        out, radius=sybil_radius, t_win=sybil_time_win, sig_col=sig_col)
    out["sybil_neighbors"] = sybil_neighbors
    out["sybil_rssi_absdiff"] = sybil_absdiff
    # F3: correlated RSSI across co-located distinct identities. Two
    # pseudonyms emitted by ONE physical radio share the same channel, so their
    # RSSI time series are highly correlated — the discriminating Sybil signature
    # that |RSSI diff| alone misses (it is also small for two genuinely-nearby cars).
    out["sybil_rssi_corr"] = _sybil_rssi_correlation(
        out, radius=sybil_radius, t_win=sybil_time_win,
        min_overlap=sybil_min_overlap, sig_col=sig_col)

    if network_mode == "timing":
        # map plausibility: metres OFF the drivable road the claim lies
        # (unknown -> 0.0 = the benign on-road assumption; never a false alarm)
        if "road_edge_dist" not in out.columns:
            raise KeyError("timing mode needs a road_edge_dist column "
                           "(claimed-position distance to road edge)")
        out["offroad_m"] = np.clip(-out["road_edge_dist"].fillna(0.0), 0.0, None)
        # cross-sender physical-overlap plausibility (vehicle footprint scale)
        overlap_cnt, _ = _sybil_features(out, radius=float(overlap_radius),
                                         t_win=sybil_time_win, sig_col=sig_col)
        out["claim_overlap"] = overlap_cnt

    out["nominal_rate_hz"] = float(nominal_rate_hz)
    return out


def _sybil_features(df: pd.DataFrame, radius: float, t_win: float,
                    sig_col: str = "rssi_dbm"):
    """For each message: # of other claimed identities co-located (<= radius) at
    the same receiver within +/- t_win, and the minimum |signature difference| to
    them (signature = RSSI or delay, per `sig_col`). Many co-located identities
    sharing one PHY signature => Sybil."""
    n = len(df)
    neighbors = np.zeros(n, dtype=np.float64)
    absdiff = np.full(n, np.nan, dtype=np.float64)

    t = df["t_rx"].to_numpy()
    cid = df["claimed_id"].to_numpy()
    x = df["claimed_x"].to_numpy()
    y = df["claimed_y"].to_numpy()
    r = df[sig_col].to_numpy()

    for _, idx in df.groupby("rx_node", sort=False).indices.items():
        idx = np.asarray(idx)
        order = np.argsort(t[idx])
        idx = idx[order]
        tt = t[idx]
        lo = 0
        hi = 0
        m = len(idx)
        for k in range(m):
            while tt[k] - tt[lo] > t_win:
                lo += 1
            if hi < k:
                hi = k
            while hi + 1 < m and tt[hi + 1] - tt[k] <= t_win:
                hi += 1
            gi = idx[k]
            cnt = 0
            best = np.inf
            for j in range(lo, hi + 1):
                gj = idx[j]
                if gj == gi or cid[gj] == cid[gi]:
                    continue
                if abs(x[gj] - x[gi]) > radius or abs(y[gj] - y[gi]) > radius:
                    continue
                if (x[gj] - x[gi]) ** 2 + (y[gj] - y[gi]) ** 2 <= radius * radius:
                    cnt += 1
                    diff = abs(r[gj] - r[gi])
                    if diff < best:
                        best = diff
            neighbors[gi] = cnt
            if cnt > 0:
                absdiff[gi] = best
    # No co-located identity => large RSSI "difference" (i.e. not Sybil-like).
    absdiff = np.where(np.isfinite(absdiff), absdiff, 100.0)
    return neighbors, absdiff


def _sybil_rssi_correlation(df: pd.DataFrame, radius: float, t_win: float,
                            min_overlap: int, sig_col: str = "rssi_dbm"):
    """Per message: the MAX Pearson correlation of this identity's RSSI time series
    with any *co-located distinct* identity's RSSI series at the same receiver.

    Two pseudonyms emitted by ONE physical radio traverse the same channel, so their
    received-power fluctuations (shadowing/fading) are near-perfectly correlated — a
    Sybil signature that the |RSSI difference| feature alone cannot separate from two
    genuinely-nearby honest vehicles. RSSI series are aligned by binning t_rx into
    `t_win` bins; a pair contributes only if it shares >= `min_overlap` bins. Returns
    0 for identities with no sufficiently-overlapping co-located partner (not Sybil)."""
    n = len(df)
    corr = np.zeros(n, dtype=np.float64)
    t = df["t_rx"].to_numpy()
    cid = df["claimed_id"].to_numpy()
    x = df["claimed_x"].to_numpy()
    y = df["claimed_y"].to_numpy()
    r = df[sig_col].to_numpy()
    bin_w = max(float(t_win), 1e-6)

    for _, idx in df.groupby("rx_node", sort=False).indices.items():
        idx = np.asarray(idx)
        ids = cid[idx]
        uniq = np.unique(ids)
        if len(uniq) < 2:
            continue
        # Per-identity profile: time-binned mean RSSI + mean claimed position + rows.
        prof = {}
        for uid in uniq:
            sel = idx[ids == uid]
            bins = np.round(t[sel] / bin_w).astype(np.int64)
            ub, inv = np.unique(bins, return_inverse=True)
            sums = np.zeros(len(ub)); cnts = np.zeros(len(ub))
            np.add.at(sums, inv, r[sel]); np.add.at(cnts, inv, 1.0)
            binmean = dict(zip(ub.tolist(), (sums / cnts).tolist()))
            prof[uid] = (binmean, float(np.mean(x[sel])), float(np.mean(y[sel])), sel)

        ulist = uniq.tolist()
        maxcorr = {uid: 0.0 for uid in ulist}
        for ia in range(len(ulist)):
            ba, xa, ya, _ = prof[ulist[ia]]
            for ib in range(ia + 1, len(ulist)):
                bb, xb, yb, _ = prof[ulist[ib]]
                if (xa - xb) ** 2 + (ya - yb) ** 2 > radius * radius:
                    continue  # not co-located -> cannot be the same physical source
                shared = sorted(ba.keys() & bb.keys())
                if len(shared) < min_overlap:
                    continue
                va = np.fromiter((ba[k] for k in shared), dtype=np.float64, count=len(shared))
                vb = np.fromiter((bb[k] for k in shared), dtype=np.float64, count=len(shared))
                if va.std() < 1e-9 or vb.std() < 1e-9:
                    continue
                c = float(np.corrcoef(va, vb)[0, 1])
                if not np.isfinite(c):
                    continue
                if c > maxcorr[ulist[ia]]:
                    maxcorr[ulist[ia]] = c
                if c > maxcorr[ulist[ib]]:
                    maxcorr[ulist[ib]] = c
        for uid in ulist:
            corr[prof[uid][3]] = maxcorr[uid]
    return corr
