"""Sliding-window dataset assembly.

Windows are built per (receiver, claimed identity) stream; each window becomes one
labelled example x in R^D. The AoI of the window is carried separately so the
quantum encoder can down-weight stale windows.

Two windowing MODES exist (``features_cfg['window']['mode']``):

  'count'  DEFAULT and the mode every published result uses — a window is W
           consecutive received MESSAGES of a stream, advanced by `stride`
           messages, kept only if it holds >= `min_messages` messages. Because
           the window is defined by message COUNT, injecting packet loss
           re-windows the stream and destroys windows near the `min_messages`
           floor: the surviving population at high loss is a denser, easier
           subset (the survivorship caveat documented in experiments/packetloss.py).

  'time'  OPT-IN (time-based window alternative, used by the fixed_span
           packet-loss study) — a window is a fixed-DURATION span on a global
           wall-clock grid, [k*stride_s, k*stride_s + span_s) for integer k
           (half-open, so span_s == stride_s tiles the timeline exactly), holding
           however many messages happen to arrive in it. Span identity therefore
           does NOT depend on how many messages survive the channel.

`build_dataset(..., fixed_spans=...)` is the second half of the same opt-in: it
evaluates EXACTLY the spans it is handed — no derivation, no `min_messages` filter,
no onset filter — so a test population can be frozen once (e.g. at zero packet loss)
and then re-featurised under degradation with the population held literally constant.
Thin and empty spans are retained and identified by ``meta['n_msgs']``.

Both new paths are inert unless explicitly requested: with no `mode` key and no
`fixed_spans` argument the count path below runs byte-for-byte as before.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .crosslayer import compute_message_features, feature_spec

_EMPTY = np.zeros(0, dtype=np.float64)
SPAN_COLUMNS = ("rx_node", "claimed_id", "t_start", "t_end")


@dataclass
class WindowedDataset:
    X: np.ndarray                  # [n_windows, n_raw_features]
    y: np.ndarray                  # [n_windows] in {0,1}
    attack_name: np.ndarray        # [n_windows] str (multi-class label)
    groups: np.ndarray             # [n_windows] physical source id (leakage-safe split)
    aoi: np.ndarray                # [n_windows] mean Age-of-Information [s]
    feature_names: list[str]
    blocks: list[str]              # 'mobility' | 'network' per feature column
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)
    # optional per-window MESSAGE sequences (for sequence baselines like an
    # LSTM): [n_windows, window_size, n_seq_sources], zero-padded, + lengths.
    sequences: np.ndarray | None = None
    seq_lengths: np.ndarray | None = None
    seq_sources: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return self.X.shape[0]


def _agg(values: np.ndarray, how: str) -> float:
    if values.size == 0:
        return 0.0
    if how == "mean":
        return float(np.mean(values))
    if how == "max":
        return float(np.max(values))
    if how == "min":
        return float(np.min(values))
    if how == "std":
        return float(np.std(values))
    raise ValueError(f"unknown aggregation {how}")


def normalize_spans(spans, scope: str = "stream") -> pd.DataFrame:
    """Coerce a `fixed_spans` argument to a canonical span table, ORDER PRESERVED.

    Accepts a DataFrame with columns (rx_node, claimed_id, t_start, t_end), an
    iterable of 4-tuples in that order, or an iterable of mappings with those keys.
    With scope='receiver' the claimed_id column may be omitted (it is set to -1,
    the receiver-scope sentinel used by the count path).

    The returned row order IS the output row order of `build_dataset(fixed_spans=...)`,
    which is what makes a frozen population alignable across re-featurisations.
    """
    if isinstance(spans, pd.DataFrame):
        df = spans.copy()
    else:
        recs = list(spans)
        if recs and isinstance(recs[0], dict):
            df = pd.DataFrame(recs)
        else:
            cols = (["rx_node", "t_start", "t_end"] if scope == "receiver"
                    and recs and len(recs[0]) == 3 else list(SPAN_COLUMNS))
            df = pd.DataFrame(recs, columns=cols)
    if "claimed_id" not in df.columns:
        if scope != "receiver":
            raise ValueError("fixed_spans needs a claimed_id column when "
                             "window.scope='stream'")
        df["claimed_id"] = -1
    missing = [c for c in SPAN_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"fixed_spans missing column(s) {missing}; "
                         f"required: {list(SPAN_COLUMNS)}")
    out = df.loc[:, list(SPAN_COLUMNS)].reset_index(drop=True)
    out["rx_node"] = out["rx_node"].astype(np.int64)
    out["claimed_id"] = out["claimed_id"].astype(np.int64)
    out["t_start"] = out["t_start"].astype(np.float64)
    out["t_end"] = out["t_end"].astype(np.float64)
    if bool((out["t_end"] <= out["t_start"]).any()):
        raise ValueError("fixed_spans requires t_end > t_start for every span")
    return out


def _derive_time_spans(msg: pd.DataFrame, scope: str, span_s: float,
                       stride_s: float) -> pd.DataFrame:
    """Fixed-duration spans on the GLOBAL grid [k*stride_s, k*stride_s + span_s).

    For each stream only the grid cells that can contain one of its messages are
    emitted: k in [floor((t_min - span_s)/stride_s) + 1, floor(t_max/stride_s)].
    Streams are visited in `msg` order and k ascends, so the output is deterministic.
    Emptiness/thinness is NOT judged here — the caller applies `min_messages`.
    """
    rows = []
    it = (((int(k), -1), g) for k, g in msg.groupby("rx_node", sort=False)) \
        if scope == "receiver" else \
        (((int(k[0]), int(k[1])), g)
         for k, g in msg.groupby(["rx_node", "claimed_id"], sort=False))
    for (rx_node, claimed_id), grp in it:
        t = grp["t_rx"].to_numpy()
        t_min, t_max = float(t.min()), float(t.max())
        k_lo = int(np.floor((t_min - span_s) / stride_s)) + 1
        k_hi = int(np.floor(t_max / stride_s))
        for k in range(k_lo, k_hi + 1):
            t0 = k * stride_s
            rows.append((rx_node, claimed_id, t0, t0 + span_s))
    if not rows:
        raise ValueError("window.mode='time' produced no candidate spans — "
                         "check span_s/stride_s against the reception times.")
    return pd.DataFrame(rows, columns=list(SPAN_COLUMNS))


def _spans_to_dataset(msg: pd.DataFrame, spans: pd.DataFrame, spec: list,
                      feature_names: list[str], blocks: list[str], scope: str,
                      group_by: str, label_rule: str, drop_onset: bool,
                      min_required: int | None,
                      nominal_rate_hz: float) -> WindowedDataset:
    """Featurise fixed-duration spans (shared by mode='time' and fixed_spans=...).

    Rate/PDR use the span's KNOWN duration D = t_end - t_start rather than the
    observed first-to-last message gap (which is the count path's only option):
        emp_rate = n_msgs / D,  rate_ratio = emp_rate / nominal_rate_hz,
        pdr      = n_msgs / (D * nominal_rate_hz + 1).
    That is the unbiased rate estimator for a fixed observation window and is what
    makes the loss signal show up in the features instead of in the population.

    `min_required=None` retains every span, including empty ones (all aggregates
    0.0, rate_ratio = pdr = 0, label 0, attack_frac NaN, meta n_msgs 0).
    """
    src_cols = [src for _, src, _, _ in spec if not src.startswith("__")]
    streams: dict[tuple[int, int], dict] = {}
    it = (((int(k), -1), g) for k, g in msg.groupby("rx_node", sort=False)) \
        if scope == "receiver" else \
        (((int(k[0]), int(k[1])), g)
         for k, g in msg.groupby(["rx_node", "claimed_id"], sort=False))
    for key, grp in it:
        grp = grp.sort_values("t_rx")
        streams[key] = {
            "t": grp["t_rx"].to_numpy(),
            "cols": {c: grp[c].to_numpy() for c in src_cols},
            "is_atk": grp["is_attack"].to_numpy(),
            "atk_name": grp["attack_name"].to_numpy(),
            "tx_node": grp["tx_node"].to_numpy(),
        }

    aoi_idx = [i for i, (_, src, _, _) in enumerate(spec) if src == "aoi"]
    rows_X, rows_y, rows_atk, rows_grp, rows_aoi, meta_rows = [], [], [], [], [], []
    for row in spans.itertuples(index=False):
        rx_node, claimed_id = int(row.rx_node), int(row.claimed_id)
        t_start, t_end = float(row.t_start), float(row.t_end)
        st = streams.get((rx_node, claimed_id))
        if st is None:
            lo = hi = 0
        else:                                  # half-open [t_start, t_end)
            lo = int(np.searchsorted(st["t"], t_start, side="left"))
            hi = int(np.searchsorted(st["t"], t_end, side="left"))
        n_in = hi - lo
        if min_required is not None and n_in < int(min_required):
            continue
        sl = slice(lo, hi)
        win_is_atk = st["is_atk"][sl] if n_in else _EMPTY
        attack_frac = float(win_is_atk.mean()) if n_in else float("nan")
        if drop_onset and n_in and 0.0 < attack_frac < 1.0:
            continue

        span_dur = max(t_end - t_start, 1e-6)
        emp_rate = n_in / span_dur
        rate_ratio = emp_rate / nominal_rate_hz if nominal_rate_hz > 0 else 0.0
        pdr = n_in / (span_dur * nominal_rate_hz + 1.0)

        feat = []
        for name, src, how, _ in spec:
            if src == "__rate_ratio__":
                feat.append(rate_ratio)
            elif src == "__pdr__":
                feat.append(pdr)
            else:
                feat.append(_agg(st["cols"][src][sl] if n_in else _EMPTY, how))
        rows_X.append(feat)

        if n_in == 0:
            label = 0
        elif label_rule == "majority":
            label = int(attack_frac >= 0.5)
        elif label_rule == "fully_active":
            label = int(bool(win_is_atk.all()))
        else:                                   # any (default, published)
            label = int(win_is_atk.any())
        rows_y.append(label)

        mal = st["atk_name"][sl][win_is_atk == 1] if n_in else _EMPTY
        if getattr(mal, "size", 0):
            vals, counts = np.unique(mal, return_counts=True)
            rows_atk.append(str(vals[np.argmax(counts)]))
        else:
            rows_atk.append("none")

        if scope == "receiver":
            rows_grp.append(rx_node)
        elif group_by == "sender":
            rows_grp.append(claimed_id)
        elif n_in:
            vals, counts = np.unique(st["tx_node"][sl], return_counts=True)
            rows_grp.append(int(vals[np.argmax(counts)]))
        else:
            rows_grp.append(-1)   # empty span: no transmitter observed

        rows_aoi.append(float(feat[aoi_idx[0]]) if aoi_idx else 0.0)
        meta_rows.append({
            "rx_node": rx_node, "claimed_id": claimed_id,
            "t_start": t_start, "t_end": t_end,
            "n_msgs": int(n_in), "attack_frac": attack_frac,
        })

    if not rows_X:
        raise ValueError(
            "no spans produced — check window.span_s/stride_s/min_messages "
            "(or the fixed_spans table) against the reception times.")

    return WindowedDataset(
        X=np.asarray(rows_X, dtype=np.float64),
        y=np.asarray(rows_y, dtype=np.int64),
        attack_name=np.asarray(rows_atk, dtype=object),
        groups=np.asarray(rows_grp, dtype=np.int64),
        aoi=np.asarray(rows_aoi, dtype=np.float64),
        feature_names=feature_names,
        blocks=blocks,
        meta=pd.DataFrame(meta_rows),
        sequences=None, seq_lengths=None, seq_sources=[],
    )


def build_dataset(
    rx: pd.DataFrame,
    features_cfg: dict,
    propagation: dict,
    nominal_rate_hz: float,
    drop_frac: float = 0.0,
    drop_seed: int = 0,
    store_sequences: bool = False,
    fixed_spans=None,
) -> WindowedDataset:
    win = features_cfg["window"]
    W = int(win["size"])
    stride = int(win["stride"])
    min_msgs = int(win["min_messages"])
    # Windowing mode. 'count' (DEFAULT)
    # = the published message-count windows below; 'time' = fixed-duration spans on
    # the global grid [k*stride_s, k*stride_s + span_s). Absent key -> 'count', so
    # every existing config/caller keeps the byte-identical count path.
    mode = win.get("mode", "count")
    if mode not in ("count", "time"):
        raise ValueError(f"window.mode must be count|time, got {mode!r}")
    span_s = float(win.get("span_s", 0.0) or 0.0)
    stride_s = float(win.get("stride_s", 0.0) or 0.0)
    # window scope: 'stream' (default — per (receiver, claimed
    # identity), the per-sender detection setting) or 'receiver' (all identities
    # heard by one receiver pooled into one time-ordered stream — the ONLY view
    # that can see one-shot Sybil ghosts, which never reach min_messages as
    # individual identities; window features like sybil_neighbors/claim_overlap/
    # rate_ratio then aggregate the receiver's whole neighbourhood).
    scope = win.get("scope", "stream")
    if scope not in ("stream", "receiver"):
        raise ValueError(f"window.scope must be stream|receiver, got {scope!r}")
    # Window label rule (label-rule sensitivity study): 'any' (default, published) marks a
    # window malicious if ANY message was sent with attack active; 'majority' requires
    # >=50% attack-active messages; 'fully_active' requires ALL messages attack-active.
    # The per-window attack fraction is always recorded in meta['attack_frac'] so
    # arbitrary-threshold reanalyses need no rebuild. `drop_onset_windows` (default off)
    # removes windows straddling an attack onset/offset (0 < fraction < 1).
    label_rule = features_cfg.get("label_rule", "any")
    if label_rule not in ("any", "majority", "fully_active"):
        raise ValueError(
            f"label_rule must be any|majority|fully_active, got {label_rule!r}")
    drop_onset = bool(features_cfg.get("drop_onset_windows", False))
    group_by = features_cfg["splits"].get("group_by", "txnode")
    sybil_cfg = features_cfg.get("sybil", {})

    # Restrict to the receiver set BEFORE windowing. Detection is per (sender,
    # receiver) stream; with receiver_set='all', every vehicle is a receiver and each
    # sender is shattered across thousands of ephemeral V2V contacts (~1 msg each)
    # that never reach min_messages — leaving only high-rate/persistent (flooding,
    # Sybil) streams and collapsing the benign class. 'rsu' uses the fixed infrastructure
    # receivers (stable streams, the intended (sender, RSU) detection setting).
    receiver_set = features_cfg.get("receiver_set", "all")
    if receiver_set == "rsu":
        rx = rx[rx["rx_is_rsu"] == 1].reset_index(drop=True)
    elif receiver_set == "vehicles":
        rx = rx[rx["rx_is_rsu"] == 0].reset_index(drop=True)
    elif receiver_set != "all":
        raise ValueError(f"receiver_set must be all|rsu|vehicles, got {receiver_set!r}")

    # H2 packet-loss injection: randomly drop a fraction of RECEPTIONS before
    # windowing, simulating a lossy channel. Loss is modelled at the message level (the
    # faithful model), so windows thin and some fall below min_messages, and the rate/pdr
    # features pick up the loss automatically. Seeded for reproducibility; drop_frac=0 (the
    # default) is a no-op, so every existing caller is unaffected.
    if drop_frac and float(drop_frac) > 0.0:
        rng = np.random.default_rng(int(drop_seed))
        rx = rx[rng.random(len(rx)) >= float(drop_frac)].reset_index(drop=True)

    # 'rssi' (default — bit-identical to the original pipeline) or 'timing'
    # (delay-anchored network features, for RSSI-less data like VeReMi NextGen).
    network_mode = features_cfg.get("network_mode", "rssi")
    spec = feature_spec(network_mode)

    msg = compute_message_features(
        rx,
        propagation=propagation,
        nominal_rate_hz=nominal_rate_hz,
        sybil_radius=float(sybil_cfg.get("radius", 50.0)),
        sybil_min_overlap=int(sybil_cfg.get("min_overlap", 6)),
        network_mode=network_mode,
        overlap_radius=float(features_cfg.get("overlap", {}).get("radius", 3.0)),
    )

    feature_names = [s[0] for s in spec]
    blocks = [s[3] for s in spec]

    # ---- OPT-IN fixed-duration span paths (inert unless requested) -------------
    # fixed_spans wins over `mode`: the caller has handed us the exact population.
    if fixed_spans is not None or mode == "time":
        if store_sequences:
            raise NotImplementedError(
                "store_sequences is only implemented for window.mode='count' "
                "(a time span holds a variable number of messages, so zero-padding "
                "to window.size would silently truncate).")
        if fixed_spans is not None:
            if drop_onset:
                raise ValueError(
                    "drop_onset_windows cannot be combined with fixed_spans: the "
                    "point of fixed_spans is that the population is exactly the "
                    "spans supplied. Filter the span table before passing it.")
            spans = normalize_spans(fixed_spans, scope)
            min_required = None            # retain thin AND empty spans (n_msgs=0)
        else:
            if span_s <= 0.0 or stride_s <= 0.0:
                raise ValueError("window.mode='time' needs positive window.span_s "
                                 "and window.stride_s (seconds)")
            spans = _derive_time_spans(msg, scope, span_s, stride_s)
            min_required = min_msgs        # ordinary windowing rule for this mode
        return _spans_to_dataset(msg, spans, spec, feature_names, blocks, scope,
                                 group_by, label_rule, drop_onset, min_required,
                                 float(nominal_rate_hz))

    rows_X, rows_y, rows_atk, rows_grp, rows_aoi = [], [], [], [], []
    meta_rows = []
    # per-window message sequences: the per-message SOURCE columns of the
    # active spec, in spec order (deduped), zero-padded to the window size.
    seq_sources = list(dict.fromkeys(
        src for _, src, _, _ in spec if not src.startswith("__")))
    rows_seq, rows_seqlen = [], []

    if scope == "receiver":
        stream_iter = (((rx_node, -1), grp)
                       for rx_node, grp in msg.groupby("rx_node", sort=False))
    else:
        stream_iter = msg.groupby(["rx_node", "claimed_id"], sort=False)

    for (rx_node, claimed_id), grp in stream_iter:
        grp = grp.sort_values("t_rx")
        nmsg = len(grp)
        if nmsg < min_msgs:
            continue
        cols = {src: grp[src].to_numpy()
                for _, src, _, _ in spec if not src.startswith("__")}
        t_rx = grp["t_rx"].to_numpy()
        is_atk = grp["is_attack"].to_numpy()
        atk_name = grp["attack_name"].to_numpy()
        tx_node = grp["tx_node"].to_numpy()

        start = 0
        while start + W <= nmsg or (start == 0 and nmsg >= min_msgs):
            end = min(start + W, nmsg)
            if end - start < min_msgs:
                break
            sl = slice(start, end)
            win_is_atk = is_atk[sl]
            attack_frac = float(win_is_atk.mean())
            if drop_onset and 0.0 < attack_frac < 1.0:
                # skip onset/offset-straddling windows (label-rule sensitivity variant)
                if start + W > nmsg:
                    break
                start += stride
                continue
            span = max(t_rx[end - 1] - t_rx[start], 1e-6)
            n_in = end - start
            emp_rate = (n_in - 1) / span if n_in > 1 else 0.0
            rate_ratio = emp_rate / nominal_rate_hz if nominal_rate_hz > 0 else 0.0
            expected = span * nominal_rate_hz + 1.0
            pdr = n_in / expected

            feat = []
            for name, src, how, _ in spec:
                if src == "__rate_ratio__":
                    feat.append(rate_ratio)
                elif src == "__pdr__":
                    feat.append(pdr)
                else:
                    feat.append(_agg(cols[src][sl], how))
            rows_X.append(feat)

            if store_sequences:
                seq = np.zeros((W, len(seq_sources)), dtype=np.float32)
                for j, src in enumerate(seq_sources):
                    seq[: n_in, j] = cols[src][sl]
                rows_seq.append(seq)
                rows_seqlen.append(n_in)

            if label_rule == "majority":
                label = int(attack_frac >= 0.5)
            elif label_rule == "fully_active":
                label = int(bool(win_is_atk.all()))
            else:  # any (default, published)
                label = int(win_is_atk.any())
            rows_y.append(label)

            mal = atk_name[sl][win_is_atk == 1]
            if mal.size:
                vals, counts = np.unique(mal, return_counts=True)
                rows_atk.append(str(vals[np.argmax(counts)]))
            else:
                rows_atk.append("none")

            if scope == "receiver":
                # receiver-scope windows mix many senders; the physical unit
                # (and leakage-safe split key) is the RECEIVER vehicle
                rows_grp.append(int(rx_node))
            elif group_by == "sender":
                rows_grp.append(int(claimed_id))
            else:  # leakage-safe: group by physical transmitter
                vals, counts = np.unique(tx_node[sl], return_counts=True)
                rows_grp.append(int(vals[np.argmax(counts)]))

            aoi_idx = [i for i, (_, src, _, _) in enumerate(spec) if src == "aoi"]
            rows_aoi.append(float(feat[aoi_idx[0]]) if aoi_idx else 0.0)

            meta_rows.append({
                "rx_node": int(rx_node), "claimed_id": int(claimed_id),
                "t_start": float(t_rx[start]), "t_end": float(t_rx[end - 1]),
                "n_msgs": int(n_in), "attack_frac": attack_frac,
            })

            if start + W > nmsg:
                break
            start += stride

    if not rows_X:
        raise ValueError(
            "no windows produced — check window.size/min_messages vs received density.")

    return WindowedDataset(
        X=np.asarray(rows_X, dtype=np.float64),
        y=np.asarray(rows_y, dtype=np.int64),
        attack_name=np.asarray(rows_atk, dtype=object),
        groups=np.asarray(rows_grp, dtype=np.int64),
        aoi=np.asarray(rows_aoi, dtype=np.float64),
        feature_names=feature_names,
        blocks=blocks,
        meta=pd.DataFrame(meta_rows),
        sequences=(np.stack(rows_seq) if store_sequences else None),
        seq_lengths=(np.asarray(rows_seqlen, dtype=np.int64)
                     if store_sequences else None),
        seq_sources=(seq_sources if store_sequences else []),
    )
