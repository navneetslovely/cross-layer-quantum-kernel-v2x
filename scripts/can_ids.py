"""CAN-IDS detection study on HCRL Car-Hacking + CIC-IoV-2024 (extends scripts/road_ids.py to 3 datasets).

Runs the same model zoo (qsvm_crosslayer, qsvm_zz, rbf_svm, random_forest, xgboost, mlp) on two more
in-vehicle CAN datasets, with leakage-safe family-aware splits and grouped bootstrap CIs. Honest
headline checks per dataset: (a) does the trained cross-layer kernel beat the untrained ZZ ablation?
(b) is there any quantum advantage over the best classical baseline? (expected: cross-layer>>ZZ, no advantage.)

Dataset fit notes (honest):
  * HCRL has TIMESTAMPS -> the two qubit registers are timing/rate (WHEN) x payload/ID content (WHAT),
    the same cross-layer split as ROAD.
  * CIC-IoV-2024 has NO timestamp (per-frame ID + 8 payload bytes) -> the split is ID-STRUCTURE (A) x
    PAYLOAD-CONTENT (B); message-COUNT windows replace time windows. NOT a timing x content cross-layer.

Outputs: results/hcrl_ids/hcrl_detection.csv + metadata.json ; results/cic_ids/cic_detection.csv + metadata.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import road_data as R  # FEATURE_NAMES, BLOCKS, N_FEATURES, _window_features, family_aware_split, _build_cfgs  # noqa: E402
from _study_lib import REPO, RESULTS, now_iso, save_table, write_metadata  # noqa: E402

DATA = REPO / "data"
MODELS = ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest", "xgboost", "mlp"]


# --------------------------------------------------------------------------- #
# HCRL Car-Hacking (timestamped -> timing x content, reuse ROAD window features)
# --------------------------------------------------------------------------- #
HCRL_FILES = {"dos": "DoS_dataset.csv", "fuzzy": "Fuzzy_dataset.csv",
              "spoofing_gear": "gear_dataset.csv", "spoofing_rpm": "RPM_dataset.csv"}


def _load_hcrl_file(path: Path, max_rows: int = 800000) -> pd.DataFrame:
    """Parse a ragged HCRL CSV (ts, id_hex, dlc, b0..b{dlc-1}, Flag) -> normalized df with
    Time, ID, Signal_1_of_ID (= DATA[0]), Label (1 if injected 'T'). Capped to max_rows."""
    # force 12 columns (ts,id,dlc,8 bytes,flag) so ragged short-DLC rows pad with NaN
    # instead of crashing the C parser (which infers width from the first row).
    raw = pd.read_csv(path, header=None, names=list(range(12)), dtype=str, nrows=max_rows)
    arr = raw.to_numpy()
    n = len(raw)
    ts = raw[0].astype(float).to_numpy()
    cid = np.array([int(h, 16) for h in raw[1].to_numpy()])
    dlc = raw[2].astype(int).to_numpy()
    flag = arr[np.arange(n), 3 + dlc]              # Flag sits right after the DLC payload bytes
    label = (flag == "T").astype(int)
    b0 = np.array([int(x, 16) if (isinstance(x, str) and d >= 1) else 0
                   for x, d in zip(raw[3].to_numpy(), dlc)], dtype=float)
    return pd.DataFrame({"Time": ts, "ID": cid, "Signal_1_of_ID": b0, "Label": label})


def _window_df(df: pd.DataFrame, window_s: float, min_msgs: int, max_win: int, rng):
    """Time-window a normalized df into ROAD's 16-feature vectors (reuses R._window_features)."""
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    df["_w"] = np.floor(df["Time"].to_numpy() / window_s).astype(np.int64)
    feats, labels = [], []
    prev: set = set()
    for _, g in df.groupby("_w", sort=True):
        if len(g) < min_msgs:
            prev |= set(g["ID"].to_numpy().tolist())
            continue
        vec, prev = R._window_features(g, prev)
        feats.append(vec)
        labels.append(int((g["Label"].to_numpy() == 1).any()))
    if not feats:
        return np.empty((0, R.N_FEATURES)), np.empty((0,), int)
    X, y = np.vstack(feats), np.asarray(labels, int)
    if max_win and len(y) > max_win:
        sel = np.sort(rng.choice(len(y), max_win, replace=False))
        X, y = X[sel], y[sel]
    return X, y


def load_hcrl(window_s=0.2, min_msgs=20, max_win_per_file=900, seed=7):
    from qkv2x.experiments.common import ExperimentData
    rng = np.random.default_rng(seed)
    Xs, ys, atks, grps = [], [], [], []
    gid = 0
    for fam, fn in HCRL_FILES.items():
        p = DATA / "hcrl_carhacking" / fn
        if not p.exists():
            continue
        X, y = _window_df(_load_hcrl_file(p), window_s, min_msgs, max_win_per_file, rng)
        if len(y) == 0 or int(y.sum()) == 0:
            continue
        Xs.append(X); ys.append(y)
        atks.append(np.where(y == 1, fam, "none"))
        grps.append(np.full(len(y), gid)); gid += 1
    X = np.vstack(Xs); y = np.concatenate(ys)
    return ExperimentData(
        X=X, y=y, attack_name=np.concatenate(atks).astype(str),
        groups=np.concatenate(grps).astype(int), aoi=np.zeros(len(y)),
        t_end=np.arange(len(y), dtype=float), feature_names=list(R.FEATURE_NAMES),
        blocks=list(R.BLOCKS), points=pd.DataFrame({"dataset": ["hcrl"] * len(y)}), meta=pd.DataFrame())


# --------------------------------------------------------------------------- #
# CIC-IoV-2024 (no timestamp -> ID-structure x payload-content, message-count windows)
# --------------------------------------------------------------------------- #
CIC_FILES = {"benign": "decimal_benign.csv", "dos": "decimal_DoS.csv",
             "spoof_gas": "decimal_spoofing-GAS.csv", "spoof_rpm": "decimal_spoofing-RPM.csv",
             "spoof_speed": "decimal_spoofing-SPEED.csv", "spoof_steer": "decimal_spoofing-STEERING_WHEEL.csv"}
CIC_FEATURES = [
    "n_messages", "n_distinct_ids", "top_id_frac", "id_entropy", "id_mean", "id_std", "id_range", "new_id_count",
    "byte_mean", "byte_std", "byte_entropy", "frac_nonzero", "d0_mean", "d0_std", "payload_range", "d0_jump",
]


def _cic_window_feats(ids, B, prev_ids):
    n = len(ids)
    uniq, counts = np.unique(ids, return_counts=True)
    p = counts / counts.sum()
    new_ids = float(len(set(uniq.tolist()) - prev_ids))
    bytes_all = B.reshape(-1)
    vals, vc = np.unique(np.clip(bytes_all, 0, 255).astype(int), return_counts=True)
    pb = vc / vc.sum()
    d0 = B[:, 0].astype(float)
    vec = np.array([
        float(n), float(len(uniq)), float(counts.max() / n), float(-(p * np.log2(p + 1e-12)).sum()),
        float(ids.mean()), float(ids.std()), float(ids.max() - ids.min()), new_ids,
        float(bytes_all.mean()), float(bytes_all.std()), float(-(pb * np.log2(pb + 1e-12)).sum()),
        float((bytes_all > 0).mean()), float(d0.mean()), float(d0.std()),
        float(bytes_all.max() - bytes_all.min()), float(np.abs(np.diff(d0)).max()) if n > 1 else 0.0,
    ], dtype=float)
    return vec, prev_ids | set(uniq.tolist())


def load_cic(win_msgs=60, max_rows_per_file=120000, groups_per_file=8, max_win_per_file=900, seed=7):
    from qkv2x.experiments.common import ExperimentData
    rng = np.random.default_rng(seed)
    Xs, ys, atks, grps = [], [], [], []
    gid = 0
    for fam, fn in CIC_FILES.items():
        p = DATA / "cic_iov_2024" / "decimal" / fn
        if not p.exists():
            continue
        df = pd.read_csv(p, nrows=max_rows_per_file)
        ids = df["ID"].to_numpy()
        B = df[[f"DATA_{i}" for i in range(8)]].to_numpy()
        lab = (df["label"].astype(str).to_numpy() == "ATTACK").astype(int)
        feats, labels = [], []
        prev: set = set()
        for s in range(0, len(df) - win_msgs, win_msgs):
            sl = slice(s, s + win_msgs)
            vec, prev = _cic_window_feats(ids[sl], B[sl], prev)
            feats.append(vec); labels.append(int(lab[sl].any()))
        if not feats:
            continue
        X, y = np.vstack(feats), np.asarray(labels, int)
        if max_win_per_file and len(y) > max_win_per_file:
            sel = np.sort(rng.choice(len(y), max_win_per_file, replace=False)); X, y = X[sel], y[sel]
        an = np.where(y == 1, fam, "none")
        # chunk this file's windows into contiguous pseudo-captures (groups) for leakage-safe split
        chunk = max(1, len(y) // groups_per_file)
        g = gid + (np.arange(len(y)) // chunk)
        Xs.append(X); ys.append(y); atks.append(an); grps.append(g)
        gid = int(g.max()) + 1
    X = np.vstack(Xs); y = np.concatenate(ys)
    return ExperimentData(
        X=X, y=y, attack_name=np.concatenate(atks).astype(str),
        groups=np.concatenate(grps).astype(int), aoi=np.zeros(len(y)),
        t_end=np.arange(len(y), dtype=float), feature_names=list(CIC_FEATURES),
        blocks=list(R.BLOCKS), points=pd.DataFrame({"dataset": ["cic"] * len(y)}), meta=pd.DataFrame())


# --------------------------------------------------------------------------- #
# Generic detection runner (mirrors road_ids.run)
# --------------------------------------------------------------------------- #
def run_detection(data, tag: str, fit_note: str, seed=7, max_windows=2400, n_boot=300):
    from qkv2x.experiments.common import fit_reducer, run_model
    from qkv2x.models.metrics import detection_report, grouped_bootstrap_ci
    import road_ids as RI

    t0 = time.time(); started = now_iso()
    out_dir = RESULTS / f"{tag}_ids"; out_dir.mkdir(parents=True, exist_ok=True)
    cfgs = R._build_cfgs(seed)

    mask = RI._balance_for_quantum(data, max_windows, seed)
    data = data.subset(mask)
    fams = sorted(set(data.attack_name) - {"none"})
    tr, va, te = R.family_aware_split(data, test_frac=0.30, val_frac=0.15, seed=seed)
    reducer = fit_reducer(cfgs["features"]["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)
    idx = {"tr": tr, "va": va, "te": te}
    print(f"[{tag}] windows={len(data)} pos={int(data.y.sum())} ({100*data.y.mean():.1f}%) "
          f"groups={len(set(data.groups))} families={fams}")

    rows = []
    for model in MODELS:
        try:
            res = run_model(model, data, Xr, idx, cfgs)
        except Exception as exc:
            print(f"  {model} FAILED: {exc}"); rows.append({"model": model, "status": f"failed: {exc}"}); continue
        rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
        ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"], n_boot=n_boot, seed=seed)
        rows.append({"model": model, "pr_auc": rep.get("pr_auc"), "roc_auc": rep.get("roc_auc"),
                     "f1": rep.get("f1"), "mcc": rep.get("mcc"), "precision": rep.get("precision"),
                     "recall": rep.get("recall"), "roc_auc_lo": ci["roc_auc"]["lo"],
                     "roc_auc_hi": ci["roc_auc"]["hi"], "n_test": rep["n"], "n_pos": rep["n_pos"]})
        print(f"  {model:16s} PR={rep.get('pr_auc',float('nan')):.3f} ROC={rep.get('roc_auc',float('nan')):.3f} "
              f"F1={rep.get('f1',float('nan')):.3f} MCC={rep.get('mcc',float('nan')):.3f} "
              f"(ROC CI {ci['roc_auc']['lo']:.3f}-{ci['roc_auc']['hi']:.3f})")

    df = pd.DataFrame(rows)
    save_table(df, out_dir, f"{tag}_detection.csv")

    def _get(m, k):
        r = df[df["model"] == m]
        return float(r[k].iloc[0]) if len(r) and pd.notna(r[k].iloc[0]) else None
    cl, zz = _get("qsvm_crosslayer", "roc_auc"), _get("qsvm_zz", "roc_auc")
    best_c = max(((m, _get(m, "roc_auc")) for m in ("random_forest", "xgboost", "rbf_svm", "mlp")),
                 key=lambda x: (x[1] if x[1] is not None else -1))
    headline = {
        "fit_note": fit_note,
        "crosslayer_vs_zz": {"cl_roc": cl, "zz_roc": zz,
                             "beats": bool(cl is not None and zz is not None and cl > zz)},
        "no_advantage": {"best_classical": best_c[0], "best_classical_roc": best_c[1],
                         "qsvm_crosslayer_roc": cl,
                         "no_quantum_advantage": bool(cl is not None and best_c[1] is not None and best_c[1] >= cl)},
    }
    write_metadata(out_dir, study=f"{tag}_ids_detection", started_iso=started, t0=t0,
                   hyperparams={"models": MODELS, "seed": seed, "max_windows": max_windows},
                   results_summary=headline, notes=fit_note)
    (out_dir / f"{tag}_headline.json").write_text(json.dumps(headline, indent=2))
    print(f"[{tag}] cross-layer ROC={cl} vs ZZ={zz} -> beats_zz={headline['crosslayer_vs_zz']['beats']}")
    print(f"[{tag}] best classical {best_c[0]}={best_c[1]} vs qsvm_crosslayer={cl} -> "
          f"no_advantage={headline['no_advantage']['no_quantum_advantage']}")
    return headline


def main() -> None:
    print("=" * 70 + "\nHCRL Car-Hacking (timing x content cross-layer)\n" + "=" * 70)
    try:
        run_detection(load_hcrl(), "hcrl",
                      "HCRL has timestamps -> timing(WHEN) x payload/ID-content(WHAT), same split as ROAD.")
    except Exception as exc:
        print(f"HCRL FAILED: {type(exc).__name__}: {exc}")
    print("\n" + "=" * 70 + "\nCIC-IoV-2024 (ID-structure x payload-content; no timestamps)\n" + "=" * 70)
    try:
        run_detection(load_cic(), "cic",
                      "CIC-IoV has NO timestamp -> ID-structure(A) x payload-content(B), message-count windows; "
                      "NOT a timing x content cross-layer.")
    except Exception as exc:
        print(f"CIC FAILED: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
