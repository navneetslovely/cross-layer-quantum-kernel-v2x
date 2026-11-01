"""External comparability #2 — VeReMi NextGen.

Evaluates the cross-layer quantum-kernel methodology on the published VeReMi
NextGen dataset (Hermann et al., IEEE VNC 2026; MOSAIC+SUMO InTAS scenarios,
15 attack types, official Train/Validation/Test splits).

Two deliberate methodology adaptations (both documented here):

 1. NextGen logs NO RSSI, so the cross-layer F2 residual is anchored on the
    ns-resolution one-way DELAY instead: a linear delay-vs-claimed-distance
    model is calibrated on BENIGN train receptions per (scenario, subset) —
    the timing analogue of veremi_exp.estimate_propagation — and
    e_xl = |observed delay - expected delay at the claimed distance|
    (`network_mode: timing` in crosslayer.py). The F3 Sybil signature features
    correlate per-identity delay series instead of RSSI series.
 2. Splits follow the OFFICIAL NextGen Train/Validation/Test sets (time-split,
    with Train/Validation from a separate simulation area) instead of our
    group split, for comparability with the dataset's own baselines.

A `network_ablation` pass re-runs key models with the network feature block
zeroed (mobility-blind ablation of the cross-layer claim).

The experiment is a sequence of CHECKPOINTED, idempotent stages keyed by files
under <out_dir>/checkpoints, so an interrupted run resumes where it stopped
(`run()` simply executes whatever is still pending). Set the env var
QKV2X_BUDGET_S to a number of seconds to make `run()` return after that budget
with status "in_progress" (used by constrained sandboxes); without it, `run()`
executes to completion exactly like every other experiment driver.
"""
from __future__ import annotations

import json
import os
import time
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from ..config import load_yaml
from ..data.veremi_nextgen import (NEXTGEN_SPLITS, estimate_delay_model,
                                   load_nextgen_files)
from ..features.dataset import build_dataset
from ..features.reduce import BlockReducer
from ..models.metrics import (detection_latency, detection_report,
                              grouped_bootstrap_ci, macro_report)
from ..models.qsvm import QuantumKernelMultiSVM, QuantumKernelSVM
from ..quantum.kernels import fidelity_gram, psd_clip
from ..utils.provenance import capture_provenance
from .common import ExperimentData, make_feature_map, predictions_long, run_model

_T0 = time.time()


def _savez_atomic(path: Path, **arrays):
    """np.savez_compressed via a temp file + rename, so a killed process can
    never leave a half-written (corrupt) checkpoint behind."""
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


class BudgetExceeded(Exception):
    """Raised between stages when QKV2X_BUDGET_S is exhausted (resume later)."""


def _check_budget():
    budget = os.environ.get("QKV2X_BUDGET_S")
    if budget and (time.time() - _T0) > float(budget):
        raise BudgetExceeded(f"stage budget {budget}s exhausted")


# --------------------------------------------------------------------------- #
# Stage 1 — extraction (outer zip -> Train/Validation/Test inner zips -> json)
# --------------------------------------------------------------------------- #
def _extract_inner(inner_bytes: bytes, dest: Path):
    """Extract veh_*.json members (file-level resumable; some inner zips use
    backslash separators). Budget-checked every few hundred files so a large
    split resumes rather than restarting."""
    import io
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as zf:
        for i, m in enumerate(zf.namelist()):
            name = m.replace("\\", "/").split("/")[-1]
            if not (name.startswith("veh_") and name.endswith(".json")):
                continue
            target = dest / name
            if target.exists():
                continue
            target.write_bytes(zf.read(m))
            if i % 200 == 0:
                _check_budget()


def _stage_extract(root: Path, scenario: str, subset: str) -> Path:
    """Extract one (scenario, subset) outer zip into extracted/<sc>/<sub>/<Split>."""
    outer_name = scenario if subset == "none" else f"{scenario}_{subset}"
    out_base = root / "extracted" / scenario / subset
    marker = out_base / ".done"
    if marker.exists():
        return out_base
    outer = root / "dl" / f"{outer_name}.zip"
    if not outer.exists():
        raise FileNotFoundError(
            f"NextGen zip not found: {outer} — run scripts/fetch_veremi_nextgen.sh")
    with zipfile.ZipFile(outer) as zf:
        for split in NEXTGEN_SPLITS:
            split_marker = out_base / split / ".extracted"
            if split_marker.exists():
                continue
            members = [m for m in zf.namelist()
                       if f"/{split}/" in m.replace("\\", "/") and m.endswith(".zip")]
            if not members:
                raise FileNotFoundError(f"{outer}: no inner zip for split {split}")
            _extract_inner(zf.read(members[0]), out_base / split)
            split_marker.write_text(datetime.now().isoformat())
            _check_budget()
    marker.write_text(datetime.now().isoformat())
    return out_base


# --------------------------------------------------------------------------- #
# Stage 2 — parse json -> canonical parquet shards (checkpoint per shard)
# --------------------------------------------------------------------------- #
def _stage_parse(ck: Path, extracted: Path, scenario: str, subset: str,
                 split: str, n_shards: int = 8) -> Path:
    """Shard receiver files by rx id and parse shard-by-shard (resumable)."""
    out = ck / f"parsed_{scenario}_{subset}_{split}"
    if (out / ".done").exists():
        return out
    out.mkdir(parents=True, exist_ok=True)
    files = sorted((extracted / split).glob("veh_*.json"))
    if not files:
        raise FileNotFoundError(f"no veh_*.json under {extracted / split}")
    attack_name = subset if subset != "none" else "none"
    for k in range(n_shards):
        part = out / f"shard{k}.parquet"
        if part.exists():
            continue
        _check_budget()
        batch = [f for i, f in enumerate(files) if i % n_shards == k]
        if not batch:
            pd.DataFrame().to_parquet(part)
            continue
        df = load_nextgen_files(batch, attack_name)
        df.to_parquet(part)
    (out / ".done").write_text(str(len(files)))
    return out


# --------------------------------------------------------------------------- #
# Stage 3 — benign delay-model calibration per (scenario, subset)
# --------------------------------------------------------------------------- #
def _stage_delaycal(ck: Path, scenario: str, subset: str) -> dict:
    path = ck / f"delay_{scenario}_{subset}.json"
    if path.exists():
        return json.loads(path.read_text())
    parsed = ck / f"parsed_{scenario}_{subset}_Train"
    frames = [pd.read_parquet(p) for p in sorted(parsed.glob("shard*.parquet"))]
    rx = pd.concat([f for f in frames if len(f)], ignore_index=True)
    model = estimate_delay_model(rx)
    path.write_text(json.dumps(model))
    return model


# --------------------------------------------------------------------------- #
# Stage 4 — windowed features per (scenario, subset, split)
# --------------------------------------------------------------------------- #
def _stage_windows(ck: Path, features_cfg: dict, scenario: str, subset: str,
                   split: str, propagation: dict, nominal_rate_hz: float,
                   n_shards: int = 6, tag: str = "win",
                   store_sequences: bool = False) -> Path:
    """Windowed features, sharded by receiver (exact: windows are built per
    (receiver, claimed-id) stream — or per receiver in scope=receiver — so
    receiver-sharding changes nothing) and checkpointed per shard so large
    splits resume under a budget. `tag` namespaces variant window sets (e.g.
    the receiver-scope Sybil pass; `winseq` adds per-window message sequences
    for the LSTM baseline, earlier)."""
    path = ck / f"{tag}_{scenario}_{subset}_{split}.npz"
    if path.exists():
        return path
    parsed = ck / f"parsed_{scenario}_{subset}_{split}"
    frames = [pd.read_parquet(p) for p in sorted(parsed.glob("shard*.parquet"))]
    rx = pd.concat([f for f in frames if len(f)], ignore_index=True)

    parts, names, blocks = [], None, None
    for k in range(n_shards):
        part_path = ck / f"{tag}_{scenario}_{subset}_{split}.part{k}.npz"
        if part_path.exists():
            parts.append(part_path)
            continue
        _check_budget()
        sub_rx = rx[rx["rx_node"] % n_shards == k]
        try:
            ds = build_dataset(sub_rx.reset_index(drop=True), features_cfg,
                               propagation, nominal_rate_hz,
                               store_sequences=store_sequences)
            payload = dict(X=ds.X, y=ds.y, attack_name=ds.attack_name.astype(str),
                           groups=ds.groups, aoi=ds.aoi,
                           t_end=ds.meta["t_end"].to_numpy(),
                           feature_names=np.asarray(ds.feature_names, dtype=str),
                           blocks=np.asarray(ds.blocks, dtype=str))
            if store_sequences:
                payload["sequences"] = ds.sequences
                payload["seq_lengths"] = ds.seq_lengths
                payload["seq_sources"] = np.asarray(ds.seq_sources, dtype=str)
        except ValueError:   # shard produced no windows (sparse receivers)
            payload = dict(X=np.zeros((0, 0)), y=np.zeros(0, dtype=np.int64),
                           attack_name=np.asarray([], dtype=str),
                           groups=np.zeros(0, dtype=np.int64), aoi=np.zeros(0),
                           t_end=np.zeros(0),
                           feature_names=np.asarray([], dtype=str),
                           blocks=np.asarray([], dtype=str))
            if store_sequences:
                payload["sequences"] = np.zeros((0, 1, 1), dtype=np.float32)
                payload["seq_lengths"] = np.zeros(0, dtype=np.int64)
                payload["seq_sources"] = np.asarray([], dtype=str)
        _savez_atomic(part_path, **payload)
        parts.append(part_path)

    zs = [np.load(p, allow_pickle=True) for p in parts]
    zs_full = [z for z in zs if z["y"].size]
    if not zs_full:
        raise ValueError(f"no windows produced for {scenario}/{subset}/{split}")
    names = zs_full[0]["feature_names"]
    blocks = zs_full[0]["blocks"]
    payload = dict(
        X=np.vstack([z["X"] for z in zs_full]),
        y=np.concatenate([z["y"] for z in zs_full]),
        attack_name=np.concatenate([z["attack_name"] for z in zs_full]).astype(str),
        groups=np.concatenate([z["groups"] for z in zs_full]),
        aoi=np.concatenate([z["aoi"] for z in zs_full]),
        t_end=np.concatenate([z["t_end"] for z in zs_full]),
        feature_names=names, blocks=blocks)
    if store_sequences:
        payload["sequences"] = np.vstack([z["sequences"] for z in zs_full])
        payload["seq_lengths"] = np.concatenate([z["seq_lengths"] for z in zs_full])
        payload["seq_sources"] = zs_full[0]["seq_sources"]
    _savez_atomic(path, **payload)
    for p in parts:   # tidy shard parts once merged
        try:
            p.unlink()
        except OSError:
            pass
    return path


# --------------------------------------------------------------------------- #
# Stage 5 — assemble official-split dataset (stratified caps for O(N^2) kernel)
# --------------------------------------------------------------------------- #
def _cap_stratified(y, attack_name, cap, rng):
    """Stratified subsample indices: proportional per (attack_name) stratum,
    at least 1 from each non-empty stratum."""
    n = len(y)
    if cap is None or n <= int(cap):
        return np.arange(n)
    cap = int(cap)
    strata = pd.Series(attack_name).astype(str)
    out = []
    for name, idx in strata.groupby(strata).groups.items():
        idx = np.asarray(idx)
        take = max(1, int(round(cap * len(idx) / n)))
        out.append(rng.choice(idx, size=min(take, len(idx)), replace=False))
    sel = np.concatenate(out)
    if len(sel) > cap:
        sel = rng.choice(sel, size=cap, replace=False)
    return np.sort(sel)


def _subsets(cfg: dict) -> list[str]:
    """Subsets in scope. include_baseline: false drops the no-attack baselines
    (they lack the distance_to_road_edge enrichment; the attack subsets are
    majority-benign, so benign coverage stays large)."""
    base = ["none"] if cfg.get("include_baseline", True) else []
    return base + list(cfg["attacks"])


def _stage_assemble(ck: Path, cfg: dict, features_cfg: dict, seed: int,
                    caps: dict, fname: str, tag: str = "win",
                    with_seq: bool = False) -> Path:
    """Concatenate the per-(scenario,subset,split) window sets in a FIXED order,
    apply the stratified caps with `seed`, and save as <fname>. The fixed order
    + seeded rng make the selection reproducible, so a sequence-tagged assembly
    (`tag=winseq`, same seed/caps) selects EXACTLY the same windows."""
    path = ck / fname
    if path.exists():
        return path
    _check_budget()
    rng = np.random.default_rng(int(seed))

    parts = {s: [] for s in NEXTGEN_SPLITS}
    feature_names = blocks = seq_sources = None
    run_idx = 0
    for scenario in cfg["scenarios"]:
        for subset in _subsets(cfg):
            for split in NEXTGEN_SPLITS:
                z = np.load(ck / f"{tag}_{scenario}_{subset}_{split}.npz",
                            allow_pickle=True)
                feature_names = [str(s) for s in z["feature_names"]]
                blocks = [str(s) for s in z["blocks"]]
                rec = {
                    "X": z["X"], "y": z["y"],
                    "attack_name": z["attack_name"].astype(object),
                    "groups": z["groups"] + run_idx * 10_000_000,
                    "aoi": z["aoi"], "t_end": z["t_end"],
                    "scenario": scenario, "subset": subset,
                }
                if with_seq:
                    rec["sequences"] = z["sequences"]
                    rec["seq_lengths"] = z["seq_lengths"]
                    seq_sources = z["seq_sources"]
                parts[split].append(rec)
            run_idx += 1

    keys = ["X", "y", "attack_name", "groups", "aoi", "t_end",
            "scenario", "subset"] + (["sequences", "seq_lengths"] if with_seq else [])

    def _concat(split):
        ps = parts[split]
        out = {}
        for k in keys:
            if k in ("X", "sequences"):
                out[k] = np.vstack([p[k] for p in ps])
            elif k in ("scenario", "subset"):
                out[k] = np.concatenate(
                    [np.full(len(p["y"]), p[k], dtype=object) for p in ps])
            else:
                out[k] = np.concatenate([p[k] for p in ps])
        return out

    split_tag, n_total = [], {}
    arrays = {k: [] for k in keys}
    for split in NEXTGEN_SPLITS:
        d = _concat(split)
        n_total[split] = int(len(d["y"]))
        sel = _cap_stratified(d["y"], d["attack_name"], caps.get(split), rng)
        for k in keys:
            arrays[k].append(d[k][sel])
        split_tag.append(np.full(len(sel), split, dtype=object))

    payload = dict(
        X=np.vstack(arrays["X"]),
        y=np.concatenate(arrays["y"]),
        attack_name=np.concatenate(arrays["attack_name"]).astype(str),
        groups=np.concatenate(arrays["groups"]),
        aoi=np.concatenate(arrays["aoi"]),
        t_end=np.concatenate(arrays["t_end"]),
        scenario=np.concatenate(arrays["scenario"]).astype(str),
        subset=np.concatenate(arrays["subset"]).astype(str),
        split=np.concatenate(split_tag).astype(str),
        feature_names=np.asarray(feature_names, dtype=str),
        blocks=np.asarray(blocks, dtype=str),
        n_total_json=np.asarray([json.dumps(n_total)], dtype=str))
    if with_seq:
        payload["sequences"] = np.vstack(arrays["sequences"])
        payload["seq_lengths"] = np.concatenate(arrays["seq_lengths"])
        payload["seq_sources"] = seq_sources
    _savez_atomic(path, **payload)
    return path


def _load_assembled(ck: Path, fname: str = "assembled.npz"):
    z = np.load(ck / fname, allow_pickle=True)
    split = z["split"].astype(str)
    idx = {"tr": np.where(split == "Train")[0],
           "va": np.where(split == "Validation")[0],
           "te": np.where(split == "Test")[0]}
    data = ExperimentData(
        X=z["X"], y=z["y"], attack_name=z["attack_name"].astype(object),
        groups=z["groups"], aoi=z["aoi"], t_end=z["t_end"],
        feature_names=[str(s) for s in z["feature_names"]],
        blocks=[str(s) for s in z["blocks"]],
        points=pd.DataFrame({"scenario": z["scenario"].astype(str),
                             "subset": z["subset"].astype(str),
                             "split": split}),
        meta=pd.DataFrame({"t_end": z["t_end"]}),
    )
    n_total = json.loads(str(z["n_total_json"][0]))
    return data, idx, n_total


# --------------------------------------------------------------------------- #
# Stage 6 — block reduction (full + network-blind ablation)
# --------------------------------------------------------------------------- #
def _stage_reduce(ck: Path, features_cfg: dict, data: ExperimentData,
                  idx: dict, fname: str = "reduced.npz") -> Path:
    path = ck / fname
    if path.exists():
        return path
    _check_budget()
    reducer = BlockReducer(features_cfg["reduction"]).fit(
        data.X[idx["tr"]], data.blocks)
    Xr = reducer.transform(data.X)
    # network-blind ablation: zero the network features BEFORE fitting, so the
    # network register carries no information (what does the network block add?)
    net_cols = np.array([i for i, b in enumerate(data.blocks) if b == "network"])
    X_abl = data.X.copy()
    X_abl[:, net_cols] = 0.0
    reducer_abl = BlockReducer(features_cfg["reduction"]).fit(
        X_abl[idx["tr"]], data.blocks)
    Xr_abl = reducer_abl.transform(X_abl)
    _savez_atomic(path, Xr=Xr, Xr_abl=Xr_abl)
    return path


# --------------------------------------------------------------------------- #
# Stage 7 — quantum-map training, checkpointed epoch-by-epoch
# --------------------------------------------------------------------------- #
def _qtrain_checkpointed(fm, X, y, aoi, train_cfg: dict, ck_path: Path,
                         multiclass: bool):
    """Replicates quantum.train.train_feature_map(_multiclass) EXACTLY, but with
    full state checkpointing (params, Adam state, numpy RNG state, best-so-far)
    after every epoch, so an interrupted run resumes mid-training with identical
    results to an uninterrupted one."""
    import torch

    from ..quantum.train import (_balanced_batch, _centered_kta_torch,
                                 _centered_kta_with_target_torch,
                                 _multiclass_target_torch, _stratified_batch,
                                 _torch_fidelity_gram)

    epochs = int(train_cfg.get("epochs", 120))
    batch = int(train_cfg.get("batch", 512))
    chunk = int(train_cfg.get("chunk", 256))

    if ck_path.exists():
        st = torch.load(ck_path, weights_only=False)
        params = {k: v.clone().requires_grad_(True) for k, v in st["params"].items()}
        opt = torch.optim.Adam([params["w"], params["u"], params["gamma"]],
                               lr=float(train_cfg.get("lr", 0.05)))
        opt.load_state_dict(st["opt"])
        rng = np.random.default_rng()
        rng.bit_generator.state = st["rng"]
        start_ep = int(st["epoch"])
        best_align = float(st["best_align"])
        best_params = {k: v.clone() for k, v in st["best_params"].items()}
        history = st["history"]
    else:
        rng = np.random.default_rng(int(train_cfg.get("seed", 11)))
        params = fm.init_params(init_w=train_cfg.get("init_w", 1.0),
                                init_u=train_cfg.get("init_u", 1.0),
                                init_gamma=train_cfg.get("init_gamma", 0.5),
                                requires_grad=True)
        opt = torch.optim.Adam([params["w"], params["u"], params["gamma"]],
                               lr=float(train_cfg.get("lr", 0.05)))
        start_ep = 0
        best_align = -np.inf
        best_params = {k: v.detach().clone() for k, v in params.items()}
        history = {"alignment": []}

    ypm = np.where(np.asarray(y) > 0, 1.0, -1.0).astype(np.float64)

    for ep in range(start_ep, epochs):
        idx = (_stratified_batch(np.asarray(y), batch, rng) if multiclass
               else _balanced_batch(np.asarray(y), batch, rng))
        Xb = X[idx]
        ab = aoi[idx] if aoi is not None else None
        opt.zero_grad()
        states = fm.statevectors(Xb, ab, params, chunk=chunk)
        K = _torch_fidelity_gram(states)
        if multiclass:
            yb = torch.as_tensor(np.asarray(y)[idx].astype(np.float64),
                                 dtype=fm.rdtype, device=fm.torch_device)
            align = _centered_kta_with_target_torch(K, _multiclass_target_torch(yb))
        else:
            yb = torch.as_tensor(ypm[idx], dtype=fm.rdtype, device=fm.torch_device)
            align = _centered_kta_torch(K, yb)
        loss = -align
        loss.backward()
        opt.step()
        a = float(align.detach().cpu())
        history["alignment"].append(a)
        if a > best_align:
            best_align = a
            best_params = {k: v.detach().clone() for k, v in params.items()}
        torch.save({"params": {k: v.detach().clone() for k, v in params.items()},
                    "opt": opt.state_dict(), "rng": rng.bit_generator.state,
                    "epoch": ep + 1, "best_align": best_align,
                    "best_params": best_params, "history": history}, ck_path)
        if ep + 1 < epochs:
            _check_budget()

    for v in best_params.values():
        v.requires_grad_(False)
    history["best_alignment"] = best_align
    return best_params, history


def _quantum_params(model, task, variant, fm, data, Xr, tr, cfgs, ck: Path,
                    suffix: str = ""):
    """Trained (or fixed, for qsvm_zz) feature-map params, checkpointed."""
    train_cfg = cfgs["quantum"].get("train", {})
    if model == "qsvm_zz" or not train_cfg.get("enabled", True):
        return fm.fixed_params()
    y_for_train = data.y[tr] if task == "binary" else data._y_multi[tr]
    params, _ = _qtrain_checkpointed(
        fm, Xr[tr], y_for_train, data.aoi[tr], train_cfg,
        ck / f"qtrain_{task}_{variant}_{model}{suffix}.pt",
        multiclass=(task == "multiclass"))
    return params


# --------------------------------------------------------------------------- #
# Stage 8 — quantum Gram + SVM evaluation (checkpointed Grams)
# --------------------------------------------------------------------------- #
def _quantum_grams(model, task, variant, fm, params, Xr, aoi, tr, te,
                   cfgs, ck: Path, suffix: str = ""):
    gpath = ck / f"gram_{task}_{variant}_{model}{suffix}.npz"
    if gpath.exists():
        try:
            z = np.load(gpath)
            return z["K_tr"], z["K_te"]
        except Exception:
            pass   # half-written/corrupt checkpoint — regenerate (atomic save)
    _check_budget()
    s_tr = fm.statevectors(Xr[tr], aoi[tr], params)
    K_tr = fidelity_gram(s_tr)
    s_te = fm.statevectors(Xr[te], aoi[te], params)
    K_te = fidelity_gram(s_te, s_tr)
    shots = cfgs["quantum"].get("shots", None)
    if shots:
        rng = np.random.default_rng(0)
        from ..quantum.kernels import add_shot_noise
        K_tr = add_shot_noise(K_tr, shots, rng)
        np.fill_diagonal(K_tr, 1.0)
        K_te = add_shot_noise(K_te, shots, rng)
        if cfgs["quantum"].get("psd_repair", "clip") == "clip":
            K_tr = psd_clip(K_tr)
    _savez_atomic(gpath, K_tr=K_tr, K_te=K_te)
    return K_tr, K_te


def _eval_binary_model(model, variant, data, Xr, idx, cfgs, ck: Path,
                       n_boot: int, seed: int, suffix: str = "") -> dict:
    rpath = ck / f"res2_binary_{variant}_{model}{suffix}.json"
    if rpath.exists():
        return json.loads(rpath.read_text())
    _check_budget()
    tr, va, te = idx["tr"], idx["va"], idx["te"]

    if model in ("qsvm_crosslayer", "qsvm_zz"):
        fm = make_feature_map(model, cfgs["quantum"],
                              cfgs["features"]["reduction"], cfgs["features"])
        params = _quantum_params(model, "binary", variant, fm, data, Xr, tr,
                                 cfgs, ck, suffix=suffix)
        K_tr, K_te = _quantum_grams(model, "binary", variant, fm, params,
                                    Xr, data.aoi, tr, te, cfgs, ck,
                                    suffix=suffix)
        svm = QuantumKernelSVM(C_grid=cfgs["quantum"]["svm"]["C_grid"],
                               cv_folds=cfgs["quantum"]["svm"]["cv_folds"],
                               class_weight=cfgs["quantum"]["svm"].get(
                                   "class_weight", "balanced"))
        svm.fit(K_tr, data.y[tr], groups=data.groups[tr])
        scores = svm.decision_function(K_te)
        preds = svm.predict(K_te)
        res = {"model": model, "y_true": data.y[te], "scores": scores,
               "y_pred": preds, "attack_name": data.attack_name[te],
               "groups": data.groups[te], "t_end": data.t_end[te]}
    else:
        res = run_model(model, data, Xr, {"tr": tr, "va": va, "te": te}, cfgs)

    rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
    lat = detection_latency(res["t_end"], res["y_true"], res["y_pred"], res["groups"])
    ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"],
                              n_boot=n_boot, seed=seed)
    row = {"variant": variant, "model": model, "seed": int(seed),
           **{k: rep.get(k) for k in ["pr_auc", "roc_auc", "f1", "mcc",
                                      "precision", "recall", "fpr_at_tpr"]},
           "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
           "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
           "detection_latency_mean": lat["detection_latency_mean"],
           "detected_fraction": lat["detected_fraction"],
           "n_test": rep["n"], "n_pos": rep["n_pos"]}
    out = {"row": row,
           "pred_rows": predictions_long(res, task="binary", variant=variant,
                                         seed=int(seed))}
    rpath.write_text(json.dumps(out, default=float))
    return out


# --------------------------------------------------------------------------- #
# LSTM sequence baseline (published-SOTA-style recurrent comparator)
# --------------------------------------------------------------------------- #
def _eval_lstm(data, seqs, seq_lengths, idx, cfgs, ck: Path,
               n_boot: int, seed: int) -> dict:
    rpath = ck / "res2_binary_full_lstm_seq.json"
    if rpath.exists():
        return json.loads(rpath.read_text())
    _check_budget()
    from ..models.seq_baselines import LSTMDetector
    lc = cfgs["baselines"].get("lstm", {})
    tr, va, te = idx["tr"], idx["va"], idx["te"]
    det = LSTMDetector(hidden=int(lc.get("hidden", 64)),
                       layers=int(lc.get("layers", 1)),
                       epochs=int(lc.get("epochs", 40)),
                       batch=int(lc.get("batch", 256)),
                       lr=float(lc.get("lr", 1e-3)), seed=seed)
    det.fit(seqs[tr], seq_lengths[tr], data.y[tr],
            seqs[va], seq_lengths[va], data.y[va])
    scores = det.decision_function(seqs[te], seq_lengths[te])
    preds = (scores > 0.0).astype(int)
    res = {"model": "lstm_seq", "y_true": data.y[te], "scores": scores,
           "y_pred": preds, "attack_name": data.attack_name[te],
           "groups": data.groups[te], "t_end": data.t_end[te]}
    rep = detection_report(res["y_true"], res["scores"], res["y_pred"])
    lat = detection_latency(res["t_end"], res["y_true"], res["y_pred"], res["groups"])
    ci = grouped_bootstrap_ci(res["y_true"], res["scores"], res["groups"],
                              n_boot=n_boot, seed=seed)
    row = {"variant": "full", "model": "lstm_seq", "seed": int(seed),
           **{k: rep.get(k) for k in ["pr_auc", "roc_auc", "f1", "mcc",
                                      "precision", "recall", "fpr_at_tpr"]},
           "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
           "roc_auc_lo": ci["roc_auc"]["lo"], "roc_auc_hi": ci["roc_auc"]["hi"],
           "detection_latency_mean": lat["detection_latency_mean"],
           "detected_fraction": lat["detected_fraction"],
           "n_test": rep["n"], "n_pos": rep["n_pos"],
           "best_val_ap": det.best_val_ap_}
    out = {"row": row,
           "pred_rows": predictions_long(res, task="binary", variant="full",
                                         seed=int(seed))}
    rpath.write_text(json.dumps(out, default=float))
    return out


def _eval_multiclass_model(model, data, Xr, idx, cfgs, ck: Path,
                           label_ids, label_names) -> dict:
    rpath = ck / f"res2_multiclass_full_{model}.json"
    if rpath.exists():
        return json.loads(rpath.read_text())
    _check_budget()
    tr, te = idx["tr"], idx["te"]
    y_multi = data._y_multi

    if model in ("qsvm_crosslayer", "qsvm_zz"):
        fm = make_feature_map(model, cfgs["quantum"],
                              cfgs["features"]["reduction"], cfgs["features"])
        params = _quantum_params(model, "multiclass", "full", fm, data, Xr, tr,
                                 cfgs, ck)
        K_tr, K_te = _quantum_grams(model, "multiclass", "full", fm, params,
                                    Xr, data.aoi, tr, te, cfgs, ck)
        svm = QuantumKernelMultiSVM(C_grid=cfgs["quantum"]["svm"]["C_grid"],
                                    cv_folds=cfgs["quantum"]["svm"]["cv_folds"])
        svm.fit(K_tr, y_multi[tr], groups=data.groups[tr])
        pred = svm.predict(K_te)
    else:
        from ..models.baselines import build_multiclass
        est = build_multiclass(model, cfgs["baselines"],
                               seed=int(cfgs["baselines"].get("seed", 0)))
        est.fit(Xr[tr], y_multi[tr])
        pred = est.predict(Xr[te])

    rep = macro_report(y_multi[te], pred, labels=label_ids)
    out = {"row": {"model": model, "macro_precision": rep["macro_precision"],
                   "macro_recall": rep["macro_recall"],
                   "macro_f1": rep["macro_f1"]},
           "confusion": {"labels": label_names, "confusion": rep["confusion"]}}
    rpath.write_text(json.dumps(out, default=float))
    return out


# --------------------------------------------------------------------------- #
# Receiver-scope Sybil pass
# --------------------------------------------------------------------------- #
# NextGen's trafficCongestionSybil ghosts are ONE-SHOT pseudonyms (~3 msgs per
# identity), invisible to per-identity stream windows (< min_messages). The
# receiver-scope pass pools everything one receiver hears into one time-ordered
# stream, so ghost storms surface through the neighbourhood features
# (sybil_neighbors, claim_overlap, rate_ratio) and become detectable.
def _sybil_receiver_pass(ck: Path, cfg: dict, features_cfg: dict, cfgs: dict,
                         nominal: float, n_boot: int, seed: int):
    sub = cfg.get("sybil_receiver", {}).get("subset", "trafficCongestionSybil")
    if sub not in cfg["attacks"]:
        return []
    import copy
    fcfg = copy.deepcopy(features_cfg)
    fcfg["window"]["scope"] = "receiver"

    for sc in cfg["scenarios"]:
        delay_model = _stage_delaycal(ck, sc, sub)
        for split in NEXTGEN_SPLITS:
            _stage_windows(ck, fcfg, sc, sub, split, delay_model, nominal,
                           tag="winrecv")

    # assemble (official splits; stratified caps)
    apath = ck / "assembled_sybil_receiver.npz"
    if not apath.exists():
        _check_budget()
        caps = cfg.get("sybil_receiver", {}).get("max_windows",
                                                 cfg.get("max_windows", {}))
        rng = np.random.default_rng(seed)
        arrays = {k: [] for k in
                  ["X", "y", "attack_name", "groups", "aoi", "t_end"]}
        split_tag, n_total = [], {}
        feature_names = blocks = None
        for split in NEXTGEN_SPLITS:
            ps = []
            for run_idx, sc in enumerate(cfg["scenarios"]):
                z = np.load(ck / f"winrecv_{sc}_{sub}_{split}.npz",
                            allow_pickle=True)
                feature_names = [str(s) for s in z["feature_names"]]
                blocks = [str(s) for s in z["blocks"]]
                ps.append({"X": z["X"], "y": z["y"],
                           "attack_name": z["attack_name"].astype(object),
                           "groups": z["groups"] + run_idx * 10_000_000,
                           "aoi": z["aoi"], "t_end": z["t_end"]})
            d = {k: (np.vstack if k == "X" else np.concatenate)([p[k] for p in ps])
                 for k in arrays}
            n_total[split] = int(len(d["y"]))
            sel = _cap_stratified(d["y"], d["attack_name"], caps.get(split), rng)
            for k in arrays:
                arrays[k].append(d[k][sel])
            split_tag.append(np.full(len(sel), split, dtype=object))
        _savez_atomic(
            apath, X=np.vstack(arrays["X"]),
            y=np.concatenate(arrays["y"]),
            attack_name=np.concatenate(arrays["attack_name"]).astype(str),
            groups=np.concatenate(arrays["groups"]),
            aoi=np.concatenate(arrays["aoi"]),
            t_end=np.concatenate(arrays["t_end"]),
            split=np.concatenate(split_tag).astype(str),
            feature_names=np.asarray(feature_names, dtype=str),
            blocks=np.asarray(blocks, dtype=str),
            n_total_json=np.asarray([json.dumps(n_total)], dtype=str))

    z = np.load(apath, allow_pickle=True)
    split = z["split"].astype(str)
    idx = {"tr": np.where(split == "Train")[0],
           "va": np.where(split == "Validation")[0],
           "te": np.where(split == "Test")[0]}
    data = ExperimentData(
        X=z["X"], y=z["y"], attack_name=z["attack_name"].astype(object),
        groups=z["groups"], aoi=z["aoi"], t_end=z["t_end"],
        feature_names=[str(s) for s in z["feature_names"]],
        blocks=[str(s) for s in z["blocks"]],
        points=pd.DataFrame({"split": split}),
        meta=pd.DataFrame({"t_end": z["t_end"]}),
    )

    rpath = ck / "reduced_sybil_receiver.npz"
    if not rpath.exists():
        _check_budget()
        reducer = BlockReducer(features_cfg["reduction"]).fit(
            data.X[idx["tr"]], data.blocks)
        _savez_atomic(rpath, Xr=reducer.transform(data.X))
    Xr = np.load(rpath)["Xr"]

    outs = []
    models = cfg.get("sybil_receiver", {}).get(
        "models", ["qsvm_crosslayer", "qsvm_zz", "rbf_svm", "random_forest"])
    for model in models:
        outs.append(_eval_binary_model(model, "sybil_receiver", data, Xr, idx,
                                       cfgs, ck, n_boot, seed))
    return outs


# --------------------------------------------------------------------------- #
# Per-attack detectability report (from the persisted per-window predictions)
# --------------------------------------------------------------------------- #
def _per_attack_rows(pred_rows: list[dict]) -> list[dict]:
    from sklearn.metrics import average_precision_score, roc_auc_score
    df = pd.DataFrame(pred_rows)
    rows = []
    for (variant, model), g in df.groupby(["variant", "model"]):
        benign = g[g.y_true == 0]
        for atk in sorted(set(g.attack_name) - {"none"}):
            sub = pd.concat([g[g.attack_name == atk], benign])
            if sub.y_true.nunique() < 2:
                continue
            rows.append({
                "variant": variant, "model": model, "attack": atk,
                "pr_auc": average_precision_score(sub.y_true, sub.score),
                "roc_auc": roc_auc_score(sub.y_true, sub.score),
                "n_pos": int(sub.y_true.sum()), "n_benign": int(len(benign)),
            })
    return rows


# --------------------------------------------------------------------------- #
# Nyström-at-scale pass: removes the matched-N asterisk
# --------------------------------------------------------------------------- #
# The dense fidelity Gram caps quantum training at ~3k windows. Here the
# official splits are re-assembled at LARGE caps and the quantum SVMs use the
# explicit Nyström feature map Phi = C W^{-1/2} (O(N·l) quantum overlaps and
# memory) with a LINEAR SVM — mathematically the Nyström approximation of the
# same fidelity kernel — while tree/boosting classical baselines train on the
# SAME N. Model selection for every model is on the official validation split.
_SCALE_GRIDS = {
    "random_forest": [{"n_estimators": n, "max_depth": d}
                      for n in (200, 500) for d in (None, 20)],
    "xgboost": [{"n_estimators": n, "max_depth": d, "learning_rate": 0.1}
                for n in (200, 400) for d in (6, 8)],
}


def _scale_quantum_scores(model, data, Xr, idx, cfgs, ck, scfg, seed):
    """Trained map -> statevectors -> Nyström features -> linear SVM scores."""
    from sklearn.metrics import average_precision_score
    from sklearn.svm import LinearSVC

    from ..quantum.kernels import nystrom_features

    tr, va, te = idx["tr"], idx["va"], idx["te"]
    fm = make_feature_map(model, cfgs["quantum"],
                          cfgs["features"]["reduction"], cfgs["features"])
    params = _quantum_params(model, "binary", "scale", fm, data, Xr, tr,
                             cfgs, ck)
    fpath = ck / f"nyfeat_scale_{model}.npz"
    if fpath.exists():
        z = np.load(fpath)
        Phi_tr, Phi_va, Phi_te = z["Phi_tr"], z["Phi_va"], z["Phi_te"]
    else:
        _check_budget()
        ell = int(scfg.get("landmarks", 512))
        rng = np.random.default_rng(int(scfg.get("landmark_seed", 13)))
        lm = np.sort(rng.choice(len(tr), size=min(ell, len(tr)), replace=False))
        s_tr = fm.statevectors(Xr[tr], data.aoi[tr], params)
        s_lm = s_tr[lm]
        W = fidelity_gram(s_lm)
        Phi_tr = nystrom_features(fidelity_gram(s_tr, s_lm), W)
        del s_tr
        s_va = fm.statevectors(Xr[va], data.aoi[va], params)
        Phi_va = nystrom_features(fidelity_gram(s_va, s_lm), W)
        del s_va
        s_te = fm.statevectors(Xr[te], data.aoi[te], params)
        Phi_te = nystrom_features(fidelity_gram(s_te, s_lm), W)
        _savez_atomic(fpath, Phi_tr=Phi_tr, Phi_va=Phi_va, Phi_te=Phi_te)

    best = (-np.inf, None)
    for C in scfg.get("C_grid", [0.01, 0.1, 1.0, 10.0]):
        _check_budget()
        svm = LinearSVC(C=float(C), class_weight="balanced", max_iter=5000)
        svm.fit(Phi_tr, data.y[tr])
        ap = average_precision_score(data.y[va], svm.decision_function(Phi_va))
        if ap > best[0]:
            best = (ap, svm)
    svm = best[1]
    return svm.decision_function(Phi_te), svm.predict(Phi_te)


def _scale_classical_scores(model, data, idx, cfgs, ck, seed):
    """Per-config checkpointed validation-split model selection at large N
    (GridSearchCV's inner CV would not fit the budgeted stage execution)."""
    from sklearn.metrics import average_precision_score

    from ..models.baselines import baseline_scores, build_baseline

    tr, va, te = idx["tr"], idx["va"], idx["te"]
    grid = _SCALE_GRIDS[model]
    val_aps = []
    for ci, params in enumerate(grid):
        cpath = ck / f"scalecfg_{model}_{ci}.json"
        if cpath.exists():
            val_aps.append(json.loads(cpath.read_text())["val_ap"])
            continue
        _check_budget()
        bcfg = dict(cfgs["baselines"])
        bcfg[model] = {k: [v] for k, v in params.items()}
        est = build_baseline(model, bcfg, seed=seed, tune=False)
        est.fit(data.X[tr], data.y[tr])
        ap = float(average_precision_score(
            data.y[va], baseline_scores(est, data.X[va])))
        cpath.write_text(json.dumps({"val_ap": ap, "params": params},
                                    default=str))
        val_aps.append(ap)
    best_ci = int(np.argmax(val_aps))
    spath = ck / f"scalescores_{model}.npz"
    if spath.exists():
        z = np.load(spath)
        return z["scores"], z["preds"]
    _check_budget()
    bcfg = dict(cfgs["baselines"])
    bcfg[model] = {k: [v] for k, v in grid[best_ci].items()}
    est = build_baseline(model, bcfg, seed=seed, tune=False)
    est.fit(data.X[tr], data.y[tr])
    scores = baseline_scores(est, data.X[te])
    preds = est.predict(data.X[te])
    _savez_atomic(spath, scores=np.asarray(scores, dtype=float),
                  preds=np.asarray(preds, dtype=int))
    return scores, preds


def _scale_pass(cfg, features_cfg, cfgs, ck: Path, n_boot: int, seed: int):
    scfg = cfg.get("scale", {})
    if not scfg.get("enabled", False):
        return [], {}
    caps = scfg.get("max_windows",
                    {"Train": 20000, "Validation": 1500, "Test": 8000})
    _stage_assemble(ck, cfg, features_cfg, seed, caps, "assembled_scale.npz")
    data, idx, n_total = _load_assembled(ck, "assembled_scale.npz")
    rpath = ck / "reduced_scale.npz"
    if not rpath.exists():
        _check_budget()
        reducer = BlockReducer(features_cfg["reduction"]).fit(
            data.X[idx["tr"]], data.blocks)
        _savez_atomic(rpath, Xr=reducer.transform(data.X))
    Xr = np.load(rpath)["Xr"]

    rows = []
    for model in scfg.get("models", ["qsvm_crosslayer", "qsvm_zz",
                                     "random_forest", "xgboost"]):
        respath = ck / f"res2_binary_scale_{model}.json"
        if respath.exists():
            rows.append(json.loads(respath.read_text())["row"])
            continue
        if model in ("qsvm_crosslayer", "qsvm_zz"):
            scores, preds = _scale_quantum_scores(model, data, Xr, idx, cfgs,
                                                  ck, scfg, seed)
        else:
            scores, preds = _scale_classical_scores(model, data, idx, cfgs,
                                                    ck, seed)
        te = idx["te"]
        rep = detection_report(data.y[te], scores, preds)
        ci = grouped_bootstrap_ci(data.y[te], scores, data.groups[te],
                                  n_boot=n_boot, seed=seed)
        row = {"variant": "scale", "model": model, "seed": int(seed),
               **{k: rep.get(k) for k in ["pr_auc", "roc_auc", "f1", "mcc",
                                          "precision", "recall", "fpr_at_tpr"]},
               "pr_auc_lo": ci["pr_auc"]["lo"], "pr_auc_hi": ci["pr_auc"]["hi"],
               "roc_auc_lo": ci["roc_auc"]["lo"],
               "roc_auc_hi": ci["roc_auc"]["hi"],
               "n_test": rep["n"], "n_pos": rep["n_pos"],
               "n_train": int(len(idx["tr"]))}
        respath.write_text(json.dumps({"row": row}, default=float))
        rows.append(row)
    return rows, n_total


# --------------------------------------------------------------------------- #
# Significance with multiple-comparison correction
# --------------------------------------------------------------------------- #
def _significance_rows(pred_rows, n_boot, seed, ck: Path):
    """Paired grouped-bootstrap qsvm_crosslayer-vs-classical tests on the SAME
    test windows (primary seed, full variant), Holm-corrected per metric.
    Each paired test is checkpointed (the grouped bootstrap is the slow part)."""
    from ..models.metrics import holm_bonferroni, paired_bootstrap_auc_diff
    df = pd.DataFrame(pred_rows)
    df = df[(df.variant == "full") & (df.seed == seed)]
    base = df[df.model == "qsvm_crosslayer"].sort_values("i")
    if not len(base):
        return []
    rows, raw = [], {"pr_auc": {}, "roc_auc": {}}
    comps = [m for m in df.model.unique()
             if m not in ("qsvm_crosslayer", "qsvm_zz")]
    for m in comps:
        other = df[df.model == m].sort_values("i")
        if len(other) != len(base):
            continue
        for metric in ("pr_auc", "roc_auc"):
            cpath = ck / f"sig_{m}_{metric}.json"
            if cpath.exists():
                t = json.loads(cpath.read_text())
            else:
                _check_budget()
                t = paired_bootstrap_auc_diff(
                    base.y_true.to_numpy(), base.score.to_numpy(),
                    other.score.to_numpy(), base.group.to_numpy(),
                    n_boot=n_boot, seed=seed, metric=metric)
                cpath.write_text(json.dumps(t, default=float))
            raw[metric][m] = t["p_two_sided"]
            rows.append({"comparison": f"qsvm_crosslayer_vs_{m}",
                         "metric": metric, "delta": t["delta"],
                         "delta_lo": t["lo"], "delta_hi": t["hi"],
                         "p_raw": t["p_two_sided"]})
    for metric in ("pr_auc", "roc_auc"):
        adj = holm_bonferroni(raw[metric]) if raw[metric] else {}
        for r in rows:
            if r["metric"] == metric:
                m = r["comparison"].replace("qsvm_crosslayer_vs_", "")
                r["p_holm"] = adj.get(m)
    return rows


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(config_path: str) -> dict:
    started = datetime.now().isoformat(timespec="seconds")
    cfg = load_yaml(config_path)
    out_dir = Path(cfg["out_dir"])
    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    root = Path(cfg["root"])
    n_boot = int(cfg.get("bootstrap", 200))

    features_cfg = load_yaml(cfg["features_config"])
    if features_cfg.get("network_mode") != "timing":
        raise ValueError("veremi_nextgen requires features config with "
                         "network_mode: timing (NextGen logs no RSSI).")
    cfgs = {"features": features_cfg,
            "quantum": load_yaml(cfg["quantum_config"]),
            "baselines": load_yaml(cfg["baselines_config"])}
    nominal = float(cfg.get("nominal_rate_hz", 1.0))
    seed = int(features_cfg["splits"].get("seed", 7))

    seeds = [int(s) for s in cfg.get("seeds", [seed])]
    primary = seeds[0]
    caps = cfg.get("max_windows", {}) or {}

    try:
        # ---- data stages ----------------------------------------------------
        for sc in cfg["scenarios"]:
            for sub in _subsets(cfg):
                _check_budget()
                extracted = _stage_extract(root, sc, sub)
                for split in NEXTGEN_SPLITS:
                    _stage_parse(ck, extracted, sc, sub, split)
        for sc in cfg["scenarios"]:
            for sub in _subsets(cfg):
                _check_budget()
                delay_model = _stage_delaycal(ck, sc, sub)
                for split in NEXTGEN_SPLITS:
                    _stage_windows(ck, features_cfg, sc, sub, split,
                                   delay_model, nominal)

        bin_rows, pred_rows = [], []
        mc_rows, mc_confusions, sybil_rows = [], {}, []
        label_names, n_total = [], {}

        # ---- per-seed pass (multi-seed replication, earlier) --------------------
        for sd in seeds:
            sfx = "" if sd == primary else f"_s{sd}"
            fcfg = {**features_cfg,
                    "splits": {**features_cfg["splits"], "seed": sd}}
            scfgs = {**cfgs, "features": fcfg}
            _stage_assemble(ck, cfg, fcfg, sd, caps, f"assembled{sfx}.npz")
            data, idx, n_tot = _load_assembled(ck, f"assembled{sfx}.npz")
            if sd == primary:
                n_total = n_tot
            _stage_reduce(ck, fcfg, data, idx, fname=f"reduced{sfx}.npz")
            z = np.load(ck / f"reduced{sfx}.npz")
            Xr, Xr_abl = z["Xr"], z["Xr_abl"]

            # multi-class labels (attack TYPE = subset name; 'none' = benign)
            le = LabelEncoder().fit(data.attack_name.astype(str))
            data._y_multi = le.transform(data.attack_name.astype(str))
            label_ids = list(range(len(le.classes_)))
            label_names = [str(c) for c in le.classes_]

            # binary: full features (+ network-blind ablation, primary only)
            variants = [("full", Xr)]
            if sd == primary and cfg.get("network_ablation", True):
                abl_models = cfg.get("ablation_models",
                                     ["qsvm_crosslayer", "rbf_svm",
                                      "random_forest"])
                variants.append(("netblind", Xr_abl))
            for variant, Xv in variants:
                if variant == "full":
                    models = (cfg["models"] if sd == primary else
                              cfg.get("multiseed_models",
                                      ["qsvm_crosslayer", "qsvm_zz",
                                       "rbf_svm", "random_forest"]))
                else:
                    models = abl_models
                for model in models:
                    out = _eval_binary_model(model, variant, data, Xv, idx,
                                             scfgs, ck, n_boot, sd,
                                             suffix=sfx)
                    bin_rows.append(out["row"])
                    pred_rows.extend(out["pred_rows"])

            if sd != primary:
                continue

            # ---- LSTM sequence baseline (primary seed; earlier / SOTA) ---------
            if cfg.get("lstm", {}).get("enabled", True):
                for sc in cfg["scenarios"]:
                    for sub in _subsets(cfg):
                        delay_model = _stage_delaycal(ck, sc, sub)
                        for split in NEXTGEN_SPLITS:
                            _stage_windows(ck, fcfg, sc, sub, split,
                                           delay_model, nominal, tag="winseq",
                                           store_sequences=True)
                _stage_assemble(ck, cfg, fcfg, sd, caps, "assembled_seq.npz",
                                tag="winseq", with_seq=True)
                zs = np.load(ck / "assembled_seq.npz", allow_pickle=True)
                assert np.array_equal(zs["y"], data.y), \
                    "sequence assembly diverged from the feature assembly"
                out = _eval_lstm(data, zs["sequences"], zs["seq_lengths"],
                                 idx, scfgs, ck, n_boot, sd)
                bin_rows.append(out["row"])
                pred_rows.extend(out["pred_rows"])

            # ---- multi-class attack-type identification ---------------------
            for model in cfg.get("multiclass_models",
                                 ["qsvm_crosslayer", "random_forest",
                                  "knn_bagging", "mlp", "rbf_svm"]):
                out = _eval_multiclass_model(model, data, Xr, idx, scfgs, ck,
                                             label_ids, label_names)
                mc_rows.append(out["row"])
                mc_confusions[model] = out["confusion"]

            # ---- receiver-scope Sybil pass (one-shot ghost pseudonyms) ------
            if cfg.get("sybil_receiver", {}).get("enabled", True):
                for out in _sybil_receiver_pass(ck, cfg, fcfg, scfgs,
                                                nominal, n_boot, sd):
                    sybil_rows.append(out["row"])
                    bin_rows.append(out["row"])
                    pred_rows.extend(out["pred_rows"])

        # ---- Nyström-at-scale pass (removes the matched-N asterisk) ----------
        scale_rows, scale_totals = _scale_pass(cfg, features_cfg, cfgs, ck,
                                               n_boot, primary)
        bin_rows.extend(scale_rows)

    except BudgetExceeded as exc:
        done = sorted(p.name for p in ck.iterdir())
        return {"status": "in_progress", "reason": str(exc),
                "out_dir": str(out_dir), "n_checkpoints": len(done)}

    # ---- report --------------------------------------------------------------
    results = pd.DataFrame(bin_rows)
    results.to_csv(out_dir / "veremi_nextgen_binary_results.csv", index=False)
    if pred_rows:
        pd.DataFrame(pred_rows).to_parquet(out_dir / "predictions.parquet")
    mc_results = pd.DataFrame(mc_rows)
    mc_results.to_csv(out_dir / "veremi_nextgen_multiclass_results.csv", index=False)
    per_attack = _per_attack_rows([r for r in pred_rows
                                   if r.get("seed", primary) == primary])
    pd.DataFrame(per_attack).to_csv(out_dir / "veremi_nextgen_per_attack.csv",
                                    index=False)

    # multi-seed aggregation: mean ± std over seeds (full variant)
    ms = results[(results["variant"] == "full")
                 & results["model"].isin(results[results["seed"] != primary]
                                         ["model"].unique())]
    agg_rows = []
    if len(seeds) > 1 and len(ms):
        for model, g in ms.groupby("model"):
            agg_rows.append({
                "model": model, "n_seeds": int(g["seed"].nunique()),
                **{f"{m}_mean": float(g[m].mean()) for m in
                   ["pr_auc", "roc_auc", "f1", "mcc"]},
                **{f"{m}_std": float(g[m].std(ddof=1)) for m in
                   ["pr_auc", "roc_auc", "f1", "mcc"]}})
        pd.DataFrame(agg_rows).to_csv(
            out_dir / "veremi_nextgen_multiseed.csv", index=False)

    # paired significance, Holm-corrected per metric (primary seed)
    try:
        sig_rows = _significance_rows(pred_rows, n_boot, primary, ck)
    except BudgetExceeded as exc:
        return {"status": "in_progress", "reason": str(exc),
                "out_dir": str(out_dir),
                "n_checkpoints": len(list(ck.iterdir()))}
    if sig_rows:
        pd.DataFrame(sig_rows).to_csv(
            out_dir / "veremi_nextgen_significance.csv", index=False)

    delay_fits = {p.stem: json.loads(p.read_text()).get("_fit", {})
                  for p in ck.glob("delay_*.json")}
    elapsed = round(time.time() - _T0, 1)
    summary = {"status": "done", "out_dir": str(out_dir),
               "scenarios": cfg["scenarios"], "attacks": cfg["attacks"],
               "splits": "official NextGen Train/Validation/Test",
               "network_mode": "timing (no RSSI in NextGen)",
               "n_windows_total": n_total,
               "n_used": {k: int(len(v)) for k, v in idx.items()},
               "delay_models": delay_fits, "n_bootstrap": n_boot,
               "wall_time_s": elapsed,
               "binary": bin_rows, "multiclass": mc_rows,
               "multiclass_labels": label_names,
               "multiclass_confusion": mc_confusions,
               "sybil_receiver": sybil_rows,
               "per_attack": per_attack,
               "seeds": seeds, "multiseed": agg_rows,
               "significance_holm": sig_rows,
               "scale": scale_rows,
               "scale_n_total": scale_totals}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=float))
    capture_provenance(out_dir, timestamp=started, config_path=config_path,
                       extra={"experiment": "veremi_nextgen",
                              "wall_time_s": elapsed, "n_bootstrap": n_boot})
    return summary
