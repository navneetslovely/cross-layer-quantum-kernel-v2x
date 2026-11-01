"""Shared experiment plumbing: dataset assembly across runs, group-aware splits,
the model zoo, and the quantum-vs-classical diagnostics.

A "model" here is identified by name and exposes a uniform fit/score over a
train/test split, so every experiment driver (robustness, low-label, cross-attack,
VeReMi) reuses exactly the same training and scoring code path.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.model_selection import GroupShuffleSplit

from ..config import load_yaml
from ..cosim.loader import iter_runs, load_meta, load_rx
from ..features.dataset import build_dataset
from ..features.reduce import BlockReducer
from ..models.baselines import baseline_scores, build_baseline, build_multiclass
from ..models.qsvm import (
    QuantumKernelMultiSVM,
    QuantumKernelOneClassSVM,
    QuantumKernelSVM,
)
from ..models.rules import RulePlausibilityDetector
from ..quantum.diagnostics import (
    centered_kta,
    effective_dimension,
    geometric_difference,
    kernel_offdiag_variance,
)
from ..quantum.featuremap import CrossLayerFeatureMap
from ..quantum.kernels import (
    add_depolarizing_noise,
    add_shot_noise,
    bloch_features,
    fidelity_gram,
    fidelity_kernel_blocks,
    nystrom_fidelity_blocks,
    nystrom_projected_blocks,
    projected_kernel_blocks,
    psd_clip,
)
from ..quantum.train import train_feature_map, train_feature_map_multiclass

QUANTUM_MODELS = {"qsvm_crosslayer", "qsvm_zz", "qsvm_projected", "ocsvm_quantum",
                  # topology ablations — opt-in, never in default configs:
                  # qsvm_none = no entanglement; qsvm_sparse = seeded random edges,
                  # edge-count-matched to the bipartite cross-layer map. Both are
                  # KTA-trained exactly like qsvm_crosslayer (only qsvm_zz stays fixed),
                  # so differences isolate coupling TOPOLOGY at matched training budget.
                  "qsvm_none", "qsvm_sparse"}
CLASSICAL_MODELS = {"rbf_svm", "random_forest", "knn_bagging", "mlp", "xgboost"}
# Quantum models usable for MULTI-CLASS attack-type identification (fidelity/projected
# Gram + multi-class SVC). The one-class detector is anomaly-only, so it is excluded.
MULTICLASS_QUANTUM = {"qsvm_crosslayer", "qsvm_zz", "qsvm_projected"}


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
@dataclass
class ExperimentData:
    X: np.ndarray
    y: np.ndarray
    attack_name: np.ndarray
    groups: np.ndarray
    aoi: np.ndarray
    t_end: np.ndarray
    feature_names: list[str]
    blocks: list[str]
    points: pd.DataFrame          # per-window sweep-point params (e.g. density_scale)
    meta: pd.DataFrame = field(default_factory=pd.DataFrame)

    def subset(self, mask: np.ndarray) -> "ExperimentData":
        return ExperimentData(
            X=self.X[mask], y=self.y[mask], attack_name=self.attack_name[mask],
            groups=self.groups[mask], aoi=self.aoi[mask], t_end=self.t_end[mask],
            feature_names=self.feature_names, blocks=self.blocks,
            points=self.points.iloc[mask].reset_index(drop=True),
            meta=self.meta.iloc[mask].reset_index(drop=True) if len(self.meta) else self.meta,
        )

    def __len__(self):
        return len(self.y)


def _manifest_point_map(runs_root: Path) -> dict[str, dict]:
    mf = runs_root / "manifest.json"
    mapping: dict[str, dict] = {}
    if mf.exists():
        data = json.loads(mf.read_text())
        for rec in data.get("points", []):
            mapping[str(Path(rec["out_dir"]).resolve())] = rec.get("point", {})
    return mapping


def load_experiment_data(runs_root, features_cfg: dict,
                         drop_frac: float = 0.0, drop_seed: int = 0) -> ExperimentData:
    """Build the windowed dataset from all runs under `runs_root`.

    `drop_frac` injects packet loss: a fraction of receptions is dropped at
    random before windowing, per run (seed = drop_seed + run index for variety). Default
    0.0 → identical behaviour to before, so all existing callers are unaffected.
    """
    runs_root = Path(runs_root)
    point_map = _manifest_point_map(runs_root)
    propagation = features_cfg["propagation"]

    Xs, ys, atks, grps, aois, tends, pts, metas = [], [], [], [], [], [], [], []
    feat_names = blocks = None
    group_offset = 0
    for run_idx, run_dir in enumerate(iter_runs(runs_root)):
        rx = load_rx(run_dir)
        meta = load_meta(run_dir)
        nominal = float(meta.get("bsm_rate_hz", 10.0))
        try:
            ds = build_dataset(rx, features_cfg, propagation, nominal,
                               drop_frac=drop_frac, drop_seed=int(drop_seed) + run_idx)
        except ValueError:
            if drop_frac and float(drop_frac) > 0.0:
                continue   # heavy loss left this run with no windows — skip it
            raise
        feat_names = ds.feature_names
        blocks = ds.blocks
        Xs.append(ds.X)
        ys.append(ds.y)
        atks.append(ds.attack_name)
        # keep groups disjoint across runs (different seeds/points are different worlds)
        grps.append(ds.groups + group_offset)
        group_offset += int(ds.groups.max()) + 1 if len(ds.groups) else 0
        aois.append(ds.aoi)
        tends.append(ds.meta["t_end"].to_numpy())
        point = point_map.get(str(Path(run_dir).resolve()), {})
        prow = {**point, "run": run_dir.name, "seed": meta.get("seed", 0)}
        pts.append(pd.DataFrame([prow] * len(ds)))
        metas.append(ds.meta)

    if not Xs:
        raise ValueError(f"load_experiment_data: no windows produced (drop_frac={drop_frac}); "
                         "lower the packet-loss level.")
    return ExperimentData(
        X=np.vstack(Xs), y=np.concatenate(ys),
        attack_name=np.concatenate(atks), groups=np.concatenate(grps),
        aoi=np.concatenate(aois), t_end=np.concatenate(tends),
        feature_names=feat_names, blocks=blocks,
        points=pd.concat(pts, ignore_index=True),
        meta=pd.concat(metas, ignore_index=True),
    )


# --------------------------------------------------------------------------- #
# Splits + reduction
# --------------------------------------------------------------------------- #
def group_split(groups, y, test_frac, val_frac, seed, tries=20):
    """Group-aware train/val/test split with both classes present in each part."""
    n = len(y)
    for t in range(tries):
        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed + t)
        tr_va, te = next(gss.split(np.zeros(n), y, groups))
        if val_frac > 0:
            rel = val_frac / (1.0 - test_frac)
            gss2 = GroupShuffleSplit(n_splits=1, test_size=rel, random_state=seed + t)
            tr, va = next(gss2.split(np.zeros(len(tr_va)), y[tr_va], groups[tr_va]))
            tr, va = tr_va[tr], tr_va[va]
        else:
            tr, va = tr_va, np.array([], dtype=int)
        ok = len(np.unique(y[tr])) == 2 and len(np.unique(y[te])) == 2
        if val_frac > 0:
            ok = ok and len(np.unique(y[va])) == 2
        if ok:
            return tr, va, te
    raise RuntimeError("group_split: could not find a split with both classes present.")


def fit_reducer(reduction_cfg, X, blocks, idx_tr) -> BlockReducer:
    return BlockReducer(reduction_cfg).fit(X[idx_tr], blocks)


# --------------------------------------------------------------------------- #
# Feature map construction
# --------------------------------------------------------------------------- #
def _resolve_device(spec: str) -> str:
    # 'auto' -> cuda if available else cpu. MPS is intentionally NOT auto-selected:
    # the statevector kernels use complex64 tensors, and Torch's complex support on
    # Apple's MPS backend is incomplete. On a Mac, set device: mps explicitly only if
    # your Torch build supports complex MPS ops; otherwise cpu (the CPU is fast for
    # the n<=12 exact statevector sims here).
    if spec and spec != "auto":
        return spec
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def make_feature_map(model_name, quantum_cfg, reduction_cfg, features_cfg) -> CrossLayerFeatureMap:
    fmcfg = quantum_cfg["feature_map"]
    _name_entangler = {"qsvm_zz": "zz_full", "qsvm_none": "none",
                       "qsvm_sparse": "sparse_random"}
    entangler = _name_entangler.get(model_name, fmcfg.get("entangler", "cross_layer"))
    return CrossLayerFeatureMap(
        n_mobility=int(reduction_cfg["n_mobility"]),
        n_network=int(reduction_cfg["n_network"]),
        n_layers=int(fmcfg.get("n_layers", 2)),
        entangler=entangler,
        coupling=fmcfg.get("coupling", "full_bipartite"),
        aoi_modulation=bool(fmcfg.get("aoi_modulation", True)),
        kappa=float(features_cfg.get("aoi", {}).get("kappa", 0.5)),
        device=_resolve_device(quantum_cfg.get("device", "auto")),
        dtype=quantum_cfg.get("dtype", "complex64"),
        sparse_edges=fmcfg.get("sparse_edges", None),
        sparse_seed=int(fmcfg.get("sparse_seed", 7)),
    )


def _maybe_train_params(fm, model_name, quantum_cfg, Xtr, ytr, aoitr):
    train_cfg = quantum_cfg.get("train", {})
    if model_name == "qsvm_zz" or not train_cfg.get("enabled", True):
        return fm.fixed_params(), {}
    return train_feature_map(fm, Xtr, ytr, aoitr, train_cfg)


def _maybe_train_params_multiclass(fm, model_name, quantum_cfg, Xtr, ytr, aoitr):
    """Multi-class analogue: train (w,u,gamma) by multi-class KTA unless this is the
    untrained ZZ ablation or training is disabled."""
    train_cfg = quantum_cfg.get("train", {})
    if model_name == "qsvm_zz" or not train_cfg.get("enabled", True):
        return fm.fixed_params(), {}
    return train_feature_map_multiclass(fm, Xtr, ytr, aoitr, train_cfg)


# --------------------------------------------------------------------------- #
# Model zoo — uniform fit/score
# --------------------------------------------------------------------------- #
def _cap_train_groupwise(tr, groups, max_train, seed):
    """Group-aware cap on the #train windows feeding the O(N^2) quantum Gram.

    The quantum kernel builds an N_train x N_train fidelity matrix, so an uncapped
    train split (tens of thousands of windows on the clean dataset) needs tens of GB
    and OOM-kills the process. We keep WHOLE physical-transmitter groups (so the cap
    stays leakage-safe) until ~max_train rows are reached. Returns tr
    unchanged when it is already small enough (lowlabel/veremi already cap upstream)."""
    if not max_train or len(tr) <= int(max_train):
        return tr
    rng = np.random.default_rng(int(seed))
    g = groups[tr]
    vals, counts = np.unique(g, return_counts=True)
    size_map = dict(zip(vals.tolist(), counts.tolist()))
    chosen, total = [], 0
    for u in rng.permutation(vals):
        chosen.append(u)
        total += size_map[u]
        if total >= int(max_train):
            break
    return tr[np.isin(g, chosen)]


def matched_train_idx(tr, data: ExperimentData, cfgs):
    """Cap a train-index group-wise to the quantum `max_train` so quantum AND classical
    models can be compared at MATCHED N. The quantum kernel cannot use the full split
    (O(N^2) Gram), so a fair significance test trains the classical comparator on the same
    rows. Experiments report classical@full-data separately as the scaling-ceiling context.
    Returns tr unchanged when already small enough."""
    mt = cfgs["quantum"].get("max_train", 3000)
    seed = cfgs["features"]["splits"].get("seed", 7)
    return _cap_train_groupwise(tr, data.groups, mt, seed)


def run_model(name, data: ExperimentData, Xr, idx, cfgs) -> dict:
    """Fit `name` on idx['tr'] and score idx['te']. Returns scores/preds/labels."""
    quantum_cfg = cfgs["quantum"]
    baselines_cfg = cfgs["baselines"]
    reduction_cfg = cfgs["features"]["reduction"]
    features_cfg = cfgs["features"]
    tr, va, te = idx["tr"], idx["va"], idx["te"]
    y = data.y

    if name in QUANTUM_MODELS:
        # Nyström is wired for the supervised quantum SVMs: it builds only
        # the [N,l]+[l,l] quantum overlaps, so a larger train cap is affordable. The
        # one-class detector keeps the full benign Gram (small benign set anyway).
        ny = quantum_cfg.get("nystrom", {}) or {}
        use_ny = bool(ny.get("enabled", False)) and name != "ocsvm_quantum"
        cap = (int(ny.get("max_train", quantum_cfg.get("max_train", 3000))) if use_ny
               else quantum_cfg.get("max_train", 3000))
        tr = _cap_train_groupwise(tr, data.groups, cap,
                                  features_cfg["splits"].get("seed", 7))
        fm = make_feature_map(name, quantum_cfg, reduction_cfg, features_cfg)
        params, hist = _maybe_train_params(fm, name, quantum_cfg,
                                           Xr[tr], y[tr], data.aoi[tr])
        shots = quantum_cfg.get("shots", None)
        projected = name == "qsvm_projected" or quantum_cfg.get("kernel") == "projected"
        gamma = float(quantum_cfg.get("projected_gamma", 1.0))

        if name == "ocsvm_quantum":
            benign = tr[y[tr] == 0]
            s_b = fm.statevectors(Xr[benign], data.aoi[benign], params)
            K_bb = fidelity_gram(s_b)
            s_va = fm.statevectors(Xr[va], data.aoi[va], params) if len(va) else None
            K_vb = fidelity_gram(s_va, s_b) if s_va is not None else None
            oc = QuantumKernelOneClassSVM(nu_grid=quantum_cfg["ocsvm"]["nu_grid"])
            oc.fit(K_bb, K_vb, y[va] if len(va) else None)
            s_te = fm.statevectors(Xr[te], data.aoi[te], params)
            K_tb = fidelity_gram(s_te, s_b)
            scores = oc.decision_function(K_tb)
            preds = oc.predict(K_tb)
        else:
            if use_ny:
                ell = min(int(ny.get("landmarks", 512)), len(tr))
                lm = np.sort(np.random.default_rng(int(ny.get("seed", 13)))
                             .choice(len(tr), size=ell, replace=False))
                if projected:
                    K_tr, K_te = nystrom_projected_blocks(
                        fm, Xr[tr], data.aoi[tr], lm, Xr[te], data.aoi[te],
                        params, gamma=gamma)
                else:
                    K_tr, K_te = nystrom_fidelity_blocks(
                        fm, Xr[tr], data.aoi[tr], lm, Xr[te], data.aoi[te],
                        params, shots=shots)
                if quantum_cfg.get("psd_repair", "clip") == "clip":
                    K_tr = psd_clip(K_tr)
            elif projected:
                K_tr, K_te = projected_kernel_blocks(
                    fm, Xr[tr], data.aoi[tr], Xr[te], data.aoi[te], params, gamma=gamma)
            else:
                K_tr, K_te = fidelity_kernel_blocks(
                    fm, Xr[tr], data.aoi[tr], Xr[te], data.aoi[te], params,
                    shots=shots, psd_repair=quantum_cfg.get("psd_repair", "clip"))
            svm = QuantumKernelSVM(
                C_grid=quantum_cfg["svm"]["C_grid"],
                cv_folds=quantum_cfg["svm"]["cv_folds"],
                class_weight=quantum_cfg["svm"].get("class_weight", "balanced"))
            svm.fit(K_tr, y[tr], groups=data.groups[tr])
            scores = svm.decision_function(K_te)
            preds = svm.predict(K_te)
        return _result(name, data, te, scores, preds)

    if name in CLASSICAL_MODELS:
        npos = int(np.bincount(y[tr].astype(int), minlength=2).min())  # minority-class count
        cv_folds = int(baselines_cfg.get("cv_folds", 5))
        est = build_baseline(name, baselines_cfg, seed=int(baselines_cfg.get("seed", 0)),
                             folds=max(2, min(cv_folds, npos)), tune=(npos >= 2))
        est.fit(Xr[tr], y[tr])
        scores = baseline_scores(est, Xr[te])
        preds = est.predict(Xr[te])
        return _result(name, data, te, scores, preds)

    if name.endswith("_full") and name[:-5] in CLASSICAL_MODELS:
        base = name[:-5]
        npos = int(np.bincount(y[tr].astype(int), minlength=2).min())
        cv_folds = int(baselines_cfg.get("cv_folds", 5))
        est = build_baseline(base, baselines_cfg, seed=int(baselines_cfg.get("seed", 0)),
                             folds=max(2, min(cv_folds, npos)), tune=(npos >= 2))
        est.fit(data.X[tr], y[tr])
        scores = baseline_scores(est, data.X[te])
        preds = est.predict(data.X[te])
        return _result(name, data, te, scores, preds)

    if name == "ocsvm_classical":
        # Classical one-class SVM (RBF) — the honest counterpart to ocsvm_quantum. Learns the
        # benign manifold from benign-only windows and flags anomalies, so it barely needs
        # labelled ATTACKS (the H3 low-label question). nu is picked on the val set by ROC-AUC,
        # mirroring the quantum OC-SVM's nu selection so the two are a fair head-to-head.
        from sklearn.metrics import roc_auc_score
        from sklearn.svm import OneClassSVM
        oc_cfg = baselines_cfg.get("ocsvm", {})
        nu_grid = oc_cfg.get("nu_grid", [0.01, 0.05, 0.10])
        gamma = oc_cfg.get("gamma", "scale")
        Xb = Xr[tr[y[tr] == 0]]                          # benign-only training windows
        nu = float(nu_grid[len(nu_grid) // 2])
        if len(va) and len(np.unique(y[va])) == 2 and len(Xb):
            best_auc = -1.0
            for cand in nu_grid:
                m = OneClassSVM(kernel="rbf", nu=float(cand), gamma=gamma).fit(Xb)
                auc = roc_auc_score(y[va], -m.decision_function(Xr[va]))  # higher = anomaly
                if auc > best_auc:
                    best_auc, nu = auc, float(cand)
        oc = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma).fit(Xb)
        scores = -oc.decision_function(Xr[te])           # anomaly score (higher => attack)
        preds = (oc.predict(Xr[te]) == -1).astype(int)   # -1 = outlier = predicted attack
        return _result(name, data, te, scores, preds)

    if name == "rules":
        det = RulePlausibilityDetector(baselines_cfg.get("rules", {}))
        det.fit(data.X[tr], y[tr], data.feature_names)
        scores = det.decision_function(data.X[te])
        preds = det.predict(data.X[te])
        return _result(name, data, te, scores, preds)

    raise ValueError(f"unknown model {name}")


def _result(name, data, te, scores, preds) -> dict:
    return {
        "model": name,
        "y_true": data.y[te],
        "scores": np.asarray(scores, dtype=float),
        "y_pred": np.asarray(preds, dtype=int),
        "attack_name": data.attack_name[te],
        "groups": data.groups[te],
        "t_end": data.t_end[te],
    }


def predictions_long(res: dict, **ctx) -> list[dict]:
    """Per-window prediction rows from a run_model result, tagged with `ctx` (e.g.
    density, holdout_attack). Persisting these lets you regenerate ROC/PR curves,
    confusion matrices, thresholds and significance tests later WITHOUT re-running."""
    n = len(res["y_true"])
    base = {"model": res["model"], **ctx}
    yt, sc, yp = res["y_true"], res["scores"], res["y_pred"]
    an, gr, te = res["attack_name"], res["groups"], res["t_end"]
    return [{**base, "i": int(i), "y_true": int(yt[i]), "score": float(sc[i]),
             "y_pred": int(yp[i]), "attack_name": str(an[i]), "group": str(gr[i]),
             "t_end": float(te[i])} for i in range(n)]


def run_model_multiclass(name, data: ExperimentData, Xr, idx, cfgs, y) -> np.ndarray:
    """Fit `name` as a MULTI-CLASS classifier on idx['tr'], predict idx['te'].

    `y` is the (full-length, integer-encoded) attack-TYPE label array. Quantum models
    build the same fidelity/projected Gram as the binary path but feed it to a
    multi-class precomputed-kernel SVC (QuantumKernelMultiSVM); classical models use
    build_multiclass. Returns predicted class labels for idx['te']."""
    tr, te = idx["tr"], idx["te"]
    quantum_cfg = cfgs["quantum"]
    if name in MULTICLASS_QUANTUM:
        fm = make_feature_map(name, quantum_cfg, cfgs["features"]["reduction"],
                              cfgs["features"])
        # Train the map by MULTI-CLASS kernel-target alignment (same-class/different-
        # class ideal target) so the entangling mobility-network kernel separates all
        # attack types at once; qsvm_zz / train.enabled=false fall back to fixed params.
        params, _ = _maybe_train_params_multiclass(
            fm, name, quantum_cfg, Xr[tr], y[tr], data.aoi[tr])
        if name == "qsvm_projected" or quantum_cfg.get("kernel") == "projected":
            gamma = float(quantum_cfg.get("projected_gamma", 1.0))
            K_tr, K_te = projected_kernel_blocks(
                fm, Xr[tr], data.aoi[tr], Xr[te], data.aoi[te], params, gamma=gamma)
        else:
            K_tr, K_te = fidelity_kernel_blocks(
                fm, Xr[tr], data.aoi[tr], Xr[te], data.aoi[te], params,
                shots=quantum_cfg.get("shots"),
                psd_repair=quantum_cfg.get("psd_repair", "clip"))
        svm = QuantumKernelMultiSVM(C_grid=quantum_cfg["svm"]["C_grid"],
                                    cv_folds=quantum_cfg["svm"]["cv_folds"])
        svm.fit(K_tr, y[tr], groups=data.groups[tr])
        return svm.predict(K_te)
    # classical multi-class
    est = build_multiclass(name, cfgs["baselines"], seed=int(cfgs["baselines"].get("seed", 0)))
    est.fit(Xr[tr], y[tr])
    return est.predict(Xr[te])


# --------------------------------------------------------------------------- #
# Quantum-vs-classical geometric difference per attack class (H2 explanation)
# --------------------------------------------------------------------------- #
def geometric_difference_per_attack(data: ExperimentData, Xr, idx, cfgs,
                                    max_samples=400, seed=0) -> dict:
    quantum_cfg = cfgs["quantum"]
    reduction_cfg = cfgs["features"]["reduction"]
    features_cfg = cfgs["features"]
    rng = np.random.default_rng(seed)
    tr = idx["tr"]
    fm = make_feature_map("qsvm_crosslayer", quantum_cfg, reduction_cfg, features_cfg)
    params, _ = _maybe_train_params(fm, "qsvm_crosslayer", quantum_cfg,
                                    Xr[tr], data.y[tr], data.aoi[tr])

    out = {}
    benign_idx = tr[data.y[tr] == 0]
    for atk in sorted(set(data.attack_name[tr]) - {"none"}):
        atk_idx = tr[(data.attack_name[tr] == atk) & (data.y[tr] == 1)]
        if len(atk_idx) < 10 or len(benign_idx) < 10:
            continue
        nb = min(max_samples // 2, len(benign_idx))
        na = min(max_samples // 2, len(atk_idx))
        sel = np.concatenate([rng.choice(benign_idx, nb, replace=False),
                              rng.choice(atk_idx, na, replace=False)])
        Xq = Xr[sel]
        aoi = data.aoi[sel]
        states = fm.statevectors(Xq, aoi, params)
        Kq = fidelity_gram(states)
        Kc = rbf_kernel(Xq, gamma=1.0 / Xq.shape[1])
        out[atk] = {
            "g_cq": geometric_difference(Kc, Kq, reg=float(quantum_cfg["diagnostics"].get("gcq_reg", 1e-6))),
            "kernel_variance": kernel_offdiag_variance(Kq),
            "effective_dim": effective_dimension(Kq),
            "n_benign": int(nb), "n_attack": int(na),
        }
    return out


# --------------------------------------------------------------------------- #
# Quantum-kernel sensitivity to hardware noise
# --------------------------------------------------------------------------- #
def noise_sensitivity(data: ExperimentData, Xr, idx, cfgs,
                      max_samples: int = 300, seed: int = 0) -> dict:
    """Detection vs simulated hardware noise for the cross-layer quantum kernel.

    Sweeps two axes from quantum.yaml::diagnostics — finite shot count M
    (`shots_sweep`) and global depolarising strength p (`depolarizing.p_grid`).
    The noiseless train/test fidelity Grams are built ONCE on a subsample, then
    contracted analytically at each noise level (cheap), the cross-layer SVM is
    refit and rescored, and ROC/PR-AUC + kernel off-diagonal variance are reported.
    Depolarising drives the kernel toward the trivial 2^-n value (the 
    exponential-concentration failure mode), which the variance column tracks."""
    from ..models.metrics import detection_report

    quantum_cfg = cfgs["quantum"]
    diag = quantum_cfg.get("diagnostics", {}) or {}
    tr, te = idx["tr"], idx["te"]
    rng = np.random.default_rng(seed)
    tr = _cap_train_groupwise(tr, data.groups, min(int(max_samples), len(tr)), seed)
    if len(te) > int(max_samples):
        te = np.sort(rng.choice(te, int(max_samples), replace=False))
    if len(np.unique(data.y[tr])) < 2 or len(np.unique(data.y[te])) < 2:
        return {"status": "skipped: a split had a single class"}

    fm = make_feature_map("qsvm_crosslayer", quantum_cfg,
                          cfgs["features"]["reduction"], cfgs["features"])
    params, _ = _maybe_train_params(fm, "qsvm_crosslayer", quantum_cfg,
                                    Xr[tr], data.y[tr], data.aoi[tr])
    K_tr0 = fidelity_gram(fm.statevectors(Xr[tr], data.aoi[tr], params))
    K_te0 = fidelity_gram(fm.statevectors(Xr[te], data.aoi[te], params),
                          fm.statevectors(Xr[tr], data.aoi[tr], params))
    ytr, yte, gtr = data.y[tr], data.y[te], data.groups[tr]
    folds = min(3, int(quantum_cfg["svm"].get("cv_folds", 5)))

    def _eval(K_tr, K_te):
        svm = QuantumKernelSVM(C_grid=quantum_cfg["svm"]["C_grid"], cv_folds=folds)
        svm.fit(psd_clip(K_tr), ytr, groups=gtr)
        sc = svm.decision_function(K_te)
        rep = detection_report(yte, sc, (sc > 0).astype(int))
        return {"roc_auc": rep["roc_auc"], "pr_auc": rep["pr_auc"]}

    ideal = {**_eval(K_tr0, K_te0), "kernel_variance": kernel_offdiag_variance(K_tr0)}

    depol = []
    for p in (diag.get("depolarizing", {}) or {}).get("p_grid", []):
        K_tr = add_depolarizing_noise(K_tr0, float(p), fm.n)
        K_te = add_depolarizing_noise(K_te0, float(p), fm.n)
        depol.append({"p": float(p), **_eval(K_tr, K_te),
                      "kernel_variance": kernel_offdiag_variance(K_tr)})

    shots = []
    for M in diag.get("shots_sweep", []):
        K_tr = add_shot_noise(K_tr0, int(M), rng)
        np.fill_diagonal(K_tr, 1.0)
        K_te = add_shot_noise(K_te0, int(M), rng)
        shots.append({"shots": int(M), **_eval(K_tr, K_te),
                      "kernel_variance": kernel_offdiag_variance(K_tr)})

    return {"status": "ok", "n_train": int(len(tr)), "n_test": int(len(te)),
            "ideal": ideal, "depolarizing": depol, "shots": shots}
