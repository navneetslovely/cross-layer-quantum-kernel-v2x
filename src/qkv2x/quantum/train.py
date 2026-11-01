"""Train the feature-map parameters (w, u, gamma) by kernel-target alignment.

The SVM that consumes the kernel is convex, so we tune only the *feature map*
here, using centered KTA as a cheap, retraining-free model-selection score. Class-balanced mini-batches stand in for the weighted-KTA objective
under heavy class imbalance. Optimisation is plain Adam over the differentiable
statevector kernel.
"""
from __future__ import annotations

import numpy as np
import torch

from .featuremap import CrossLayerFeatureMap


def _torch_fidelity_gram(states: torch.Tensor) -> torch.Tensor:
    overlap = states @ states.conj().transpose(0, 1)
    return overlap.real ** 2 + overlap.imag ** 2


def _centered_kta_with_target_torch(K: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """Centered kernel-target alignment for an arbitrary target Gram T."""
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    Kc = H @ K @ H
    num = (Kc * T).sum()
    den = torch.linalg.norm(Kc) * torch.linalg.norm(T) + 1e-12
    return num / den


def _centered_kta_torch(K: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Binary: target Gram is outer(y, y) with y in {-1, +1}.
    return _centered_kta_with_target_torch(K, torch.outer(y, y))


def _multiclass_target_torch(y_codes: torch.Tensor) -> torch.Tensor:
    """Multi-class ideal target: T_ij = +1 if same class else -1. Reduces exactly to
    outer(y, y) in the binary {-1,+1} case, so the same centered-KTA objective trains
    the map to make same-class states similar and different-class states dissimilar."""
    same = (y_codes[:, None] == y_codes[None, :]).to(y_codes.dtype)
    return 2.0 * same - 1.0


def _balanced_batch(y: np.ndarray, batch: int, rng: np.random.Generator) -> np.ndarray:
    pos = np.where(y > 0)[0]
    neg = np.where(y <= 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return rng.choice(len(y), size=min(batch, len(y)), replace=False)
    half = max(1, batch // 2)
    pi = rng.choice(pos, size=half, replace=len(pos) < half)
    ni = rng.choice(neg, size=half, replace=len(neg) < half)
    idx = np.concatenate([pi, ni])
    rng.shuffle(idx)
    return idx


def _stratified_batch(y: np.ndarray, batch: int, rng: np.random.Generator) -> np.ndarray:
    """Class-stratified mini-batch: ~batch/K indices per class (the multi-class analogue
    of the balanced binary batch), so every class is represented in the KTA estimate."""
    classes = np.unique(y)
    if len(classes) < 2:
        return rng.choice(len(y), size=min(batch, len(y)), replace=False)
    per = max(1, batch // len(classes))
    picks = [rng.choice(np.where(y == c)[0], size=per, replace=(np.sum(y == c) < per))
             for c in classes]
    idx = np.concatenate(picks)
    rng.shuffle(idx)
    return idx


def train_feature_map(
    fm: CrossLayerFeatureMap,
    X: np.ndarray,
    y: np.ndarray,
    aoi: np.ndarray | None,
    cfg: dict,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Return (best_params detached, history)."""
    rng = np.random.default_rng(int(cfg.get("seed", 11)))
    params = fm.init_params(
        init_w=cfg.get("init_w", 1.0),
        init_u=cfg.get("init_u", 1.0),
        init_gamma=cfg.get("init_gamma", 0.5),
        requires_grad=True,
    )
    opt = torch.optim.Adam([params["w"], params["u"], params["gamma"]],
                           lr=float(cfg.get("lr", 0.05)))

    ypm = np.where(y > 0, 1.0, -1.0).astype(np.float64)
    epochs = int(cfg.get("epochs", 120))
    batch = int(cfg.get("batch", 512))

    history = {"alignment": []}
    best_align = -np.inf
    best_params = {k: v.detach().clone() for k, v in params.items()}

    for ep in range(epochs):
        idx = _balanced_batch(y, batch, rng)
        Xb = X[idx]
        ab = aoi[idx] if aoi is not None else None
        yb = torch.as_tensor(ypm[idx], dtype=fm.rdtype, device=fm.torch_device)

        opt.zero_grad()
        states = fm.statevectors(Xb, ab, params, chunk=int(cfg.get("chunk", 256)))
        K = _torch_fidelity_gram(states)
        align = _centered_kta_torch(K, yb)
        loss = -align
        loss.backward()
        opt.step()

        a = float(align.detach().cpu())
        history["alignment"].append(a)
        if a > best_align:
            best_align = a
            best_params = {k: v.detach().clone() for k, v in params.items()}

    for v in best_params.values():
        v.requires_grad_(False)
    history["best_alignment"] = best_align
    return best_params, history


def train_feature_map_multiclass(
    fm: CrossLayerFeatureMap,
    X: np.ndarray,
    y: np.ndarray,
    aoi: np.ndarray | None,
    cfg: dict,
) -> tuple[dict[str, torch.Tensor], dict]:
    """Train (w, u, gamma) by MULTI-CLASS centered KTA.

    `y` is an integer class-code array (e.g. attack types). The target is the
    same-class/different-class ideal Gram (_multiclass_target_torch), so the map is
    aligned to separate all attack types at once — the multi-class generalisation of
    the binary KTA training. The SVM stays convex. Returns (best_params, history)."""
    rng = np.random.default_rng(int(cfg.get("seed", 11)))
    params = fm.init_params(
        init_w=cfg.get("init_w", 1.0),
        init_u=cfg.get("init_u", 1.0),
        init_gamma=cfg.get("init_gamma", 0.5),
        requires_grad=True,
    )
    opt = torch.optim.Adam([params["w"], params["u"], params["gamma"]],
                           lr=float(cfg.get("lr", 0.05)))

    y = np.asarray(y)
    epochs = int(cfg.get("epochs", 120))
    batch = int(cfg.get("batch", 512))

    history = {"alignment": []}
    best_align = -np.inf
    best_params = {k: v.detach().clone() for k, v in params.items()}

    for ep in range(epochs):
        idx = _stratified_batch(y, batch, rng)
        Xb = X[idx]
        ab = aoi[idx] if aoi is not None else None
        yb = torch.as_tensor(y[idx].astype(np.float64), dtype=fm.rdtype,
                             device=fm.torch_device)

        opt.zero_grad()
        states = fm.statevectors(Xb, ab, params, chunk=int(cfg.get("chunk", 256)))
        K = _torch_fidelity_gram(states)
        T = _multiclass_target_torch(yb)
        align = _centered_kta_with_target_torch(K, T)
        loss = -align
        loss.backward()
        opt.step()

        a = float(align.detach().cpu())
        history["alignment"].append(a)
        if a > best_align:
            best_align = a
            best_params = {k: v.detach().clone() for k, v in params.items()}

    for v in best_params.values():
        v.requires_grad_(False)
    history["best_alignment"] = best_align
    return best_params, history


def train_feature_map_validated(
    fm: CrossLayerFeatureMap,
    X: np.ndarray,
    y: np.ndarray,
    aoi: np.ndarray | None,
    cfg: dict,
    X_val: np.ndarray,
    y_val: np.ndarray,
    aoi_val: np.ndarray | None,
    val_every: int = 1,
) -> tuple[dict[str, dict[str, torch.Tensor]], dict]:
    """Opt-in validated variant of `train_feature_map`.

    Identical optimisation to `train_feature_map` (Adam on -centered-KTA over
    class-balanced training batches), plus:
      * optional perturbation of the constant initial parameters
        (`cfg["init_jitter"]`, relative scale; `cfg["jitter_seed"]`), so that
        restarts differ in both the batch stream and the starting point;
      * alignment evaluated every `val_every` epochs on a FIXED held-out
        validation subset (`X_val`, `y_val`), under no_grad;
      * three parameter snapshots returned side by side, keyed by the
        selection rule: "best_batch" (highest single training-batch alignment,
        the rule used by `train_feature_map`), "best_val" (highest validation
        alignment), "final" (last epoch).

    `train_feature_map` itself is unchanged, so every published number still
    reproduces; this function is used only by the kta_restarts restart study.
    """
    rng = np.random.default_rng(int(cfg.get("seed", 11)))
    params = fm.init_params(
        init_w=cfg.get("init_w", 1.0),
        init_u=cfg.get("init_u", 1.0),
        init_gamma=cfg.get("init_gamma", 0.5),
        requires_grad=True,
    )
    jitter = float(cfg.get("init_jitter", 0.0))
    if jitter > 0.0:
        gen = torch.Generator(device="cpu").manual_seed(int(cfg.get("jitter_seed", 0)))
        with torch.no_grad():
            for v in params.values():
                noise = torch.randn(v.shape, generator=gen, dtype=v.dtype)
                v.add_((noise * jitter * v.abs().clamp_min(1e-3)).to(v.device))
    opt = torch.optim.Adam([params["w"], params["u"], params["gamma"]],
                           lr=float(cfg.get("lr", 0.05)))

    ypm = np.where(y > 0, 1.0, -1.0).astype(np.float64)
    yv = torch.as_tensor(np.where(np.asarray(y_val) > 0, 1.0, -1.0),
                         dtype=fm.rdtype, device=fm.torch_device)
    epochs = int(cfg.get("epochs", 120))
    batch = int(cfg.get("batch", 512))
    chunk = int(cfg.get("chunk", 256))

    def _snap():
        return {k: v.detach().clone() for k, v in params.items()}

    history = {"alignment": [], "val_alignment": [], "val_epochs": []}
    best = {"best_batch": (-np.inf, -1, _snap()), "best_val": (-np.inf, -1, _snap())}
    for ep in range(epochs):
        idx = _balanced_batch(y, batch, rng)
        Xb = X[idx]
        ab = aoi[idx] if aoi is not None else None
        yb = torch.as_tensor(ypm[idx], dtype=fm.rdtype, device=fm.torch_device)
        opt.zero_grad()
        states = fm.statevectors(Xb, ab, params, chunk=chunk)
        align = _centered_kta_torch(_torch_fidelity_gram(states), yb)
        (-align).backward()
        opt.step()
        a = float(align.detach().cpu())
        history["alignment"].append(a)
        if a > best["best_batch"][0]:
            best["best_batch"] = (a, ep, _snap())
        if (ep + 1) % val_every == 0 or ep + 1 == epochs:
            with torch.no_grad():
                sv = fm.statevectors(X_val, aoi_val, params, chunk=chunk)
                av = float(_centered_kta_torch(_torch_fidelity_gram(sv), yv).cpu())
            history["val_alignment"].append(av)
            history["val_epochs"].append(ep)
            if av > best["best_val"][0]:
                best["best_val"] = (av, ep, _snap())

    out = {"best_batch": best["best_batch"][2], "best_val": best["best_val"][2],
           "final": _snap()}
    for d in out.values():
        for v in d.values():
            v.requires_grad_(False)
    history["best_alignment"] = best["best_batch"][0]
    history["best_epoch"] = best["best_batch"][1]
    history["best_val_alignment"] = best["best_val"][0]
    history["best_val_epoch"] = best["best_val"][1]
    history["final_alignment"] = history["alignment"][-1]
    history["final_val_alignment"] = history["val_alignment"][-1]
    return out, history
