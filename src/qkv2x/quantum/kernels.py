"""Quantum kernels from statevectors.

Fidelity kernel:  k_Q(x,x') = |<phi(x')|phi(x)>|^2   -> a single Gram product.
Projected kernel:  k_PQ(x,x') = exp(-gamma * sum_i ||rho_i(x)-rho_i(x')||_F^2),
                   provably resistant to exponential concentration. Since for a
                   1-qubit state ||rho_i - rho_i'||_F^2 = 1/2 ||r_i - r_i'||^2 with
                   r_i the Bloch vector, this is an RBF kernel on stacked Bloch
                   coordinates -> also a single distance computation.

Shot noise (fidelity only): hat k ~ Binomial(M, k)/M, Var = k(1-k)/M.
"""
from __future__ import annotations

import numpy as np
import torch

from .featuremap import CrossLayerFeatureMap


def fidelity_gram(states_a: torch.Tensor, states_b: torch.Tensor | None = None) -> np.ndarray:
    """|<b|a>|^2. If states_b is None, returns the symmetric train Gram."""
    b = states_a if states_b is None else states_b
    overlap = states_a @ b.conj().transpose(0, 1)        # [Na, Nb]
    k = (overlap.real ** 2 + overlap.imag ** 2)
    return k.detach().cpu().numpy().astype(np.float64)


def bloch_features(states: torch.Tensor, n: int) -> torch.Tensor:
    """Per-qubit Bloch vectors (X,Y,Z) for each sample -> [N, 3n]."""
    N = states.shape[0]
    S = states.reshape([N] + [2] * n)
    feats = []
    for i in range(n):
        ax = 1 + i
        T = torch.movedim(S, ax, 1).reshape(N, 2, -1)
        psi0 = T[:, 0, :]
        psi1 = T[:, 1, :]
        p0 = (psi0.real ** 2 + psi0.imag ** 2).sum(dim=1)
        p1 = (psi1.real ** 2 + psi1.imag ** 2).sum(dim=1)
        c = (psi0.conj() * psi1).sum(dim=1)
        z = p0 - p1
        x = 2.0 * c.real
        y = 2.0 * c.imag
        feats.extend([x, y, z])
    return torch.stack(feats, dim=1)  # [N, 3n]


def projected_gram(bloch_a: torch.Tensor, bloch_b: torch.Tensor | None = None,
                   gamma: float = 1.0) -> np.ndarray:
    b = bloch_a if bloch_b is None else bloch_b
    a2 = (bloch_a ** 2).sum(dim=1, keepdim=True)          # [Na,1]
    b2 = (b ** 2).sum(dim=1, keepdim=True)                # [Nb,1]
    d2 = a2 + b2.transpose(0, 1) - 2.0 * (bloch_a @ b.transpose(0, 1))
    d2 = torch.clamp(d2, min=0.0)
    k = torch.exp(-0.5 * gamma * d2)
    return k.detach().cpu().numpy().astype(np.float64)


def psd_clip(K: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone (clip negative eigenvalues)."""
    Ksym = 0.5 * (K + K.T)
    w, V = np.linalg.eigh(Ksym)
    w = np.clip(w, 0.0, None)
    return (V * w) @ V.T


def add_shot_noise(K: np.ndarray, shots: int, rng: np.random.Generator) -> np.ndarray:
    """Binomial shot sampling of a fidelity kernel (entries in [0,1])."""
    if shots is None or shots <= 0:
        return K
    p = np.clip(K, 0.0, 1.0)
    return rng.binomial(shots, p).astype(np.float64) / float(shots)


def add_depolarizing_noise(K: np.ndarray, p: float, n_qubits: int) -> np.ndarray:
    """Global depolarising-noise model for the FIDELITY kernel.

    A global depolarising channel of strength ``p`` drives each embedded state toward
    the maximally mixed state, rho -> (1-p) rho + p I/2^n. For the overlap kernel
    k = Tr[rho(x) rho(x')] (both embeddings noisy) every entry contracts toward the
    trivial value 2^-n:

        k_noisy = (1-p)^2 k_ideal + (1 - (1-p)^2) / 2^n.

    p=0 returns the noiseless kernel; p=1 collapses K to the all-2^-n matrix (total
    exponential concentration — the  failure mode). The result stays PSD: it is
    (1-p)^2 K (PSD) plus a non-negative constant shift c*J (rank-1 PSD). This is the
    standard analytic global-depolarising model — far cheaper than a per-gate
    density-matrix simulation and exact for a single global channel, so the 
    noise-sensitivity sweep runs on full subsets rather than a handful of samples."""
    if p is None or float(p) <= 0.0:
        return K
    p = float(min(max(p, 0.0), 1.0))
    q2 = (1.0 - p) ** 2
    floor = (1.0 - q2) / float(2 ** int(n_qubits))
    return q2 * K + floor


def fidelity_kernel_blocks(
    fm: CrossLayerFeatureMap,
    X_train, aoi_train, X_test=None, aoi_test=None,
    params=None, shots=None, psd_repair: str = "clip",
    rng: np.random.Generator | None = None,
):
    """Return (K_train [Ntr,Ntr], K_test [Nte,Ntr] or None)."""
    s_tr = fm.statevectors(X_train, aoi_train, params)
    K_tr = fidelity_gram(s_tr)
    K_te = None
    if X_test is not None:
        s_te = fm.statevectors(X_test, aoi_test, params)
        K_te = fidelity_gram(s_te, s_tr)
    if shots:
        rng = rng or np.random.default_rng(0)
        K_tr = add_shot_noise(K_tr, shots, rng)
        np.fill_diagonal(K_tr, 1.0)
        if K_te is not None:
            K_te = add_shot_noise(K_te, shots, rng)
        if psd_repair == "clip":
            K_tr = psd_clip(K_tr)
    return K_tr, K_te


def projected_kernel_blocks(
    fm: CrossLayerFeatureMap,
    X_train, aoi_train, X_test=None, aoi_test=None,
    params=None, gamma: float = 1.0,
):
    s_tr = fm.statevectors(X_train, aoi_train, params)
    b_tr = bloch_features(s_tr, fm.n)
    K_tr = projected_gram(b_tr, gamma=gamma)
    K_te = None
    if X_test is not None:
        s_te = fm.statevectors(X_test, aoi_test, params)
        b_te = bloch_features(s_te, fm.n)
        K_te = projected_gram(b_te, b_tr, gamma=gamma)
    return K_tr, K_te


# --------------------------------------------------------------------------- #
# Nyström landmark approximation — scale the O(N^2) Gram to O(N*l)
# --------------------------------------------------------------------------- #
# The fidelity/projected Gram needs binom(N,2) quantum kernel entries. With l<<N
# landmarks the Nyström method reconstructs K ~= C W^+ C^T from just the [N, l]
# train-vs-landmark block C and the [l, l] landmark Gram W, so only O(N*l) entries
# are evaluated on the (expensive) quantum embedding — exactly the  mitigation.
# The approximate Gram slots straight into the existing precomputed-kernel SVM, CV,
# significance and diagnostics machinery; for N so large that even the dense [N, N]
# reconstruction won't fit RAM, use the explicit Nyström features below with a linear
# SVM instead (nystrom_features()).
def _nystrom_combine(C_tr, W, C_te=None, reg: float = 1e-10):
    """K ~= C W^+ C^T from the landmark blocks. Returns (K_tr, K_te)."""
    ell = W.shape[0]
    W_pinv = np.linalg.pinv(W + reg * np.eye(ell))
    M_tr = C_tr @ W_pinv                         # [N, l]
    K_tr = M_tr @ C_tr.T                         # [N, N]
    K_tr = 0.5 * (K_tr + K_tr.T)
    K_te = (C_te @ W_pinv) @ C_tr.T if C_te is not None else None  # [Nte, N]
    return K_tr, K_te


def nystrom_fidelity_blocks(
    fm: CrossLayerFeatureMap,
    X_train, aoi_train, landmark_idx, X_test=None, aoi_test=None,
    params=None, shots=None, rng: np.random.Generator | None = None,
    reg: float = 1e-10,
):
    """Nyström-approximated FIDELITY Gram. `landmark_idx` indexes rows of
    X_train (the l landmarks). Only the [N,l] and [l,l] quantum overlaps are computed."""
    s_tr = fm.statevectors(X_train, aoi_train, params)
    s_lm = s_tr[landmark_idx]
    W = fidelity_gram(s_lm)                       # [l, l]
    C_tr = fidelity_gram(s_tr, s_lm)              # [N, l]
    C_te = None
    if X_test is not None:
        s_te = fm.statevectors(X_test, aoi_test, params)
        C_te = fidelity_gram(s_te, s_lm)          # [Nte, l]
    if shots:
        rng = rng or np.random.default_rng(0)
        W = add_shot_noise(W, shots, rng)
        np.fill_diagonal(W, 1.0)
        C_tr = add_shot_noise(C_tr, shots, rng)
        if C_te is not None:
            C_te = add_shot_noise(C_te, shots, rng)
    return _nystrom_combine(C_tr, W, C_te, reg=reg)


def nystrom_projected_blocks(
    fm: CrossLayerFeatureMap,
    X_train, aoi_train, landmark_idx, X_test=None, aoi_test=None,
    params=None, gamma: float = 1.0, reg: float = 1e-10,
):
    """Nyström-approximated PROJECTED Gram."""
    b_tr = bloch_features(fm.statevectors(X_train, aoi_train, params), fm.n)
    b_lm = b_tr[landmark_idx]
    W = projected_gram(b_lm, gamma=gamma)
    C_tr = projected_gram(b_tr, b_lm, gamma=gamma)
    C_te = None
    if X_test is not None:
        b_te = bloch_features(fm.statevectors(X_test, aoi_test, params), fm.n)
        C_te = projected_gram(b_te, b_lm, gamma=gamma)
    return _nystrom_combine(C_tr, W, C_te, reg=reg)


def nystrom_features(C, W, reg: float = 1e-10) -> np.ndarray:
    """Explicit Nyström feature map Phi = C W^{+1/2} (so Phi Phi^T ~= K). Feed Phi to a
    LINEAR SVM for true O(N*l) memory at very large N (the dense Gram above is O(N^2))."""
    ell = W.shape[0]
    w, V = np.linalg.eigh(0.5 * (W + W.T) + reg * np.eye(ell))
    w = np.clip(w, reg, None)
    W_inv_sqrt = (V * (1.0 / np.sqrt(w))) @ V.T
    return C @ W_inv_sqrt
