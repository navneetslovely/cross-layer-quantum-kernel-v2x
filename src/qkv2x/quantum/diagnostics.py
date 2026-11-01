"""QML diagnostics.

  centered_kta  cheap model-selection / feature-map-quality score
  geometric_difference  g_CQ (Huang et al.): necessary condition for a quantum
                          advantage between a classical kernel K_C and quantum K_Q
  kernel_offdiag_variance exponential-concentration guard rail
  effective_dimension  participation ratio of the kernel spectrum
"""
from __future__ import annotations

import numpy as np


def _center(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    return H @ K @ H


def centered_kta(K: np.ndarray, y: np.ndarray) -> float:
    """Centered kernel-target alignment with y in {-1,+1}."""
    ypm = np.where(np.asarray(y) > 0, 1.0, -1.0)
    Kc = _center(K)
    T = np.outer(ypm, ypm)
    num = float((Kc * T).sum())
    den = float(np.linalg.norm(Kc) * np.linalg.norm(T)) + 1e-12
    return num / den


def _normalize_trace(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    tr = np.trace(K)
    if tr <= 0:
        return K
    return n * K / tr


def geometric_difference(K_classical: np.ndarray, K_quantum: np.ndarray,
                         reg: float = 1e-6) -> float:
    """g_CQ = sqrt( || sqrt(Kc) Kq^{-1} sqrt(Kc) ||_inf ).

    Both kernels are trace-normalised to N first (Huang et al. convention). A
    large value certifies a labelling on which K_quantum separates data that
    K_classical cannot. Returned value is real and >= 0.
    """
    Kc = _normalize_trace(0.5 * (K_classical + K_classical.T))
    Kq = _normalize_trace(0.5 * (K_quantum + K_quantum.T))
    n = Kc.shape[0]

    # sqrt(Kc) via eigendecomposition (clip tiny negatives from round-off)
    wc, Vc = np.linalg.eigh(Kc)
    wc = np.clip(wc, 0.0, None)
    sqrtKc = (Vc * np.sqrt(wc)) @ Vc.T

    # Kq^{-1} with Tikhonov regularisation
    wq, Vq = np.linalg.eigh(Kq + reg * np.eye(n))
    wq = np.clip(wq, reg, None)
    Kq_inv = (Vq * (1.0 / wq)) @ Vq.T

    M = sqrtKc @ Kq_inv @ sqrtKc
    M = 0.5 * (M + M.T)
    top = float(np.linalg.eigvalsh(M)[-1])
    return float(np.sqrt(max(top, 0.0)))


def kernel_offdiag_variance(K: np.ndarray) -> float:
    """Variance of off-diagonal kernel entries. Vanishing variance (~2^-n) signals
    exponential concentration toward the identity."""
    n = K.shape[0]
    if n < 2:
        return 0.0
    mask = ~np.eye(n, dtype=bool)
    return float(np.var(K[mask]))


def effective_dimension(K: np.ndarray) -> float:
    """Participation ratio (tr K)^2 / tr(K^2): an effective rank of the kernel."""
    w = np.linalg.eigvalsh(0.5 * (K + K.T))
    w = np.clip(w, 0.0, None)
    s1 = float(w.sum())
    s2 = float((w ** 2).sum())
    if s2 <= 0:
        return 0.0
    return (s1 ** 2) / s2
