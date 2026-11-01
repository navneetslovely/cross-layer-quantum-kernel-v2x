"""Small-n rank statistics shared by the separability-correlation studies.

  * exact_one_sided_p -- exact permutation p-value for a Spearman correlation (small n only),
  * _stouffer        -- Stouffer's sqrt(n)-weighted Z for combining one-sided p-values,
  * _fisher          -- Fisher's chi-square method for combining p-values.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
from scipy.stats import chi2, norm, rankdata

def _spearman_from_ranks(rx: np.ndarray, ry: np.ndarray) -> float:
    n = len(rx)
    d2 = float(np.sum((rx - ry) ** 2))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def exact_one_sided_p(x: np.ndarray, y: np.ndarray, positive: bool) -> tuple[float, float]:
    """Exact permutation p for Spearman(x,y) in the stated direction (small n only)."""
    rx = rankdata(x)
    ry = rankdata(y)
    obs = _spearman_from_ranks(rx, ry)
    n = len(rx)
    cnt = tot = 0
    eps = 1e-12
    for perm in itertools.permutations(range(n)):
        rho = _spearman_from_ranks(rx, ry[list(perm)])
        tot += 1
        if (positive and rho >= obs - eps) or (not positive and rho <= obs + eps):
            cnt += 1
    return obs, cnt / tot


def _stouffer(ps: list[float], ns: list[int]) -> dict:
    # one-sided p -> z; sqrt(n) weights; clip to avoid +-inf at exact 0/1
    z = [norm.isf(min(max(p, 1e-12), 1 - 1e-12)) for p in ps]
    w = [math.sqrt(n) for n in ns]
    Z = sum(wi * zi for wi, zi in zip(w, z)) / math.sqrt(sum(wi * wi for wi in w))
    return {"Z": Z, "p": float(norm.sf(Z))}


def _fisher(ps: list[float]) -> dict:
    X2 = -2.0 * sum(math.log(min(max(p, 1e-12), 1.0)) for p in ps)
    return {"chi2": X2, "df": 2 * len(ps), "p": float(chi2.sf(X2, 2 * len(ps)))}
