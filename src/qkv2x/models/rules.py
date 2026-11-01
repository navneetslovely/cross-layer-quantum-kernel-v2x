"""Rule-based plausibility checks — a non-ML reference detector.

Implements the classic V2X misbehaviour plausibility family (ART / SSC / DMV style)
as calibrated thresholds on the cross-layer window features:

  position / cross-layer (ART): e_xl_max  (claim vs RSSI-implied distance)
  kinematics (DMV / SSC):  e_kin_max, speed_consistency_max
  flooding (DoS):  rate_ratio
  Sybil:  sybil_neighbors_max

Thresholds are calibrated from BENIGN training windows (a high quantile), so this
reference adapts to the scenario instead of relying on hand-tuned magic numbers.
A window is flagged if any normalised feature exceeds its threshold.
"""
from __future__ import annotations

import numpy as np

CHECK_FEATURES = [
    "e_xl_max",
    "e_kin_max",
    "speed_consistency_max",
    "rate_ratio",
    "sybil_neighbors_max",
]


class RulePlausibilityDetector:
    def __init__(self, rules_cfg: dict | None = None, quantile: float = 0.99):
        self.cfg = rules_cfg or {}
        self.quantile = float(quantile)
        self.thresholds_: dict[str, float] = {}
        self.cols_: dict[str, int] = {}

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]):
        y = np.asarray(y).astype(int)
        benign = X[y == 0]
        if benign.shape[0] == 0:
            benign = X
        for f in CHECK_FEATURES:
            if f not in feature_names:
                continue
            col = feature_names.index(f)
            self.cols_[f] = col
            thr = float(np.quantile(benign[:, col], self.quantile))
            # never allow a degenerate zero threshold
            self.thresholds_[f] = thr if thr > 1e-9 else max(1e-6, float(np.max(benign[:, col])))
        if not self.cols_:
            raise ValueError("RulePlausibilityDetector: no known check features present.")
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        ratios = []
        for f, col in self.cols_.items():
            thr = self.thresholds_[f]
            ratios.append(X[:, col] / thr)
        return np.max(np.vstack(ratios), axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.decision_function(X) > 1.0).astype(int)
