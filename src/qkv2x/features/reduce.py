"""Reduce raw window features to the qubit budget.

Block structure is preserved: the mobility features and the network features are
reduced separately, so the resulting coordinates map cleanly onto the disjoint
mobility / network qubit registers of the cross-layer feature map. Final
coordinates are scaled into [0, pi] for rotation-angle encoding.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class BlockReducer:
    def __init__(self, reduction_cfg: dict):
        self.cfg = reduction_cfg
        self.method = reduction_cfg.get("method", "per_block_pca")
        self.n_mobility = int(reduction_cfg["n_mobility"])
        self.n_network = int(reduction_cfg["n_network"])
        self.scale_range = tuple(reduction_cfg.get("scale_range", [0.0, np.pi]))
        self.clip_q = tuple(reduction_cfg.get("clip_quantiles", [0.005, 0.995]))
        self._fitted = False

    # -- internal helpers ----------------------------------------------------
    def _block_cols(self, blocks: list[str], which: str) -> np.ndarray:
        return np.array([i for i, b in enumerate(blocks) if b == which], dtype=int)

    def _fit_block(self, Xb: np.ndarray, n_out: int, tag: str):
        lo = np.quantile(Xb, self.clip_q[0], axis=0)
        hi = np.quantile(Xb, self.clip_q[1], axis=0)
        Xc = np.clip(Xb, lo, hi)
        scaler = StandardScaler().fit(Xc)
        Xs = scaler.transform(Xc)
        store = {"lo": lo, "hi": hi, "scaler": scaler}
        if self.method == "per_block_pca":
            k = min(n_out, Xs.shape[1], max(1, Xs.shape[0] - 1))
            if k < n_out:
                raise ValueError(
                    f"{tag}: cannot reduce to {n_out} components from shape {Xs.shape}.")
            pca = PCA(n_components=n_out, random_state=0).fit(Xs)
            Z = pca.transform(Xs)
            store["pca"] = pca
        elif self.method == "select":
            Z = Xs[:, :n_out]
            store["pca"] = None
        else:
            raise ValueError(f"unknown reduction method {self.method}")
        rmin = Z.min(axis=0)
        rmax = Z.max(axis=0)
        store["rmin"] = rmin
        store["rmax"] = rmax
        return store, self._scale01(Z, rmin, rmax)

    def _transform_block(self, Xb: np.ndarray, store: dict) -> np.ndarray:
        Xc = np.clip(Xb, store["lo"], store["hi"])
        Xs = store["scaler"].transform(Xc)
        Z = store["pca"].transform(Xs) if store["pca"] is not None else Xs[:, : self._n(store)]
        return self._scale01(Z, store["rmin"], store["rmax"])

    @staticmethod
    def _n(store: dict) -> int:
        return len(store["rmin"])

    def _scale01(self, Z: np.ndarray, rmin: np.ndarray, rmax: np.ndarray) -> np.ndarray:
        span = np.where((rmax - rmin) > 1e-12, rmax - rmin, 1.0)
        u = (Z - rmin) / span
        u = np.clip(u, 0.0, 1.0)
        a, b = self.scale_range
        return a + u * (b - a)

    # -- public API ----------------------------------------------------------
    def fit(self, X: np.ndarray, blocks: list[str]) -> "BlockReducer":
        if self.method == "select":
            sel = self.cfg.get("select", {})
            self._select_mob = sel.get("mobility", [])
            self._select_net = sel.get("network", [])
        self._mob_idx = self._block_cols(blocks, "mobility")
        self._net_idx = self._block_cols(blocks, "network")
        self._mob_store, _ = self._fit_block(X[:, self._mob_idx], self.n_mobility, "mobility")
        self._net_store, _ = self._fit_block(X[:, self._net_idx], self.n_network, "network")
        self.reduced_blocks = ["mobility"] * self.n_mobility + ["network"] * self.n_network
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("BlockReducer.transform called before fit")
        Zm = self._transform_block(X[:, self._mob_idx], self._mob_store)
        Zn = self._transform_block(X[:, self._net_idx], self._net_store)
        return np.hstack([Zm, Zn])

    def fit_transform(self, X: np.ndarray, blocks: list[str]) -> np.ndarray:
        return self.fit(X, blocks).transform(X)

    @property
    def n_qubits(self) -> int:
        return self.n_mobility + self.n_network
