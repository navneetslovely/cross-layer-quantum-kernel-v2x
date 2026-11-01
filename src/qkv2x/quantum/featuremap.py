"""Cross-layer entangling quantum feature map — the novel core.

U_phi(x) = prod_l  U_ent^xl(x; gamma_l)  U_loc(x; w_l, u_l)  (data re-uploading)

  U_loc        : per-qubit RY(w_i x_i) RZ(u_i x_i)  (trainable scales -> trainable kernel)
  U_ent^xl     : ZZ couplings placed ONLY between a mobility qubit and a network
                 qubit, angle  gamma_ij * (pi - x_i)(pi - x_j)  (cross-layer prior)
  AoI modulation: network-block coordinates x_i <- exp(-AoI/kappa) * x_i, so stale
                 channel-derived features rotate toward zero.

Ablations (set via config): entangler = 'zz_full' reproduces an off-the-shelf
ZZFeatureMap-style coupling over ALL pairs; entangler = 'none' drops entanglement;
entangler = 'sparse_random' draws `sparse_edges` seeded random pairs from
ALL pairs — an edge-count-matched control for the bipartite cross-layer topology.

States are produced by exact statevector simulation (PennyLane default.qubit,
torch backend) so the fidelity kernel is a single Gram product and the map's
trainable parameters are differentiable end-to-end.
"""
from __future__ import annotations

import numpy as np
import pennylane as qml
import torch


class CrossLayerFeatureMap:
    def __init__(
        self,
        n_mobility: int,
        n_network: int,
        n_layers: int = 2,
        entangler: str = "cross_layer",
        coupling: str = "full_bipartite",
        aoi_modulation: bool = True,
        kappa: float = 0.5,
        device: str = "cpu",
        dtype: str = "complex64",
        sparse_edges: int | None = None,
        sparse_seed: int = 7,
    ):
        self.n_mobility = int(n_mobility)
        self.n_network = int(n_network)
        self.n = self.n_mobility + self.n_network
        self.n_layers = int(n_layers)
        self.entangler = entangler
        self.coupling = coupling
        self.aoi_modulation = bool(aoi_modulation)
        self.kappa = float(kappa)
        self.sparse_edges = sparse_edges
        self.sparse_seed = int(sparse_seed)
        self.torch_device = torch.device(device)
        self.cdtype = torch.complex64 if dtype == "complex64" else torch.complex128
        self.rdtype = torch.float32 if dtype == "complex64" else torch.float64

        self.mob_wires = list(range(self.n_mobility))
        self.net_wires = list(range(self.n_mobility, self.n))
        self._couples = self._build_couples()

        self._dev = qml.device("default.qubit", wires=self.n)
        self._qnode = qml.QNode(self._circuit, self._dev, interface="torch",
                                diff_method="backprop")

    # -- coupling topology ---------------------------------------------------
    def _build_couples(self) -> list[tuple[int, int]]:
        if self.entangler == "none":
            return []
        if self.entangler == "zz_full":
            return [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
        if self.entangler == "sparse_random":
            # edge-count-matched random control. Draw `sparse_edges`
            # pairs (default = |mobility x network|, matching cross_layer) from ALL
            # (i<j) pairs, seeded -> deterministic, register-agnostic.
            pool = [(i, j) for i in range(self.n) for j in range(i + 1, self.n)]
            k = self.sparse_edges if self.sparse_edges is not None else (
                self.n_mobility * self.n_network)
            k = min(int(k), len(pool))
            rng = np.random.default_rng(self.sparse_seed)
            sel = rng.choice(len(pool), size=k, replace=False)
            return [pool[int(s)] for s in np.sort(sel)]
        # cross_layer: only mobility <-> network
        couples = []
        if self.coupling == "banded":
            for a, i in enumerate(self.mob_wires):
                for b, j in enumerate(self.net_wires):
                    if abs(a - b) <= 1:
                        couples.append((i, j))
        else:  # full_bipartite
            for i in self.mob_wires:
                for j in self.net_wires:
                    couples.append((i, j))
        return couples

    # -- trainable parameters ------------------------------------------------
    def init_params(self, init_w=1.0, init_u=1.0, init_gamma=0.5,
                    requires_grad=True) -> dict[str, torch.Tensor]:
        L, n = self.n_layers, self.n
        nc = max(1, len(self._couples))
        w = torch.full((L, n), float(init_w), dtype=self.rdtype, device=self.torch_device)
        u = torch.full((L, n), float(init_u), dtype=self.rdtype, device=self.torch_device)
        g = torch.full((L, nc), float(init_gamma), dtype=self.rdtype, device=self.torch_device)
        if requires_grad:
            w.requires_grad_(True)
            u.requires_grad_(True)
            g.requires_grad_(True)
        return {"w": w, "u": u, "gamma": g}

    def fixed_params(self) -> dict[str, torch.Tensor]:
        """Non-trainable params reproducing a standard (Havlicek-style) map:
        unit local scales, unit ZZ weights."""
        return self.init_params(1.0, 1.0, 1.0, requires_grad=False)

    # -- data preparation ----------------------------------------------------
    def modulate(self, x: torch.Tensor, aoi: torch.Tensor | None) -> torch.Tensor:
        if not self.aoi_modulation or aoi is None:
            return x
        lam = torch.exp(-aoi / self.kappa).reshape(-1, 1)  # [B,1]
        xm = x.clone()
        if self.net_wires:
            xm[:, self.net_wires] = x[:, self.net_wires] * lam
        return xm

    # -- circuit -------------------------------------------------------------
    def _circuit(self, x, w, u, gamma):
        # x: [B, n] torch tensor (already AoI-modulated, in [0, pi])
        for layer in range(self.n_layers):
            for i in range(self.n):
                qml.RY(w[layer, i] * x[:, i], wires=i)
                qml.RZ(u[layer, i] * x[:, i], wires=i)
            for c, (i, j) in enumerate(self._couples):
                psi = (np.pi - x[:, i]) * (np.pi - x[:, j])
                # exp(i a Z_i Z_j) == IsingZZ(-2a); a = gamma * psi
                qml.IsingZZ(-2.0 * gamma[layer, c] * psi, wires=[i, j])
        return qml.state()

    # -- public: statevectors ------------------------------------------------
    def statevectors(self, X: np.ndarray | torch.Tensor, aoi=None,
                     params: dict[str, torch.Tensor] | None = None,
                     chunk: int = 256) -> torch.Tensor:
        if params is None:
            params = self.fixed_params()
        Xt = torch.as_tensor(np.asarray(X), dtype=self.rdtype, device=self.torch_device)
        aoit = (torch.as_tensor(np.asarray(aoi), dtype=self.rdtype,
                                device=self.torch_device) if aoi is not None else None)
        out = []
        for s in range(0, Xt.shape[0], chunk):
            xb = Xt[s:s + chunk]
            ab = aoit[s:s + chunk] if aoit is not None else None
            xb = self.modulate(xb, ab)
            state = self._qnode(xb, params["w"], params["u"], params["gamma"])
            out.append(state.to(self.cdtype))
        return torch.cat(out, dim=0)
