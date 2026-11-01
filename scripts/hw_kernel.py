"""Circuit and exact-kernel helpers for the on-hardware fidelity-kernel runs.

  * _u_gate       -- appends the trained cross-layer map U_phi(x) to a Qiskit circuit,
  * _pair_circuit -- compute-uncompute circuit U_phi(x_j)^dagger U_phi(x_i) with measurement,
  * _sim_kernel   -- exact fidelity kernel from the statevector simulator (reference).
"""
from __future__ import annotations

import numpy as np


def _u_gate(qc, x, params, n_mob=4, n_net=4):
    """Append U_phi(x) (the trained cross-layer map) to circuit qc."""
    n = n_mob + n_net
    couples = [(i, j) for i in range(n_mob) for j in range(n_mob, n)]
    L = params["w"].shape[0]
    for layer in range(L):
        for i in range(n):
            qc.ry(float(params["w"][layer, i] * x[i]), i)
            qc.rz(float(params["u"][layer, i] * x[i]), i)
        for c, (i, j) in enumerate(couples):
            psi = (np.pi - x[i]) * (np.pi - x[j])
            qc.rzz(float(-2.0 * params["gamma"][layer, c] * psi), i, j)


def _pair_circuit(x_i, x_j, params):
    from qiskit import QuantumCircuit
    n = len(x_i)
    qc = QuantumCircuit(n, n)
    _u_gate(qc, x_i, params)
    inv = QuantumCircuit(n)
    _u_gate(inv, x_j, params)
    qc.compose(inv.inverse(), inplace=True)
    qc.measure(range(n), range(n))
    return qc


def _sim_kernel(x_a, x_b, params):
    """Exact fidelity kernel via the repo's own statevector path (ground truth)."""
    import torch

    from qkv2x.quantum.featuremap import CrossLayerFeatureMap
    from qkv2x.quantum.kernels import fidelity_gram
    fm = CrossLayerFeatureMap(4, 4, n_layers=params["w"].shape[0],
                              aoi_modulation=False)   # inputs pre-modulated
    p = {k: torch.as_tensor(v, dtype=fm.rdtype) for k, v in params.items()}
    s_a = fm.statevectors(x_a, None, p)
    s_b = fm.statevectors(x_b, None, p)
    return fidelity_gram(s_a, s_b)
