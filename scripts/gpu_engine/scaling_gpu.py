"""scaling_gpu.py -- pure-torch statevector engine for the n=14..20 scaling extension (GPU).

Reimplements CrossLayerFeatureMap's circuit EXACTLY (PennyLane default.qubit conventions:
wire 0 = leftmost/most-significant amplitude index; RY/RZ/IsingZZ definitions) in raw torch,
so KTA training runs on CUDA at qubit counts where PennyLane backprop cannot fit.

Circuit per layer l (same op order as featuremap.py::_circuit):
    for i in wires:  RY(w[l,i]*x_i) then RZ(u[l,i]*x_i)          -- NB: per-wire RY,RZ pair
    for c,(i,j) in couples:  IsingZZ(-2*gamma[l,c]*(pi-x_i)(pi-x_j))
AoI modulation on network wires first: x_net *= exp(-aoi/kappa).

Gate conventions (PennyLane):
    RY(t) = [[cos t/2, -sin t/2],[sin t/2, cos t/2]]
    RZ(t) = diag(e^{-it/2}, e^{+it/2})
    IsingZZ(p) = exp(-i p/2 Z(x)Z) = diag phases: e^{-ip/2} on |00>,|11>; e^{+ip/2} on |01>,|10>

Training = train_feature_map semantics: Adam(lr), epochs x precomputed balanced batches,
loss = -centred_KTA(fidelity_gram(states), y_pm), best-alignment snapshot kept.
Memory strategy (exact, not approximate): full-batch loss on detached states -> dL/dS via a
tiny autograd pass at the Gram level -> chain into the circuit per MICROBATCH with
torch.autograd.backward(states_mb, grad_tensors=dL/dS[mb]). Peak graph memory ~ microbatch.

Outputs per (n, training, seed): grams/gram_n{n}_{training}_s{seed}.npz with the 800x800
eval Gram (float64; states c64 as in the published protocol, GEMM accumulated in c128),
final params, alignment history, timings. Metrics/ROC are computed LOCALLY afterwards.

Usage (on the box):
    python scaling_gpu.py --inputs scaling_inputs_n14.npz --training kta --seed 11 --gpu 0
    python scaling_gpu.py --inputs scaling_inputs_n20.npz --training fixed --seed 11 --gpu 1
    python scaling_gpu.py --selftest          # gate-level unit checks (any machine, CPU ok)
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- circuit ops
def apply_ry(state: torch.Tensor, wire: int, theta: torch.Tensor) -> torch.Tensor:
    """state: [B, 2^n] complex; theta: [B] real. Returns new state."""
    B = state.shape[0]
    n = int(round(math.log2(state.shape[1])))
    left, right = 2 ** wire, 2 ** (n - wire - 1)
    s = state.view(B, left, 2, right)
    c = torch.cos(theta / 2).view(B, 1, 1).to(state.dtype)
    sn = torch.sin(theta / 2).view(B, 1, 1).to(state.dtype)
    s0, s1 = s[:, :, 0, :], s[:, :, 1, :]
    return torch.stack((c * s0 - sn * s1, sn * s0 + c * s1), dim=2).reshape(B, -1)


def apply_rz(state: torch.Tensor, wire: int, theta: torch.Tensor) -> torch.Tensor:
    B = state.shape[0]
    n = int(round(math.log2(state.shape[1])))
    left, right = 2 ** wire, 2 ** (n - wire - 1)
    s = state.view(B, left, 2, right)
    half = theta / 2
    ph0 = torch.exp(-1j * half).view(B, 1, 1).to(state.dtype)
    ph1 = torch.exp(1j * half).view(B, 1, 1).to(state.dtype)
    return torch.stack((ph0 * s[:, :, 0, :], ph1 * s[:, :, 1, :]), dim=2).reshape(B, -1)


def apply_zz(state: torch.Tensor, wi: int, wj: int, phi: torch.Tensor,
             sign_cache: dict) -> torch.Tensor:
    """IsingZZ(phi) with per-sample phi: phase e^{-i phi/2 * s}, s=+1 equal bits, -1 differing."""
    B = state.shape[0]
    n = int(round(math.log2(state.shape[1])))
    key = (wi, wj, n, state.device.type)
    if key not in sign_cache:
        zi = torch.tensor([1.0, -1.0], device=state.device)
        si = zi.view([1] + [2 if k == wi else 1 for k in range(n)])
        sj = zi.view([1] + [2 if k == wj else 1 for k in range(n)])
        full = (si * sj).expand([1] + [2] * n)               # materialise the 2^n lattice
        sign_cache[key] = full.reshape(1, -1).contiguous()   # [1, 2^n] float, +-1
    s = sign_cache[key]
    phase = torch.exp((-1j) * (phi.view(B, 1) / 2) * s).to(state.dtype)
    return state * phase


def circuit_states(x: torch.Tensor, w: torch.Tensor, u: torch.Tensor, g: torch.Tensor,
                   couples: list, n: int, L: int, cdtype, sign_cache: dict) -> torch.Tensor:
    """x: [B, n] real (already AoI-modulated). Returns [B, 2^n] states."""
    B = x.shape[0]
    state = torch.zeros((B, 2 ** n), dtype=cdtype, device=x.device)
    state[:, 0] = 1.0
    for l in range(L):
        for i in range(n):
            state = apply_ry(state, i, w[l, i] * x[:, i])
            state = apply_rz(state, i, u[l, i] * x[:, i])
        for c, (i, j) in enumerate(couples):
            psi = (math.pi - x[:, i]) * (math.pi - x[:, j])
            state = apply_zz(state, i, j, -2.0 * g[l, c] * psi, sign_cache)
    return state


# --------------------------------------------------------------------------- KTA pieces
def fidelity_gram(states: torch.Tensor) -> torch.Tensor:
    o = states @ states.conj().transpose(0, 1)
    return o.real ** 2 + o.imag ** 2


def centred_kta(K: torch.Tensor, ypm: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    H = torch.eye(n, dtype=K.dtype, device=K.device) - 1.0 / n
    Kc = H @ K @ H
    T = torch.outer(ypm, ypm)
    return (Kc * T).sum() / (torch.linalg.norm(Kc) * torch.linalg.norm(T) + 1e-12)


# --------------------------------------------------------------------------- engine
class Engine:
    def __init__(self, nm, nn, L, kappa, device, cdtype=torch.complex64):
        self.nm, self.nn, self.n, self.L = nm, nn, nm + nn, L
        self.kappa = kappa
        self.device = torch.device(device)
        self.cdtype = cdtype
        self.rdtype = torch.float32 if cdtype == torch.complex64 else torch.float64
        self.couples = [(i, j) for i in range(nm) for j in range(nm, nm + nn)]
        self.sign_cache: dict = {}

    def modulate(self, x, aoi):
        lam = torch.exp(-aoi / self.kappa).reshape(-1, 1)
        xm = x.clone()
        xm[:, self.nm:] = x[:, self.nm:] * lam
        return xm

    def states(self, X, aoi, params, grad=False, mb=None):
        """Forward states, optionally without grad, optionally in microbatches (no-grad only)."""
        Xt = torch.as_tensor(X, dtype=self.rdtype, device=self.device)
        at = torch.as_tensor(aoi, dtype=self.rdtype, device=self.device)
        xm = self.modulate(Xt, at)
        if grad:
            return circuit_states(xm, params["w"], params["u"], params["gamma"],
                                  self.couples, self.n, self.L, self.cdtype, self.sign_cache)
        outs = []
        mb = mb or Xt.shape[0]
        with torch.no_grad():
            for s in range(0, Xt.shape[0], mb):
                outs.append(circuit_states(xm[s:s + mb], params["w"], params["u"],
                                           params["gamma"], self.couples, self.n, self.L,
                                           self.cdtype, self.sign_cache))
        return torch.cat(outs, 0)

    def kta_step_grads(self, Xb, ab, ypm, params, mb):
        """Exact full-batch KTA gradient via state-level VJP over microbatches.
        Returns (alignment float, and leaves grads accumulated in params tensors)."""
        # pass 1: states without graph
        S = self.states(Xb, ab, params, grad=False, mb=mb)
        S_leaf = S.detach().clone().requires_grad_(True)
        K = fidelity_gram(S_leaf)
        y = torch.as_tensor(ypm, dtype=K.dtype, device=K.device)
        align = centred_kta(K, y)
        (-align).backward()                       # -> S_leaf.grad = dL/dS (conj convention)
        gS = S_leaf.grad
        # pass 2: chain into circuit per microbatch
        Xt = torch.as_tensor(Xb, dtype=self.rdtype, device=self.device)
        at = torch.as_tensor(ab, dtype=self.rdtype, device=self.device)
        xm = self.modulate(Xt, at)
        for s in range(0, Xt.shape[0], mb):
            smb = circuit_states(xm[s:s + mb], params["w"], params["u"], params["gamma"],
                                 self.couples, self.n, self.L, self.cdtype, self.sign_cache)
            torch.autograd.backward(smb, grad_tensors=gS[s:s + mb])
        return float(align.detach().cpu())


def train_kta(eng: Engine, Xtr, aoitr, ytr, batch_idx, lr=0.05, mb=8):
    L, n, nc = eng.L, eng.n, len(eng.couples)
    params = {
        "w": torch.full((L, n), 1.0, dtype=eng.rdtype, device=eng.device, requires_grad=True),
        "u": torch.full((L, n), 1.0, dtype=eng.rdtype, device=eng.device, requires_grad=True),
        "gamma": torch.full((L, nc), 0.5, dtype=eng.rdtype, device=eng.device, requires_grad=True),
    }
    opt = torch.optim.Adam([params["w"], params["u"], params["gamma"]], lr=lr)
    ypm_all = np.where(ytr > 0, 1.0, -1.0)
    best_a, best = -np.inf, None
    hist = []
    for ep in range(batch_idx.shape[0]):
        bi = batch_idx[ep]
        opt.zero_grad()
        a = eng.kta_step_grads(Xtr[bi], aoitr[bi], ypm_all[bi], params, mb)
        opt.step()
        hist.append(a)
        if a > best_a:
            best_a = a
            best = {k: v.detach().clone() for k, v in params.items()}
        if ep % 20 == 0:
            print(f"  ep {ep:3d} align={a:.4f} (best {best_a:.4f})", flush=True)
    return best, best_a, hist


# --------------------------------------------------------------------------- selftest
def selftest():
    """Gate-level checks against closed forms (no PennyLane needed here; the PennyLane
    cross-check runs locally via test_engine_equivalence.py)."""
    torch.manual_seed(0)
    dev = "cpu"
    # RY on |0>: [cos t/2, sin t/2]
    st = torch.zeros((1, 2), dtype=torch.complex128); st[0, 0] = 1
    t = torch.tensor([0.7], dtype=torch.float64)
    out = apply_ry(st, 0, t)[0]
    assert abs(out[0] - math.cos(0.35)) < 1e-12 and abs(out[1] - math.sin(0.35)) < 1e-12
    # RZ phases
    st = torch.ones((1, 2), dtype=torch.complex128) / math.sqrt(2)
    out = apply_rz(st, 0, t)[0]
    assert abs(out[0] - math.e ** 0j * np.exp(-1j * 0.35) / math.sqrt(2)) < 1e-12
    # IsingZZ on |++>: phases by parity
    st = torch.ones((1, 4), dtype=torch.complex128) / 2
    out = apply_zz(st, 0, 1, torch.tensor([0.9], dtype=torch.float64), {})[0]
    exp = np.array([np.exp(-1j * 0.45), np.exp(1j * 0.45), np.exp(1j * 0.45), np.exp(-1j * 0.45)]) / 2
    assert np.abs(out.numpy() - exp).max() < 1e-12
    # norm preservation on random circuit
    eng = Engine(2, 2, 2, 0.5, dev, torch.complex128)
    X = np.random.default_rng(1).uniform(0, math.pi, (5, 4))
    A = np.random.default_rng(2).uniform(0, 1, 5)
    p = {"w": torch.ones(2, 4, dtype=torch.float64), "u": torch.ones(2, 4, dtype=torch.float64),
         "gamma": torch.full((2, 4), 0.5, dtype=torch.float64)}
    S = eng.states(X, A, p)
    nrm = torch.linalg.norm(S, dim=1)
    assert torch.allclose(nrm, torch.ones(5, dtype=torch.float64), atol=1e-12)
    # VJP gradient == direct autograd gradient (small case)
    ypm = np.array([1., -1., 1., -1., 1.])
    pg = {k: v.clone().requires_grad_(True) for k, v in p.items()}
    a1 = eng.kta_step_grads(X, A, ypm, pg, mb=2)
    g_vjp = {k: v.grad.clone() for k, v in pg.items()}
    pg2 = {k: v.detach().clone().requires_grad_(True) for k, v in p.items()}
    Xt = torch.as_tensor(X, dtype=torch.float64); at = torch.as_tensor(A, dtype=torch.float64)
    S2 = circuit_states(eng.modulate(Xt, at), pg2["w"], pg2["u"], pg2["gamma"],
                        eng.couples, eng.n, eng.L, torch.complex128, {})
    K2 = fidelity_gram(S2)
    a2 = centred_kta(K2, torch.as_tensor(ypm, dtype=torch.float64))
    (-a2).backward()
    for k in pg:
        d = (g_vjp[k] - pg2[k].grad).abs().max().item()
        assert d < 1e-9, (k, d)
    assert abs(a1 - float(a2)) < 1e-9
    print("SELFTEST OK (gates, norms, VJP==autograd)")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs")
    ap.add_argument("--training", choices=["kta", "fixed"])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mb", type=int, default=0, help="microbatch (0 = auto)")
    ap.add_argument("--outdir", default="grams")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); return

    z = np.load(a.inputs)
    nm, nn = int(z["n_mobility"]), int(z["n_network"])
    n = nm + nn
    dev = f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu"
    eng = Engine(nm, nn, int(z["n_layers"]), float(z["kappa"]), dev)
    seeds = list(z["seeds"])
    si = seeds.index(a.seed)
    Xr, y, aoi = z["Xr"], z["y"], z["aoi"]
    idx_tr = z["idx_tr"]
    mb = a.mb or max(1, int((12 * 2 ** 30) // ((2 * n + len(eng.couples)) * eng.L
                                              * (2 ** n) * 8)))
    print(f"n={n} ({nm}+{nn}) dev={dev} training={a.training} seed={a.seed} mb={mb}", flush=True)

    t0 = time.time()
    if a.training == "kta":
        best, best_a, hist = train_kta(eng, Xr[idx_tr], aoi[idx_tr], y[idx_tr],
                                       z["batch_idx"][si], mb=mb)
        params = best
    else:
        L, nc = eng.L, len(eng.couples)
        params = {"w": torch.ones(L, n, dtype=eng.rdtype, device=eng.device),
                  "u": torch.ones(L, n, dtype=eng.rdtype, device=eng.device),
                  "gamma": torch.ones(L, nc, dtype=eng.rdtype, device=eng.device)}
        best_a, hist = float("nan"), []
    t_train = time.time() - t0

    both = np.concatenate([z["tr_e"][si], z["te_e"][si]])
    t1 = time.time()
    S = eng.states(Xr[both], aoi[both], params, grad=False, mb=max(mb, 16))
    S128 = S.to(torch.complex128)
    O = S128 @ S128.conj().transpose(0, 1)
    K = (O.real ** 2 + O.imag ** 2).cpu().numpy()
    t_eval = time.time() - t1

    outdir = Path(a.outdir); outdir.mkdir(exist_ok=True)
    np.savez_compressed(
        outdir / f"gram_n{n}_{a.training}_s{a.seed}.npz",
        K=K.astype(np.float64), y_both=y[both], n_tr=len(z["tr_e"][si]),
        w=params["w"].detach().cpu().numpy(), u=params["u"].detach().cpu().numpy(),
        gamma=params["gamma"].detach().cpu().numpy(),
        best_alignment=best_a, align_hist=np.array(hist, dtype=np.float64),
        wall_train_s=t_train, wall_eval_s=t_eval, device=dev, cdtype="complex64",
        n_mobility=nm, n_network=nn,
    )
    print(f"DONE n={n} {a.training} s{a.seed}: train {t_train:.0f}s eval {t_eval:.0f}s "
          f"best_align={best_a:.4f}", flush=True)


if __name__ == "__main__":
    main()
