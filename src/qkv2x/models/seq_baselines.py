"""Sequence baselines over per-window MESSAGE sequences.

The windowed detectors consume per-window aggregate vectors; the published V2X
misbehaviour-detection state of the art also includes recurrent models over the
raw message sequence (e.g. Alladi et al.). `LSTMDetector` is that comparator:
a small LSTM over the [window_size, n_features] per-message feature sequence
(zero-padded, packed by true length), trained with class-weighted BCE, model
selection on the validation split by average precision. CPU-friendly at the
window counts used here. Deterministic given `seed`.
"""
from __future__ import annotations

import numpy as np


class LSTMDetector:
    def __init__(self, hidden: int = 64, layers: int = 1, epochs: int = 40,
                 batch: int = 256, lr: float = 1e-3, seed: int = 0,
                 device: str = "cpu"):
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.epochs = int(epochs)
        self.batch = int(batch)
        self.lr = float(lr)
        self.seed = int(seed)
        self.device = device
        self._net = None
        self._mu = None
        self._sd = None
        self.best_val_ap_ = None

    # ------------------------------------------------------------------ #
    def _build(self, n_feat: int):
        import torch.nn as nn

        class _Net(nn.Module):
            def __init__(self, f, h, l):
                super().__init__()
                self.lstm = nn.LSTM(f, h, num_layers=l, batch_first=True)
                self.head = nn.Linear(h, 1)

            def forward(self, x, lengths):
                import torch
                from torch.nn.utils.rnn import (pack_padded_sequence)
                packed = pack_padded_sequence(
                    x, lengths.cpu(), batch_first=True, enforce_sorted=False)
                _, (h_n, _) = self.lstm(packed)
                return self.head(h_n[-1]).squeeze(-1)

        return _Net(n_feat, self.hidden, self.layers)

    def _norm(self, seq, lengths, fit=False):
        """Per-feature standardisation over REAL (unpadded) steps."""
        if fit:
            mask = (np.arange(seq.shape[1])[None, :] < lengths[:, None])
            flat = seq[mask]                       # [sum(lengths), F]
            self._mu = flat.mean(axis=0)
            self._sd = np.where(flat.std(axis=0) > 1e-9, flat.std(axis=0), 1.0)
        out = (seq - self._mu) / self._sd
        # keep padding at zero
        mask = (np.arange(seq.shape[1])[None, :] < lengths[:, None])
        return out * mask[:, :, None]

    # ------------------------------------------------------------------ #
    def fit(self, seq_tr, len_tr, y_tr, seq_va=None, len_va=None, y_va=None):
        import torch
        from sklearn.metrics import average_precision_score

        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        y_tr = np.asarray(y_tr).astype(np.float32)
        Xn = self._norm(np.asarray(seq_tr, dtype=np.float32),
                        np.asarray(len_tr), fit=True)
        net = self._build(Xn.shape[2]).to(self.device)
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        npos = max(float(y_tr.sum()), 1.0)
        pw = torch.tensor((len(y_tr) - npos) / npos, dtype=torch.float32)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

        Xt = torch.as_tensor(Xn)
        Lt = torch.as_tensor(np.asarray(len_tr), dtype=torch.int64)
        yt = torch.as_tensor(y_tr)
        best_ap, best_state = -np.inf, None
        n = len(y_tr)
        for ep in range(self.epochs):
            net.train()
            order = rng.permutation(n)
            for s in range(0, n, self.batch):
                idx = order[s: s + self.batch]
                opt.zero_grad()
                out = net(Xt[idx], Lt[idx])
                loss = lossf(out, yt[idx])
                loss.backward()
                opt.step()
            if seq_va is not None and y_va is not None and len(np.unique(y_va)) == 2:
                sva = self.decision_function(seq_va, len_va, _net=net)
                ap = average_precision_score(np.asarray(y_va).astype(int), sva)
                if ap > best_ap:
                    best_ap = ap
                    best_state = {k: v.detach().clone()
                                  for k, v in net.state_dict().items()}
        if best_state is not None:
            net.load_state_dict(best_state)
            self.best_val_ap_ = float(best_ap)
        self._net = net
        return self

    def decision_function(self, seq, lengths, _net=None):
        import torch
        net = _net if _net is not None else self._net
        Xn = self._norm(np.asarray(seq, dtype=np.float32), np.asarray(lengths))
        net.eval()
        outs = []
        with torch.no_grad():
            X = torch.as_tensor(Xn)
            L = torch.as_tensor(np.asarray(lengths), dtype=torch.int64)
            for s in range(0, len(Xn), 1024):
                outs.append(net(X[s: s + 1024], L[s: s + 1024]).numpy())
        return np.concatenate(outs)

    def predict(self, seq, lengths):
        return (self.decision_function(seq, lengths) > 0.0).astype(int)
