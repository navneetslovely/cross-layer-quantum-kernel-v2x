"""Transmitter-GROUPED, CROSS-FITTED separability divergence.

Recomputes D_full for the separability ordering (D_full vs realised detectability; ungrouped pooled rho=0.82)
with a transmitter-grouped, cross-fitted estimator. The ungrouped estimator subsamples random windows,
standardises on all data (benign+attack), and gives the rho no grouped CI. Here:

  * GROUPED subsampling: draw whole TRANSMITTER GROUPS (not random windows) for attack and benign,
    so within-transmitter correlation cannot inflate the divergence.
  * CROSS-FIT standardisation: compute the standardisation (median/scale) on a DISJOINT benign fold,
    apply to the held-out divergence sample -> the attack-vs-benign divergence is not self-referential.
  * GROUPED BOOTSTRAP CI: resample transmitter groups to put a CI on per-attack D_full, and resample
    attacks to put a CI on the frontier Spearman rho.

Reads runs/dev_grid + results/veremi_nextgen + the realised per-attack detectability Y (max-over-models
ROC). Writes results/frontier/frontier_grouped.json. Compares to the ungrouped rho=0.82.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _study_lib import REPO, RESULTS  # noqa: E402
import controllability as a1  # reuse realised-Y helpers + energy divergence  # noqa: E402

N_PER_CLASS = 300
BOOT_DFULL = 200
BOOT_RHO = 3000
RNG = np.random.default_rng(7)


def _energy(A, B):
    return float(2.0 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def _grouped_sample(X, groups, n, rng):
    """Draw WHOLE transmitter groups (shuffled) until >= n windows; return their row indices."""
    uniq = rng.permutation(np.unique(groups))
    idx, tot = [], 0
    for g in uniq:
        gi = np.where(groups == g)[0]
        idx.append(gi)
        tot += len(gi)
        if tot >= n:
            break
    return np.concatenate(idx)


def _xfit_standardize(fit, apply_to):
    """Robust standardisation fit on `fit` (a disjoint benign fold), applied to `apply_to`."""
    lo = np.quantile(fit, 0.005, axis=0)
    hi = np.quantile(fit, 0.995, axis=0)
    mu = np.clip(fit, lo, hi).mean(axis=0)
    sd = np.clip(fit, lo, hi).std(axis=0) + 1e-9
    return (np.clip(apply_to, lo, hi) - mu) / sd


def _d_full_grouped(Xa, Ga, Xb, Gb, rng):
    """One grouped, cross-fitted D_full(energy): standardise on benign fold-1, score attack vs benign fold-2."""
    bg = rng.permutation(np.unique(Gb))
    half = max(1, len(bg) // 2)
    b1 = np.isin(Gb, bg[:half])   # fold-1 benign: standardisation only
    b2 = np.isin(Gb, bg[half:])   # fold-2 benign: enters the divergence
    if b1.sum() < 10 or b2.sum() < 10:
        return None
    a_idx = _grouped_sample(Xa, Ga, min(N_PER_CLASS, (Ga.size)), rng)
    b2_idx = np.where(b2)[0]
    b2_idx = b2_idx[_grouped_sample(Xb[b2_idx], Gb[b2_idx], min(N_PER_CLASS, b2.sum()), rng)] \
        if b2.sum() > N_PER_CLASS else b2_idx
    fitX = Xb[b1]
    A = _xfit_standardize(fitX, Xa[a_idx])
    B = _xfit_standardize(fitX, Xb[b2_idx])
    return _energy(A, B)


def _per_attack(ds):
    X, y, atk, grp = ds["X"], ds["y"], ds["attack"], ds["groups"]
    benign = y == 0
    out = {}
    for a in sorted(set(atk[(y == 1)]) - {"none"}):
        am = (atk == a) & (y == 1)
        if am.sum() < 20 or benign.sum() < 40:
            continue
        Xa, Ga = X[am], grp[am]
        Xb, Gb = X[benign], grp[benign]
        vals = [v for v in (_d_full_grouped(Xa, Ga, Xb, Gb, RNG) for _ in range(BOOT_DFULL)) if v is not None]
        if len(vals) < 20:
            continue
        vals = np.array(vals)
        out[a] = {"D_full_grouped": float(vals.mean()),
                  "D_full_ci": [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))],
                  "n_attack_groups": int(np.unique(Ga).size)}
    return out


def _rho_with_ci(d_full, Y):
    attacks = [a for a in d_full if a in Y]
    x = np.array([d_full[a]["D_full_grouped"] for a in attacks])
    yv = np.array([Y[a] for a in attacks])
    rho = float(spearmanr(x, yv)[0])
    boots = []
    n = len(attacks)
    for _ in range(BOOT_RHO):
        s = RNG.integers(0, n, n)
        if len(np.unique(yv[s])) < 2:
            continue
        boots.append(spearmanr(x[s], yv[s])[0])
    boots = np.array(boots)
    return {"n_attacks": n, "spearman_rho": rho,
            "rho_ci": [float(np.nanquantile(boots, 0.025)), float(np.nanquantile(boots, 0.975))],
            "attacks": attacks}


def _loaders():
    from qkv2x.config import load_yaml
    from qkv2x.experiments.common import load_experiment_data
    cfg = load_yaml(REPO / "configs" / "features.yaml")
    d = load_experiment_data(REPO / "runs" / "dev_grid", cfg)
    dev = {"X": np.asarray(d.X, float), "y": np.asarray(d.y, int),
           "attack": np.asarray(d.attack_name), "groups": np.asarray(d.groups, int)}
    ng = None
    f = RESULTS / "veremi_nextgen" / "checkpoints" / "assembled.npz"
    if f.exists():
        z = np.load(f, allow_pickle=True)
        ng = {"X": np.asarray(z["X"], float), "y": np.asarray(z["y"], int),
              "attack": np.asarray([str(s) for s in z["attack_name"]]),
              "groups": np.asarray(z["groups"], int)}
    return dev, ng


def _Y_map(dfY, key, val="Y_rocauc"):
    return {str(r[key]): float(r[val]) for _, r in dfY.iterrows()}


def main() -> None:
    dev, ng = _loaders()
    res = {"study": "frontier_grouped_crossfit",
           "method": ("transmitter-grouped subsampling + cross-fit standardisation (benign fold-1 fits, "
                      "fold-2 scored) + grouped bootstrap CI; energy distance. Compare to ungrouped rho=0.82."),
           "params": {"n_per_class": N_PER_CLASS, "boot_dfull": BOOT_DFULL, "boot_rho": BOOT_RHO},
           "per_dataset": {}}

    dev_df = _per_attack(dev)
    Ydev = _Y_map(a1._Y_dev_grid(), "attack")
    res["per_dataset"]["dev_grid"] = {"d_full": dev_df, "frontier": _rho_with_ci(dev_df, Ydev)}

    if ng is not None:
        ng_df = _per_attack(ng)
        Yng = _Y_map(a1._Y_nextgen(), "attack")
        res["per_dataset"]["nextgen"] = {"d_full": ng_df, "frontier": _rho_with_ci(ng_df, Yng)}

    # pooled (per-dataset z-standardised D_full to avoid the scale-pooling trap)
    pooled_x, pooled_y = [], []
    for dsname, dd in res["per_dataset"].items():
        Y = Ydev if dsname == "dev_grid" else Yng
        xs = np.array([v["D_full_grouped"] for a, v in dd["d_full"].items() if a in Y])
        ys = np.array([Y[a] for a in dd["d_full"] if a in Y])
        if len(xs) >= 2:
            pooled_x += list((xs - xs.mean()) / (xs.std() + 1e-9))
            pooled_y += list(ys)
    pooled_x, pooled_y = np.array(pooled_x), np.array(pooled_y)
    prho = float(spearmanr(pooled_x, pooled_y)[0]) if len(pooled_x) >= 3 else None
    res["pooled_within_standardized"] = {"n": int(len(pooled_x)), "spearman_rho": prho,
                                          "note": "D_full z-standardised WITHIN each dataset before pooling (avoids scale-pooling)."}

    out = RESULTS / "frontier" / "frontier_grouped.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))

    print("paper GROUPED + CROSS-FIT")
    for dsname, dd in res["per_dataset"].items():
        fr = dd["frontier"]
        print(f"  {dsname:8s}: grouped Spearman(D_full, Y) = {fr['spearman_rho']:+.2f} "
              f"CI[{fr['rho_ci'][0]:+.2f},{fr['rho_ci'][1]:+.2f}] (n={fr['n_attacks']} attacks)")
    print(f"  pooled (within-standardised, n={res['pooled_within_standardized']['n']}): "
          f"rho = {prho:+.2f}  (vs ungrouped headline 0.82)")


if __name__ == "__main__":
    main()
