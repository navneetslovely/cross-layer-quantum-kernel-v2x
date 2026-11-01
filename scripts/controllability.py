"""Separability ordering: model-free divergence D_full(a) vs realised detectability Y(a) (Fig. 4).

For each attack family a, D_S(a) is a MODEL-FREE divergence (MMD with RBF/median-heuristic +
energy distance) between attack-`a` windows and benign windows restricted to feature subset S
(full = all features; phys = non-controllable features from controllability_tags.py). No
detector enters D. Y(a) is the realised best-over-models detectability (dev_grid: held-out-attack
protocol of configs/experiments/crossattack.yaml; NextGen: per-attack results). The script also
reports the descriptive ratio C(a) = 1 - D_phys(a) / D_full(a) (C=1 by convention when
D_full ~ 0) and the mimicry erosion C_mim(a) = 1 - recall_after/recall_before (results/evasion).

Screens: Spearman/Kendall(D, Y) per dataset and pooled, and leave-one-attack-out (LOAO) rank
extrapolation. These are ungrouped screens; the transmitter-grouped, cross-fitted estimates
reported in the paper are computed by scripts/studies/divergence.py, correlations.py and
veremi_separability.py.

Outputs: results/frontier/{controllability.csv, controllability_validation.json,
controllability_scatter.{png,pdf}}.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import kendalltau, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _study_lib import REPO, RESULTS, now_iso, save_table, write_metadata  # noqa: E402

N_PER_CLASS = 200
N_SEEDS = 3
EPS_DFULL = 1e-4  # below this, attack is undetectable everywhere -> C := 1


# ---------------- model-free divergences ----------------
def _mmd2_rbf(A: np.ndarray, B: np.ndarray) -> float:
    Z = np.vstack([A, B])
    d = cdist(Z, Z)
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    g = 1.0 / (2.0 * med * med + 1e-12)
    Kaa = np.exp(-g * cdist(A, A) ** 2)
    Kbb = np.exp(-g * cdist(B, B) ** 2)
    Kab = np.exp(-g * cdist(A, B) ** 2)
    return float(Kaa.mean() + Kbb.mean() - 2.0 * Kab.mean())


def _energy(A: np.ndarray, B: np.ndarray) -> float:
    return float(2.0 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def _c_div(attack: np.ndarray, benign: np.ndarray, phys_idx, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    a = attack[rng.choice(len(attack), min(N_PER_CLASS, len(attack)), replace=False)]
    b = benign[rng.choice(len(benign), min(N_PER_CLASS, len(benign)), replace=False)]
    out = {}
    for metric, fn in (("mmd", _mmd2_rbf), ("energy", _energy)):
        d_full = fn(a, b)
        d_phys = fn(a[:, phys_idx], b[:, phys_idx])
        if d_full < EPS_DFULL:
            c = 1.0
        else:
            c = float(np.clip(1.0 - d_phys / d_full, 0.0, 1.0))
        out[metric] = {"C": c, "D_full": d_full, "D_phys": d_phys}
    return out


def _standardize(X: np.ndarray) -> np.ndarray:
    lo = np.quantile(X, 0.005, axis=0)
    hi = np.quantile(X, 0.995, axis=0)
    Xc = np.clip(X, lo, hi)
    mu = Xc.mean(axis=0)
    sd = Xc.std(axis=0) + 1e-9
    return (Xc - mu) / sd


# ---------------- dataset loaders (raw window features) ----------------
def _load_dev_grid():
    from qkv2x.config import load_yaml
    from qkv2x.experiments.common import load_experiment_data

    cfg = load_yaml(REPO / "configs" / "features.yaml")
    data = load_experiment_data(REPO / "runs" / "dev_grid", cfg)
    return {
        "X": _standardize(np.asarray(data.X, float)),
        "y": np.asarray(data.y, int),
        "attack": np.asarray(data.attack_name),
        "features": list(data.feature_names),
        "network_mode": "rssi",
    }


def _load_nextgen():
    import glob

    fs = sorted(glob.glob(str(RESULTS / "veremi_nextgen" / "checkpoints" / "assembled.npz")))
    if not fs:
        return None
    z = np.load(fs[0], allow_pickle=True)
    return {
        "X": _standardize(np.asarray(z["X"], float)),
        "y": np.asarray(z["y"], int),
        "attack": np.asarray([str(s) for s in z["attack_name"]]),
        "features": [str(s) for s in z["feature_names"]],
        "network_mode": "timing",
    }


# ---------------- Y(a): realized best-over-models detectability ----------------
def _Y_dev_grid() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "crossattack_dev" / "crossattack_results.csv")
    g = d.groupby("holdout_attack").agg(Y_rocauc=("roc_auc", "max"), Y_prauc=("pr_auc", "max")).reset_index()
    return g.rename(columns={"holdout_attack": "attack"})


def _Y_nextgen() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "veremi_nextgen" / "veremi_nextgen_per_attack.csv")
    if "variant" in d.columns:
        d = d[d.variant == "full"] if (d.variant == "full").any() else d
    g = d.groupby("attack").agg(Y_rocauc=("roc_auc", "max"), Y_prauc=("pr_auc", "max")).reset_index()
    return g


def _C_mim() -> pd.DataFrame:
    d = pd.read_csv(RESULTS / "evasion" / "evasion_results.csv")
    d = d[(d.envelope == "strong") & (d.attack != "ALL")]
    g = d.groupby("attack").agg(rb=("recall_before", "mean"), ra=("recall_after", "mean")).reset_index()
    g["C_mim"] = (1.0 - g["ra"] / g["rb"].replace(0, np.nan)).clip(0, 1)
    return g[["attack", "C_mim"]]


def _controllability_for(ds: dict) -> pd.DataFrame:
    ctab = pd.read_csv(RESULTS / "frontier" / "feature_controllability.csv")
    phys = set(ctab[(ctab.network_mode == ds["network_mode"]) & (ctab.controllability == "phys")].feature)
    phys_idx = [i for i, f in enumerate(ds["features"]) if f in phys]
    X, y, atk = ds["X"], ds["y"], ds["attack"]
    benign = X[y == 0]
    rows = []
    for a in sorted(set(atk[y == 1])):
        if a in ("none", "benign", ""):
            continue
        Xa = X[(atk == a) & (y == 1)]
        if len(Xa) < 30:
            continue
        cs, dfu, dph = [], [], []
        for s in range(N_SEEDS):
            r = _c_div(Xa, benign, phys_idx, seed=1000 + s)
            cs.append(np.mean([r["mmd"]["C"], r["energy"]["C"]]))
            dfu.append(r["energy"]["D_full"])
            dph.append(r["energy"]["D_phys"])
        rows.append({"attack": a, "C_div": float(np.mean(cs)), "C_div_std": float(np.std(cs)),
                     "D_phys": float(np.mean(dph)), "D_full": float(np.mean(dfu)), "n_windows": int(len(Xa))})
    return pd.DataFrame(rows)


def _validate(df: pd.DataFrame, key="C_div", tgt="Y_rocauc") -> dict:
    sub = df.dropna(subset=[key, tgt])
    if len(sub) < 4:
        return {"n": int(len(sub)), "note": "too few attacks"}
    rho, p = spearmanr(sub[key], sub[tgt])
    tau, pt = kendalltau(sub[key], sub[tgt])
    # LOAO: predict held-out Y by a linear fit on the rest, then rank-correlate
    preds, acts = [], []
    X = sub[key].to_numpy()
    Yv = sub[tgt].to_numpy()
    for i in range(len(sub)):
        m = np.ones(len(sub), bool)
        m[i] = False
        if np.ptp(X[m]) < 1e-9:
            continue
        b1, b0 = np.polyfit(X[m], Yv[m], 1)
        preds.append(b0 + b1 * X[i])
        acts.append(Yv[i])
    loao_rho = float(spearmanr(preds, acts)[0]) if len(preds) >= 4 else None
    return {"n": int(len(sub)), "spearman_rho": float(rho), "spearman_p": float(p),
            "kendall_tau": float(tau), "kendall_p": float(pt), "loao_spearman": loao_rho}


def main() -> None:
    t0 = time.time()
    started = now_iso()
    out = RESULTS / "frontier"

    frames = []
    dev = _load_dev_grid()
    cdev = _controllability_for(dev)
    cdev = cdev.merge(_Y_dev_grid(), on="attack", how="left")
    cdev["dataset"] = "dev_grid"
    frames.append(cdev)

    ng = _load_nextgen()
    if ng is not None:
        cng = _controllability_for(ng)
        cng = cng.merge(_Y_nextgen(), on="attack", how="left").merge(_C_mim(), on="attack", how="left")
        cng["dataset"] = "nextgen"
        frames.append(cng)

    df = pd.concat(frames, ignore_index=True)
    if "C_mim" not in df.columns:
        df["C_mim"] = np.nan
    df = df[["attack", "dataset", "C_div", "C_div_std", "C_mim", "D_phys", "D_full", "Y_rocauc", "Y_prauc", "n_windows"]]
    save_table(df, out, "controllability.csv")

    KEYS = ["D_full", "D_phys", "C_div"]
    val = {
        "primary_law": "A model-free divergence D_full (no detector) is positively associated "
                       "with realized best-over-models detectability Y(a) (attack-wise ordering); "
                       "the non-controllable part D_phys carries most of it. The controllability "
                       "ratio C(a) is a descriptive decomposition, not itself predictive.",
        "screens": {
            k: {"pooled": _validate(df, key=k),
                "per_dataset": {ds: _validate(df[df.dataset == ds], key=k) for ds in df.dataset.unique()}}
            for k in KEYS
        },
    }
    mim = df.dropna(subset=["C_div", "C_mim"])
    val["corr_c_div_c_mim"] = float(spearmanr(mim["C_div"], mim["C_mim"])[0]) if len(mim) >= 4 else None
    # asymptote anchor: undetectable / highest-C attacks should sit near Y=0.5
    hi = df.sort_values("C_div", ascending=False).head(3)[["attack", "dataset", "C_div", "Y_rocauc"]]
    lo = df.sort_values("C_div").head(3)[["attack", "dataset", "C_div", "Y_rocauc"]]
    val["asymptote_high_C"] = hi.to_dict("records")
    val["asymptote_low_C"] = lo.to_dict("records")
    val["caveat"] = f"n={df.attack.nunique()} attacks/dataset; exact small-n rank p-values; C(a) is model-free (anti-circular)."
    (out / "controllability_validation.json").write_text(json.dumps(val, indent=2, default=float))

    # figure
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for ds, mk in (("dev_grid", "o"), ("nextgen", "s")):
        s = df[df.dataset == ds].dropna(subset=["D_full", "Y_rocauc"])
        if len(s):
            ax.scatter(s["D_full"], s["Y_rocauc"], marker=mk, s=70, edgecolor="k", label=ds)
            for _, r in s.iterrows():
                ax.annotate(r["attack"][:10], (r["D_full"], r["Y_rocauc"]), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0.5, ls=":", c="gray", lw=1, label="chance (AUC=0.5)")
    ax.set_xlabel("model-free separability $D_{full}(a)$ (energy distance, no detector)")
    ax.set_ylabel("realized best-over-models ROC-AUC  $Y(a)$")
    ax.set_title("Detectability frontier: $C(a)$ predicts realized detectability")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "controllability_scatter.png", dpi=150)
    fig.savefig(out / "controllability_scatter.pdf")
    plt.close(fig)

    write_metadata(
        out, study="controllability_frontier", started_iso=started, t0=t0,
        hyperparams={"n_per_class": N_PER_CLASS, "n_seeds": N_SEEDS, "divergences": ["mmd_rbf_median", "energy"],
                     "eps_dfull": EPS_DFULL},
        splits={ds: {"n_windows": int((df.dataset == ds).sum())} for ds in df.dataset.unique()},
        results_summary=val,
        notes="Model-free controllability index C(a) + frontier-law validation (LOAO, cross-dataset, mimicry triangulation).",
    )
    print("done -> results/frontier/{controllability.csv, controllability_validation.json, scatter}")
    for k in KEYS:
        pv = val["screens"][k]["pooled"]
        print(f"  pooled Spearman({k},Y)={pv.get('spearman_rho')} (p={pv.get('spearman_p')}), LOAO={pv.get('loao_spearman')}")
    print(f"  corr(C_div,C_mim)={val['corr_c_div_c_mim']}")
    print(df[["attack", "dataset", "C_div", "D_full", "Y_rocauc"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
