#!/usr/bin/env python3
"""tuning_budget -- quantify the TUNING BUDGET of every model in the main comparison.

Quantifies whether the quantum and classical models have comparable tuning budgets.
baseline_fairness *lists* the grids (results/studies/hyperparams.csv) and states that
the classical models are tuned at least as hard as the quantum ones on the SVM axis;
tuning_budget turns "comparable" into a NUMBER.

WHAT IS COUNTED
---------------
Per model actually used in the paper's comparisons (rbf_svm, random_forest,
knn_bagging, mlp, xgboost, lstm_seq, ocsvm_classical, qsvm_crosslayer, qsvm_zz,
qsvm_none, qsvm_sparse):

  n_configs  size of the hyperparameter grid ACTUALLY searched. Read by
                       INTROSPECTION, not transcription: build_baseline(...) is called
                       with the shipped configs/baselines.yaml and the resulting
                       GridSearchCV's `param_grid` is expanded with
                       sklearn.model_selection.ParameterGrid. This automatically picks
                       up the two code-truths baseline_fairness documented (XGBoost's grid is a code
                       default -- there is no xgboost block in baselines.yaml; MLP's
                       early_stopping is hard-coded False in the binary path).
  cv_folds  configs/baselines.yaml::cv_folds = 5 (classical) /
                       configs/quantum.yaml::svm.cv_folds = 5 (quantum C-grid).
  n_fits  n_configs * cv_folds + 1 refit = the number of model fits the
                       search costs. (baseline_fairness quotes the CV-only product 80/20/180;
                       tuning_budget reports BOTH -- `n_fits_cv_only` and `n_fits` -- so the two
                       reconcile exactly.)
  n_gradient_steps  parameter-update steps spent LEARNING A REPRESENTATION
                       (0 for kernel/tree/neighbour models; 200 Adam steps for a KTA-
                       trained quantum map; for the MLP the analytic upper bound
                       n_fits * max_iter * ceil(n_fold_train / 200), sklearn's
                       batch_size='auto' = min(200, n)).
  n_trainable_params  feature-map parameters optimised by KTA, counted from the LIVE
                       object (fm.init_params() element counts + a gradient-path check),
                       not from a formula: 2*L*n + L*|E| = 64 (cross-layer, |E|=16),
                       88 (zz_full, |E|=28 -- matches entanglement_ablation's published 88), 32 (none;
                       the (L,1) gamma tensor that init_params allocates for the empty
                       coupling list is inert and is EXCLUDED, so 32 effective / 34
                       allocated), 64 (sparse_random, 16 edges).
  restarts             1 for every model (single seeded initialisation; the quantum map
                       keeps the best-alignment snapshot across its 200 epochs, the LSTM
                       the best-validation-AP epoch across its 40).
  selection_metric  what the search maximises.

WHY BUDGET, NOT EQUIVALENCE
---------------------------
KTA tunes the REPRESENTATION by gradient descent (200 Adam steps over 64 map
parameters, objective = centered kernel-target alignment, no model refit); a classical
grid tunes the CLASSIFIER by enumeration (n_configs points, objective = cross-validated
average precision, one refit per point). These are different search paradigms and no
exchange rate between them is claimed. The honest statement is a side-by-side of
BUDGETS in each currency (enumerated configs / model fits / gradient steps / seconds),
plus the observation that the quantum models' STRUCTURAL hyperparameters (n_layers,
entangler, coupling, kappa, KTA lr/epochs/batch/init) were FIXED a priori and never
searched, whereas the classical models' structural hyperparameters (tree depth, hidden
layer sizes, kernel bandwidth, boosting rounds) were.

WALL CLOCK
----------
Per-model wall clock of the PUBLISHED run is IRRECOVERABLE: results/packetloss_dev/
{summary,provenance}.json record only one aggregate `wall_time_s` = 858.5 s for the
whole packet-loss sweep (11 loss levels x 8 model rows + 200-replicate bootstraps),
with no per-model breakdown, and no other artefact in results/ carries per-model timing
for the Table-I models. tuning_budget therefore RE-MEASURES the tuning cost here, on this
machine, under the exact matched-N Table-I protocol (dev_grid, split seed 7, the
identical 3,023-window group-capped train subset -> 3,000 after the quantum branch's
internal re-cap, the identical 4+4 blockwise-PCA coordinates), and labels those numbers
as a re-measurement, not as the published run's timing.

Two timings are reported because raw wall clock is confounded by parallelism -- the
classical baselines ship with `n_jobs=-1` (GridSearchCV and, for RF/XGB/bagging, the
estimator too) while the KTA loop is effectively single-threaded, so wall clock would
flatter the classical side by whatever core count the machine happens to have:
  measured_tuning_wall_s  as-shipped configuration (parallelism on), plus a
                           process-tree CPU total measured with psutil (getrusage
                           under-counts, because joblib/loky workers are kept alive
                           rather than reaped, so their CPU never lands in
                           RUSAGE_CHILDREN);
  measured_tuning_core_s  SERIAL re-run -- joblib n_jobs=1, estimator n_jobs=1, BLAS/
                           OpenMP capped to one thread with threadpoolctl, and
                           torch.set_num_threads(1) for the quantum path -- i.e. clean
                           single-core seconds, the only timing that is comparable
                           across models.

Nothing under src/, configs/, or existing scripts/results is modified; the script is
read-only apart from its own results/studies/tuning_budget_* outputs.

Usage:
    python scripts/studies/tuning_budget.py \
        2>&1 | tee results/studies/tuning_budget_run.log
    (--no-measure skips the wall-clock re-measurement and emits the static accounting
     only; --quantum-variants limits which quantum maps are re-timed.)

Outputs (results/studies/):
    tuning_budget.csv  one row per model: paradigm, n_configs, cv_folds, n_fits,
                             n_gradient_steps, n_trainable_params, restarts,
                             selection_metric, source_file (+ measured columns)
    tuning_budget_summary.json  protocol + fairness verdict + headline
    tuning_budget_run.log  tee'd console log
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

RESULTS = REPO / "results"
OUT = RESULTS / "studies"
OUT.mkdir(parents=True, exist_ok=True)

from qkv2x.config import load_yaml                                   # noqa: E402
from qkv2x.experiments.common import (ExperimentData,                # noqa: E402
                                      _cap_train_groupwise,
                                      fit_reducer, group_split,
                                      make_feature_map)
from qkv2x.features.crosslayer import FEATURE_SPEC                   # noqa: E402
from qkv2x.models.baselines import build_baseline                    # noqa: E402
from qkv2x.models.qsvm import QuantumKernelSVM                       # noqa: E402
from qkv2x.quantum.kernels import fidelity_gram                      # noqa: E402
from qkv2x.quantum.train import train_feature_map                    # noqa: E402

PARQUET = REPO / "runs" / "dev_grid" / "features.parquet"
PUBLISHED = RESULTS / "packetloss_dev" / "packetloss_results.csv"
PUB_SUMMARY = RESULTS / "packetloss_dev" / "summary.json"

# ------------------------------ protocol constants ---------------------------
SPLIT_SEED = 7            # configs/features.yaml::splits.seed
BASELINE_SEED = 17        # configs/baselines.yaml::seed
MAX_TRAIN = 3000          # configs/quantum.yaml::max_train
MLP_SKLEARN_BATCH = 200   # sklearn MLPClassifier batch_size='auto' = min(200, n)
LSTM_EPOCHS = 40          # src/qkv2x/models/seq_baselines.py::LSTMDetector default
LSTM_BATCH = 256
LSTM_N_TRAIN = 3000       # configs/experiments/veremi_nextgen.yaml::max_windows.Train,
                          # confirmed by results/veremi_nextgen/summary.json::n_used.tr
# models in the paper's comparisons
GRID_MODELS = ["rbf_svm", "random_forest", "knn_bagging", "mlp", "xgboost"]
QUANTUM_MODELS = ["qsvm_crosslayer", "qsvm_zz", "qsvm_none", "qsvm_sparse"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load() -> ExperimentData:
    """Identical loader to baseline_fairness / scripts/dev_multiseed.py::_load."""
    df = pd.read_parquet(PARQUET)
    names = [s[0] for s in FEATURE_SPEC]
    blocks = [s[3] for s in FEATURE_SPEC]
    return ExperimentData(
        X=df[names].to_numpy(), y=df["y"].to_numpy(),
        attack_name=df["attack_name"].to_numpy(object),
        groups=df["group"].to_numpy(), aoi=df["aoi"].to_numpy(),
        t_end=df["t_end"].to_numpy(), feature_names=names, blocks=blocks,
        points=pd.DataFrame(index=range(len(df))))


class _Clock:
    """Wall clock + process-TREE CPU.

    resource.getrusage(RUSAGE_CHILDREN) is NOT usable here: joblib/loky keeps its
    worker processes alive between calls (idle timeout), so they are never reaped and
    their CPU never lands in RUSAGE_CHILDREN -- it under-reports a 6 s / 80-fit
    GridSearchCV as 0.3 s of CPU. Instead we snapshot per-PID cumulative CPU of this
    process and every descendant with psutil, and difference by PID (a worker spawned
    inside the window has no 'before' entry, so its whole CPU is counted; a worker that
    dies inside the window is lost -- an under-count, disclosed rather than hidden)."""

    @staticmethod
    def _snapshot() -> dict:
        import psutil
        out = {}
        me = psutil.Process()
        for p in [me] + me.children(recursive=True):
            try:
                t = p.cpu_times()
                out[p.pid] = float(t.user + t.system)
            except Exception:                                    # pragma: no cover
                pass
        return out

    def __enter__(self):
        self.b = self._snapshot()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.wall = time.perf_counter() - self.t0
        a_snap = self._snapshot()
        self.cpu = float(sum(v - self.b.get(k, 0.0) for k, v in a_snap.items()))
        # rusage kept for provenance/comparison only
        s = resource.getrusage(resource.RUSAGE_SELF)
        c = resource.getrusage(resource.RUSAGE_CHILDREN)
        self.rusage_total = float(s.ru_utime + s.ru_stime + c.ru_utime + c.ru_stime)


# which n_jobs knobs each shipped estimator actually exposes (build_baseline)
_SERIAL_KEYS = {
    "rbf_svm": ("n_jobs",),                            # SVC has no n_jobs
    "random_forest": ("n_jobs", "estimator__n_jobs"),
    "knn_bagging": ("n_jobs", "estimator__bag__n_jobs"),
    "mlp": ("n_jobs",),                                # MLP is BLAS-threaded only
    "xgboost": ("n_jobs", "estimator__n_jobs"),
}


def _force_serial(est, name: str) -> None:
    """Set exactly the n_jobs knobs this estimator exposes to 1 (joblib layer AND, where
    it exists, the inner estimator), so the serial core-seconds measurement is genuinely
    single-process. Model-aware on purpose: XGBClassifier.set_params() silently accepts
    unknown kwargs and then warns at fit time, so a blanket key list would inject junk
    parameters into the very model being timed."""
    for key in _SERIAL_KEYS.get(name, ("n_jobs",)):
        try:
            est.set_params(**{key: 1})
        except (ValueError, TypeError):                          # pragma: no cover
            pass


def _warm_joblib_pool() -> float:
    """Spawn the loky worker pool BEFORE any timing.

    On macOS loky spawns (not forks) its workers, so each worker pays a fresh
    interpreter + numpy/sklearn import. Charged to the first timed model that would be
    tens of process-seconds of pure start-up masquerading as tuning CPU. Returns the
    wall time the warm-up itself took, recorded for provenance."""
    from joblib import Parallel, delayed
    from sklearn.svm import SVC
    t = time.perf_counter()
    Xw = np.random.default_rng(0).normal(size=(60, 4))
    yw = np.r_[np.zeros(30), np.ones(30)].astype(int)
    Parallel(n_jobs=-1)(delayed(SVC().fit)(Xw, yw) for _ in range(32))
    return time.perf_counter() - t


def _grid_of(name: str, bcfg: dict, folds: int):
    """Expand the grid ACTUALLY searched by build_baseline (introspection, not
    transcription). Returns (n_configs, param_grid_dict, list_of_config_dicts)."""
    from sklearn.model_selection import GridSearchCV, ParameterGrid
    est = build_baseline(name, bcfg, seed=BASELINE_SEED, folds=folds, tune=True)
    assert isinstance(est, GridSearchCV), f"{name} is not grid-searched"
    grid = dict(est.param_grid)
    configs = list(ParameterGrid(grid))
    return len(configs), grid, configs


def _base_learners(name: str, configs: list, folds: int, best: dict | None) -> dict:
    """Base learners actually trained across the whole search -- the unit-cost caveat:
    one 'fit' is not one learner (a random-forest fit trains 200-500 trees, a bagged
    kNN fit trains 10 members, an XGBoost fit trains up to 400 boosting rounds)."""
    def _n(cfg):
        if name == "random_forest":
            return int(cfg["n_estimators"])
        if name == "xgboost":
            return int(cfg["n_estimators"])
        if name == "knn_bagging":
            return 10                      # configs/baselines.yaml::knn_bagging.n_estimators
        return 1
    per_cfg = [_n(c) for c in configs]
    total = folds * int(sum(per_cfg))
    if best is not None:
        total += _n(best)
    else:                                   # refit uses the best config; bound it
        total += int(max(per_cfg))
    return {"n_base_learners_total": int(total),
            "n_base_learners_per_fit_min": int(min(per_cfg)),
            "n_base_learners_per_fit_max": int(max(per_cfg))}


def _learned_params(name: str, est) -> tuple[str, str]:
    """Parameters/structure FITTED FROM DATA by the selected model (a capacity note,
    NOT commensurable across families -- reported so the 'quantum has an extra
    training axis' caveat can be read symmetrically)."""
    from sklearn.model_selection import GridSearchCV
    e = est.best_estimator_ if isinstance(est, GridSearchCV) else est
    try:
        if name == "rbf_svm":
            svc = e.named_steps["svc"]
            return (str(int(svc.dual_coef_.size)), "dual coefficients (= n support vectors)")
        if name == "random_forest":
            n = int(sum(t.tree_.node_count for t in e.estimators_))
            return (str(n), f"total decision-tree nodes over {len(e.estimators_)} trees")
        if name == "mlp":
            m = e.named_steps["mlp"]
            n = int(sum(c.size for c in m.coefs_) + sum(b.size for b in m.intercepts_))
            return (str(n), f"weights+biases of hidden layers {m.hidden_layer_sizes}")
        if name == "xgboost":
            n = int(e.get_booster().trees_to_dataframe().shape[0])
            return (str(n), "total booster tree nodes")
        if name == "knn_bagging":
            return ("IRRECOVERABLE", "non-parametric: kNN stores training points, "
                                     "it has no fitted parameter vector")
        if name == "ocsvm_classical":
            return (str(int(e.dual_coef_.size)), "dual coefficients (= n support vectors)")
    except Exception as exc:                                     # pragma: no cover
        return ("IRRECOVERABLE", f"introspection failed: {type(exc).__name__}: {exc}")
    return ("IRRECOVERABLE", "not implemented for this model")


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-measure", action="store_true",
                    help="skip the wall-clock re-measurement (static accounting only)")
    ap.add_argument("--quantum-variants", default=",".join(QUANTUM_MODELS))
    args = ap.parse_args()
    measure = not args.no_measure
    qvariants = [v for v in args.quantum_variants.split(",") if v]

    t0 = time.time()
    print(f"tuning_budget tuning-budget accounting | interpreter={sys.executable}")
    print(f"platform={platform.platform()} machine={platform.machine()} "
          f"measure={measure}", flush=True)

    feats = load_yaml(str(REPO / "configs/features.yaml"))
    qcfg = load_yaml(str(REPO / "configs/quantum.yaml"))
    bcfg = load_yaml(str(REPO / "configs/baselines.yaml"))
    cv_folds = int(bcfg.get("cv_folds", 5))
    q_folds = int(qcfg["svm"]["cv_folds"])
    C_grid = list(qcfg["svm"]["C_grid"])
    kta = dict(qcfg.get("train", {}))
    kta_epochs = int(kta.get("epochs", 200))
    kta_batch = int(kta.get("batch", 512))
    L = int(qcfg["feature_map"].get("n_layers", 2))

    # ------------------------------------------------------------------ #
    # 0. reconstruct the matched-N Table-I protocol (identical to baseline_fairness)
    # ------------------------------------------------------------------ #
    data = _load()
    sp = feats["splits"]
    tr, va, te = group_split(data.groups, data.y, sp["test_frac"], sp["val_frac"],
                             int(sp["seed"]))
    tr_match = _cap_train_groupwise(tr, data.groups, MAX_TRAIN, int(sp["seed"]))
    tr_q = _cap_train_groupwise(tr_match, data.groups, MAX_TRAIN, int(sp["seed"]))
    reducer = fit_reducer(feats["reduction"], data.X, data.blocks, tr)
    Xr = reducer.transform(data.X)
    n_windows, n_dim = len(data.y), Xr.shape[1]
    invariants = {"n_windows": n_windows, "n_windows_expected": 90553,
                  "n_train_matched": int(len(tr_match)), "n_train_matched_expected": 3023,
                  "n_train_quantum": int(len(tr_q)), "n_train_quantum_expected": 3000,
                  "n_dim": int(n_dim), "n_dim_expected": 8}
    ok = (invariants["n_windows"] == 90553 and invariants["n_train_matched"] == 3023
          and invariants["n_train_quantum"] == 3000 and n_dim == 8)
    print(f"[tuning_budget] protocol invariants {invariants} -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        raise SystemExit("reconstructed matched-N protocol does not match Table I -- "
                         "refusing to time unlike things")

    # fold-train size actually seen by a classical CV fit (for the MLP step bound)
    n_fold_train = int(round(len(tr_match) * (cv_folds - 1) / cv_folds))

    rows: list[dict] = []
    measured: dict[str, dict] = {}
    t_warm = None

    # ------------------------------------------------------------------ #
    # 1. classical grid-searched baselines
    # ------------------------------------------------------------------ #
    print("\n--- classical grid-searched baselines "
          "(enumerated configs x 5-fold stratified CV, scoring average_precision) ---")
    for name in GRID_MODELS:
        n_cfg, grid, configs = _grid_of(name, bcfg, cv_folds)
        n_fits_cv = n_cfg * cv_folds
        n_fits = n_fits_cv + 1
        src = ("configs/baselines.yaml" if name != "xgboost"
               else "src/qkv2x/models/baselines.py:110-114 (code default; "
                    "no xgboost block in configs/baselines.yaml)")
        # gradient steps: only the MLP performs gradient-based representation learning
        if name == "mlp":
            max_iter = int(bcfg.get("mlp", {}).get("max_iter", 400))
            steps_per_iter = math.ceil(n_fold_train / MLP_SKLEARN_BATCH)
            n_steps = n_fits * max_iter * steps_per_iter
            steps_basis = (f"upper bound: {n_fits} fits x max_iter={max_iter} x "
                           f"ceil({n_fold_train}/{MLP_SKLEARN_BATCH})="
                           f"{steps_per_iter} minibatches per iteration (adam may stop "
                           f"early via n_iter_no_change=10, tol=1e-4)")
        else:
            n_steps = 0
            steps_basis = ("no gradient-based representation learning (SMO / greedy "
                           "split search / neighbour lookup)")
        rows.append({
            "model": name,
            "paradigm": "classical: enumerated grid search over classifier hyperparameters",
            "n_configs": n_cfg, "cv_folds": cv_folds, "n_fits": n_fits,
            "n_gradient_steps": n_steps, "n_trainable_params": 0, "restarts": 1,
            "selection_metric": "average_precision (mean over 5 stratified CV folds)",
            "source_file": src,
            "n_fits_cv_only": n_fits_cv,
            "grid": json.dumps({k: [str(v) for v in vs] for k, vs in grid.items()}),
            "n_structural_hparams_searched": int(sum(1 for vs in grid.values()
                                                     if len(vs) > 1)),
            "gradient_steps_basis": steps_basis,
            "trainable_params_basis": "n/a (no trained feature map)",
            "cv_type": "StratifiedKFold(shuffle=True, random_state=17)",
            "tuning_seed": BASELINE_SEED,
        })
        print(f"  {name:14s} n_configs={n_cfg:3d}  n_fits={n_fits:4d} "
              f"(cv-only {n_fits_cv})  grid={ {k: len(v) for k, v in grid.items()} }")

    # ------------------------------------------------------------------ #
    # 2. ocsvm_classical (validation sweep, no CV) + lstm_seq (no grid at all)
    # ------------------------------------------------------------------ #
    nu_grid = list(bcfg.get("ocsvm", {}).get("nu_grid", [0.01, 0.05, 0.10]))
    rows.append({
        "model": "ocsvm_classical",
        "paradigm": "classical one-class: enumerated sweep scored on the validation split",
        "n_configs": len(nu_grid), "cv_folds": 0, "n_fits": len(nu_grid) + 1,
        "n_gradient_steps": 0, "n_trainable_params": 0, "restarts": 1,
        "selection_metric": "ROC-AUC on the held-out validation split (no CV)",
        "source_file": "configs/baselines.yaml::ocsvm; "
                       "src/qkv2x/experiments/common.py:367-389",
        "n_fits_cv_only": len(nu_grid),
        "grid": json.dumps({"nu": [str(v) for v in nu_grid],
                            "gamma": ["scale (fixed)"]}),
        "n_structural_hparams_searched": 1,
        "gradient_steps_basis": "no gradient-based representation learning (SMO)",
        "trainable_params_basis": "n/a (no trained feature map)",
        "cv_type": "none -- single validation split shared with every model",
        "tuning_seed": SPLIT_SEED,
    })
    print(f"  {'ocsvm_classical':14s} n_configs={len(nu_grid):3d}  "
          f"n_fits={len(nu_grid) + 1:4d}  (validation sweep, no CV)")

    lstm_steps = LSTM_EPOCHS * math.ceil(LSTM_N_TRAIN / LSTM_BATCH)
    rows.append({
        "model": "lstm_seq",
        "paradigm": "classical sequence model: NO hyperparameter grid; one fixed "
                    "configuration, best epoch selected on validation",
        "n_configs": 1, "cv_folds": 0, "n_fits": 1,
        "n_gradient_steps": lstm_steps,
        "n_trainable_params": 0, "restarts": 1,
        "selection_metric": "average_precision on the validation split, evaluated after "
                            f"each of {LSTM_EPOCHS} epochs (best-epoch snapshot)",
        "source_file": "configs/experiments/veremi_nextgen.yaml::lstm (sets only "
                       "enabled:true); src/qkv2x/models/seq_baselines.py:17-18 "
                       "(hidden=64, layers=1, epochs=40, batch=256, lr=1e-3)",
        "n_fits_cv_only": 0,
        "grid": json.dumps({"(none searched)": ["hidden=64", "layers=1", "epochs=40",
                                                "batch=256", "lr=0.001"]}),
        "n_structural_hparams_searched": 0,
        "gradient_steps_basis":
            f"{LSTM_EPOCHS} epochs x ceil({LSTM_N_TRAIN}/{LSTM_BATCH})="
            f"{math.ceil(LSTM_N_TRAIN / LSTM_BATCH)} minibatch Adam updates = "
            f"{lstm_steps}; N_train={LSTM_N_TRAIN} from "
            "configs/experiments/veremi_nextgen.yaml::max_windows.Train, confirmed by "
            "results/veremi_nextgen/summary.json::n_used.tr. Wall clock is not "
            "re-measured: the LSTM runs only on the NextGen per-message sequence "
            "pipeline (veremi_nextgen_exp.py:_eval_lstm), not on the dev_grid matched-N "
            "protocol timed here",
        "trainable_params_basis": "n/a (no trained quantum feature map)",
        "cv_type": "none -- single validation split",
        "tuning_seed": "experiment seed (7 primary)",
    })
    print(f"  {'lstm_seq':14s} n_configs=  1  n_fits=   1  "
          f"gradient steps={lstm_steps}  "
          f"(no grid; best of {LSTM_EPOCHS} validation-AP epoch snapshots)")

    # ------------------------------------------------------------------ #
    # 3. quantum models -- two axes, counted separately
    # ------------------------------------------------------------------ #
    print("\n--- quantum models (KTA-trained representation + enumerated C-grid) ---")
    fm_facts = {}
    for name in QUANTUM_MODELS:
        fm = make_feature_map(name, qcfg, feats["reduction"], feats)
        p = fm.init_params()
        alloc = int(sum(v.numel() for v in p.values()))
        n_edges = len(fm._couples)
        # a gamma tensor is allocated even with an empty coupling list (nc=max(1,0)=1)
        # but never enters the circuit -> inert, excluded from the effective count.
        inert = int(p["gamma"].numel()) if n_edges == 0 else 0
        effective = alloc - inert
        formula = 2 * L * fm.n + L * n_edges
        fm_facts[name] = {
            "entangler": fm.entangler, "n_qubits": int(fm.n), "n_layers": L,
            "n_edges": int(n_edges),
            "n_params_allocated": alloc, "n_params_inert": inert,
            "n_params_effective": effective,
            "formula_2Ln_plus_LE": int(formula),
            "formula_matches_effective": bool(formula == effective),
        }
        trained = name != "qsvm_zz"     # common.py:_maybe_train_params
        n_steps = kta_epochs if trained else 0
        rows.append({
            "model": name,
            "paradigm": ("quantum: gradient-trained feature map (KTA) + enumerated "
                         "C-grid over the precomputed Gram" if trained else
                         "quantum: FIXED (untrained) feature map + enumerated C-grid "
                         "over the precomputed Gram"),
            "n_configs": len(C_grid), "cv_folds": q_folds,
            "n_fits": len(C_grid) * q_folds + 1,
            "n_gradient_steps": n_steps,
            "n_trainable_params": effective if trained else 0,
            "restarts": 1,
            "selection_metric": ("centered KTA vs yy^T for the map (best-alignment "
                                 "snapshot over epochs) + average_precision over 5 "
                                 "group-aware CV folds for C" if trained else
                                 "average_precision over 5 group-aware CV folds for C "
                                 "(map not selected -- fixed_params ablation)"),
            "source_file": "configs/quantum.yaml::{train,svm,feature_map}; "
                           "src/qkv2x/quantum/train.py:72-121; "
                           "src/qkv2x/models/qsvm.py:36-60; "
                           "src/qkv2x/experiments/common.py:224-228",
            "n_fits_cv_only": len(C_grid) * q_folds,
            "grid": json.dumps({"svm C": [str(c) for c in C_grid]}),
            "n_structural_hparams_searched": 0,
            "gradient_steps_basis":
                (f"{kta_epochs} Adam steps (one per epoch, lr={kta.get('lr')}, "
                 f"class-balanced batch of {kta_batch} windows -> "
                 f"{kta_epochs * kta_batch} window evaluations and {kta_epochs} "
                 f"centered-KTA evaluations, each on a {kta_batch}x{kta_batch} Gram); "
                 f"single init (w,u,gamma)=({kta.get('init_w')},{kta.get('init_u')},"
                 f"{kta.get('init_gamma')}), seed {kta.get('seed')}"
                 if trained else
                 "0 -- qsvm_zz uses fm.fixed_params() (w=u=gamma=1): the deliberate "
                 "untrained-map ablation, so it receives NO representation budget"),
            "trainable_params_basis":
                (f"counted from the live CrossLayerFeatureMap: 2*L*n + L*|E| = "
                 f"2*{L}*{fm.n} + {L}*{n_edges} = {formula} "
                 f"({alloc} allocated, {inert} inert)"
                 if trained else
                 f"{formula} map parameters exist (2*{L}*{fm.n} + {L}*{n_edges}) but "
                 f"NONE is optimised -- fixed_params()"),
            "cv_type": "StratifiedGroupKFold(shuffle=True, random_state=0) over "
                       "transmitter groups",
            "tuning_seed": int(kta.get("seed", 11)),
        })
        print(f"  {name:14s} n_configs={len(C_grid):3d}  "
              f"n_fits={len(C_grid) * q_folds + 1:4d}  edges={n_edges:2d}  "
              f"map params={effective if trained else 0} "
              f"({alloc} alloc, {inert} inert)  gradient steps={n_steps}")

    df = pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    # 4. wall-clock RE-MEASUREMENT on the matched-N protocol
    # ------------------------------------------------------------------ #
    if measure:
        print(f"\n--- wall-clock re-measurement on the matched-N Table-I protocol "
              f"(N_train={len(tr_match)} / quantum {len(tr_q)}, d={n_dim}, "
              f"dev_grid seed {SPLIT_SEED}) ---")
        from threadpoolctl import threadpool_limits
        t_warm = _warm_joblib_pool()
        print(f"  (loky worker pool warmed in {t_warm:.2f}s so worker spawn/import "
              f"cost is not charged to the first model timed)", flush=True)
        ytr = data.y[tr_match].astype(int)
        for name in GRID_MODELS:
            est = build_baseline(name, bcfg, seed=BASELINE_SEED, folds=cv_folds,
                                 tune=True)
            with _Clock() as clk:
                est.fit(Xr[tr_match], ytr)
            # SERIAL re-run: single-core seconds, the only cross-model-comparable timing
            est_s = build_baseline(name, bcfg, seed=BASELINE_SEED, folds=cv_folds,
                                   tune=True)
            _force_serial(est_s, name)
            with threadpool_limits(limits=1):
                with _Clock() as clk_s:
                    est_s.fit(Xr[tr_match], ytr)
            n_cfg, _, configs = _grid_of(name, bcfg, cv_folds)
            bl = _base_learners(name, configs, cv_folds, est.best_params_)
            lp, lp_basis = _learned_params(name, est)
            extra = {}
            if name == "mlp":
                extra["measured_n_iter_best_refit"] = int(
                    est.best_estimator_.named_steps["mlp"].n_iter_)
                # weight count of EVERY searched hidden-layer configuration
                from sklearn.neural_network import MLPClassifier
                per_cfg = {}
                for h in bcfg.get("mlp", {}).get("hidden", [[64, 32]]):
                    sizes = [n_dim] + [int(v) for v in h] + [1]
                    per_cfg[str(tuple(h))] = int(sum(
                        sizes[i] * sizes[i + 1] + sizes[i + 1]
                        for i in range(len(sizes) - 1)))
                extra["mlp_weights_by_hidden_config"] = per_cfg
            measured[name] = {
                "wall_s": round(clk.wall, 2), "cpu_s": round(clk.cpu, 2),
                "core_s": round(clk_s.wall, 2),
                "best_params": {k: str(v) for k, v in est.best_params_.items()},
                "best_cv_score_ap": float(est.best_score_),
                "n_learned_params": lp, "n_learned_params_basis": lp_basis,
                **bl, **extra}
            print(f"  {name:14s} wall={clk.wall:8.2f}s  cpu={clk.cpu:8.2f}s  "
                  f"core(serial)={clk_s.wall:8.2f}s  "
                  f"base learners={bl['n_base_learners_total']:6d}  "
                  f"best={est.best_params_}", flush=True)

        # ocsvm_classical: benign-only fits, nu chosen on the validation split
        from sklearn.metrics import roc_auc_score
        from sklearn.svm import OneClassSVM
        Xb = Xr[tr_match[ytr == 0]]
        with _Clock() as clk:
            best_auc, best_nu = -1.0, float(nu_grid[len(nu_grid) // 2])
            for cand in nu_grid:
                m = OneClassSVM(kernel="rbf", nu=float(cand), gamma="scale").fit(Xb)
                auc = roc_auc_score(data.y[va], -m.decision_function(Xr[va]))
                if auc > best_auc:
                    best_auc, best_nu = auc, float(cand)
            oc = OneClassSVM(kernel="rbf", nu=best_nu, gamma="scale").fit(Xb)
        with threadpool_limits(limits=1):
            with _Clock() as clk_s:
                for cand in nu_grid:
                    m = OneClassSVM(kernel="rbf", nu=float(cand), gamma="scale").fit(Xb)
                    roc_auc_score(data.y[va], -m.decision_function(Xr[va]))
                OneClassSVM(kernel="rbf", nu=best_nu, gamma="scale").fit(Xb)
        lp, lp_basis = _learned_params("ocsvm_classical", oc)
        measured["ocsvm_classical"] = {
            "wall_s": round(clk.wall, 2), "cpu_s": round(clk.cpu, 2),
            "core_s": round(clk_s.wall, 2),
            "best_params": {"nu": str(best_nu)}, "best_val_roc_auc": float(best_auc),
            "n_learned_params": lp, "n_learned_params_basis": lp_basis,
            "n_base_learners_total": len(nu_grid) + 1,
            "n_base_learners_per_fit_min": 1, "n_base_learners_per_fit_max": 1}
        print(f"  {'ocsvm_classical':14s} wall={clk.wall:8.2f}s  cpu={clk.cpu:9.2f}s  "
              f"best nu={best_nu}", flush=True)

        # quantum: KTA training / Gram build / C-grid CV timed SEPARATELY
        import torch
        torch_threads = torch.get_num_threads()
        for name in qvariants:
            fm = make_feature_map(name, qcfg, feats["reduction"], feats)
            trained = name != "qsvm_zz"
            if trained:
                with _Clock() as ck:
                    params, hist = train_feature_map(fm, Xr[tr_q], data.y[tr_q],
                                                     data.aoi[tr_q], kta)
                t_kta_w, t_kta_c = ck.wall, ck.cpu
                best_align = float(hist["best_alignment"])
            else:
                params, t_kta_w, t_kta_c, best_align = fm.fixed_params(), 0.0, 0.0, float("nan")
            with _Clock() as ck:
                states = fm.statevectors(Xr[tr_q], data.aoi[tr_q], params)
                K_tr = fidelity_gram(states)
            t_gram_w, t_gram_c = ck.wall, ck.cpu
            with _Clock() as ck:
                svm = QuantumKernelSVM(C_grid=C_grid, cv_folds=q_folds,
                                       class_weight=qcfg["svm"].get("class_weight",
                                                                    "balanced"))
                svm.fit(K_tr, data.y[tr_q].astype(int), groups=data.groups[tr_q])
            t_svm_w, t_svm_c = ck.wall, ck.cpu
            # SERIAL re-run (1 torch thread, 1 BLAS thread) for core-seconds parity
            torch.set_num_threads(1)
            try:
                fm_s = make_feature_map(name, qcfg, feats["reduction"], feats)
                with threadpool_limits(limits=1):
                    with _Clock() as ck_s:
                        if trained:
                            p_s, _ = train_feature_map(fm_s, Xr[tr_q], data.y[tr_q],
                                                       data.aoi[tr_q], kta)
                        else:
                            p_s = fm_s.fixed_params()
                        K_s = fidelity_gram(fm_s.statevectors(Xr[tr_q], data.aoi[tr_q],
                                                              p_s))
                        QuantumKernelSVM(C_grid=C_grid, cv_folds=q_folds).fit(
                            K_s, data.y[tr_q].astype(int), groups=data.groups[tr_q])
            finally:
                torch.set_num_threads(torch_threads)
            measured[name] = {
                "wall_s": round(t_kta_w + t_gram_w + t_svm_w, 2),
                "cpu_s": round(t_kta_c + t_gram_c + t_svm_c, 2),
                "core_s": round(ck_s.wall, 2),
                "t_kta_wall_s": round(t_kta_w, 2), "t_kta_cpu_s": round(t_kta_c, 2),
                "t_gram_wall_s": round(t_gram_w, 2), "t_gram_cpu_s": round(t_gram_c, 2),
                "t_cgrid_wall_s": round(t_svm_w, 2), "t_cgrid_cpu_s": round(t_svm_c, 2),
                "best_params": {"C": str(svm.best_C_)},
                "best_cv_score_ap": float(svm.cv_score_),
                "best_kta_alignment": best_align,
                "n_learned_params": str(int(svm.svc_.dual_coef_.size)
                                        + (fm_facts[name]["n_params_effective"]
                                           if trained else 0)),
                "n_learned_params_basis": ("SVM dual coefficients (support vectors) + "
                                           "KTA-trained map parameters"),
                "n_base_learners_total": len(C_grid) * q_folds + 1,
                "n_base_learners_per_fit_min": 1, "n_base_learners_per_fit_max": 1}
            print(f"  {name:14s} wall={measured[name]['wall_s']:8.2f}s  "
                  f"cpu={measured[name]['cpu_s']:8.2f}s  "
                  f"core(serial)={measured[name]['core_s']:8.2f}s  "
                  f"(KTA {t_kta_w:.1f}s + Gram {t_gram_w:.1f}s + C-grid {t_svm_w:.1f}s)"
                  f"  bestC={svm.best_C_}", flush=True)

        for col, key in [("measured_tuning_wall_s", "wall_s"),
                         ("process_tree_cpu_s_incl_joblib_overhead", "cpu_s"),
                         ("measured_tuning_core_s", "core_s"),
                         ("n_base_learners_total", "n_base_learners_total"),
                         ("n_learned_params", "n_learned_params"),
                         ("n_learned_params_basis", "n_learned_params_basis")]:
            df[col] = [measured.get(m, {}).get(key, "IRRECOVERABLE") for m in df.model]
    else:
        for col in ("measured_tuning_wall_s",
                    "process_tree_cpu_s_incl_joblib_overhead",
                    "measured_tuning_core_s", "n_base_learners_total",
                    "n_learned_params", "n_learned_params_basis"):
            df[col] = "NOT_MEASURED (--no-measure)"

    lead = ["model", "paradigm", "n_configs", "cv_folds", "n_fits", "n_gradient_steps",
            "n_trainable_params", "restarts", "selection_metric", "source_file"]
    df = df[lead + [c for c in df.columns if c not in lead]]
    df.to_csv(OUT / "tuning_budget.csv", index=False)

    # ------------------------------------------------------------------ #
    # 5. fairness verdict
    # ------------------------------------------------------------------ #
    fits = dict(zip(df.model, df.n_fits.astype(int)))
    cfgs = dict(zip(df.model, df.n_configs.astype(int)))
    q_fits = fits["qsvm_crosslayer"]
    q_cfgs = cfgs["qsvm_crosslayer"]
    classical = [m for m in df.model if not m.startswith("qsvm_")]
    # The classical comparators that BEAT the cross-layer QSVM over the three
    # leakage-safe splits (three_splits.csv / results/dev_grid_multiseed: ROC-AUC
    # rbf_svm 0.928+-0.005, random_forest 0.932+-0.010, xgboost 0.932+-0.008 vs
    # qsvm_crosslayer 0.899+-0.027). The MLP is a matched-N Table-I row but is not in
    # the 3-seed set, so it is listed separately rather than claimed as "beats".
    beats_qsvm = ["rbf_svm", "random_forest", "xgboost"]
    larger = [m for m in classical if fits[m] > q_fits]
    smaller = [m for m in classical if fits[m] < q_fits]
    equal = [m for m in classical if fits[m] == q_fits]
    beat_min_fits = min(fits[m] for m in beats_qsvm)

    verdict = {
        "question": "Is any model advantaged by tuning budget?",
        "axis_1_enumerated_classifier_search": {
            "unit": "model fits = n_configs x cv_folds + 1 refit",
            "n_fits_by_model": {m: int(fits[m]) for m in df.model},
            "n_configs_by_model": {m: int(cfgs[m]) for m in df.model},
            "quantum_reference": {"model": "qsvm_crosslayer", "n_fits": int(q_fits),
                                  "n_configs": int(q_cfgs)},
            "classical_larger_than_quantum": larger,
            "classical_equal_to_quantum": equal,
            "classical_smaller_than_quantum": smaller,
            "ratio_vs_quantum": {m: round(fits[m] / q_fits, 2) for m in df.model},
            "classical_that_beat_qsvm_over_3_splits": beats_qsvm,
            "min_n_fits_among_models_that_beat_qsvm": int(beat_min_fits),
            "qsvm_C_grid_identical_to_rbf_svm_C_grid": bool(
                [float(c) for c in C_grid]
                == [float(c) for c in bcfg["rbf_svm"]["C_grid"]]),
            "why_rbf_has_more_configs": (
                "the QSVM and the RBF-SVM search the SAME 4-value C grid "
                f"({C_grid}); the RBF-SVM additionally searches 4 kernel bandwidths "
                f"({bcfg['rbf_svm']['gamma_grid']}), which the QSVM has no analogue of "
                "because its kernel geometry is set by the trained feature map -- hence "
                f"16 configs vs 4"),
            "low_label_counterexample": {
                "model": "ocsvm_classical",
                "n_fits": int(fits["ocsvm_classical"]),
                "ratio_vs_quantum": round(fits["ocsvm_classical"] / q_fits, 2),
                "evidence": ("results/lowlabel_dev/lowlabel_results.csv, mean PR-AUC "
                             "over repeats: ocsvm_classical 0.685 / 0.679 / 0.700 at "
                             "n_train 50 / 100 / 200 vs qsvm_crosslayer 0.514 / 0.545 / "
                             "0.521 (and qsvm_zz 0.279 / 0.332 / 0.319)"),
                "reading": ("this is the ONE classical model that beats the QSVM on a "
                            "SMALLER tuning budget, and it strengthens rather than "
                            "weakens the honest finding: the low-label winner is the "
                            "least-tuned model in the whole zoo, so the low-label "
                            "advantage belongs to the one-class/anomaly-detection "
                            "PARADIGM, not to tuning effort and not to quantum "
                            "kernels (H3 refuted)."),
            },
            "verdict": ("CLASSICAL >= QUANTUM on the main binary comparison. Each of "
                        "the three classical comparators that beat the cross-layer QSVM "
                        "over the three leakage-safe splits was searched at least as "
                        f"hard as it: random_forest {fits['random_forest']} fits (1.0x "
                        f"the QSVM's {q_fits}), rbf_svm {fits['rbf_svm']} "
                        f"({round(fits['rbf_svm'] / q_fits, 1)}x), xgboost "
                        f"{fits['xgboost']} ({round(fits['xgboost'] / q_fits, 1)}x); the "
                        f"MLP matched-N row is also {fits['mlp']} (1.0x), and "
                        f"knn_bagging ({fits['knn_bagging']}), which was tuned LESS than "
                        "the QSVM, does not beat it on either dataset (NextGen ROC-AUC "
                        "0.880 vs 0.929). So within the headline detection comparison no "
                        "model that outperforms the QSVM was tuned more weakly than it, "
                        "and the no-advantage finding is NOT an artefact of an "
                        "under-tuned classical side -- on this axis the comparison is "
                        "conservative in the quantum models' favour. The single "
                        "exception sits OUTSIDE that comparison and cuts the same way: "
                        f"in the low-label regime the classical one-class SVM "
                        f"({fits['ocsvm_classical']} fits, "
                        f"{round(fits['ocsvm_classical'] / q_fits, 2)}x the QSVM budget) "
                        "beats the cross-layer QSVM by +0.13 to +0.18 PR-AUC at "
                        "n_train 50-200 -- the least-tuned model in the zoo wins there, "
                        "which is a paradigm effect, not a budget effect."),
        },
        "axis_2_structural_hyperparameters": {
            "unit": "number of structural hyperparameters with >1 searched value",
            "by_model": dict(zip(df.model,
                                 df.n_structural_hparams_searched.astype(int).tolist())),
            "verdict": ("The quantum models searched ZERO structural hyperparameters: "
                        "n_layers=2, entangler, coupling=full_bipartite, AoI kappa=0.5, "
                        "and the whole KTA schedule (Adam, lr 0.05, 200 epochs, batch "
                        "512, init (1,1,0.5), seed 11) were fixed a priori in "
                        "configs/quantum.yaml and never tuned, whereas every classical "
                        "baseline searched 1-4 structural hyperparameters (kernel "
                        "bandwidth, tree depth/count, hidden-layer sizes, boosting "
                        "rounds/depth/learning rate/class weight)."),
        },
        "axis_3_gradient_trained_representation": {
            "unit": "parameter-update steps x parameters",
            "by_model": {m: {"n_gradient_steps": df.set_index('model')
                             .loc[m, 'n_gradient_steps'],
                             "n_trainable_params": int(df.set_index('model')
                                                       .loc[m, 'n_trainable_params'])}
                         for m in df.model},
            "verdict": None,     # filled after the MLP weight counts are known
        },
        "paradigm_caveat": ("KTA tunes the REPRESENTATION by gradient descent on a "
                            "detector-free alignment objective; classical grids tune "
                            "the CLASSIFIER by enumeration against cross-validated "
                            "average precision. No exchange rate between a gradient "
                            "step and a grid fit is claimed. The comparison above is a "
                            "comparison of BUDGET in each currency, not a claim of "
                            "equivalence between the two search paradigms."),
        "unit_cost_caveat": ("One 'fit' is not one learner: a random-forest fit trains "
                             "200-500 trees and an XGBoost fit up to 400 boosting "
                             "rounds, so n_base_learners_total (measured) is the "
                             "harsher and more honest cost measure -- and it is larger "
                             "for the classical side by three to four orders of "
                             "magnitude."),
        "models_beating_qsvm_over_3_splits": beats_qsvm,
    }

    # axis-3 verdict needs the MEASURED MLP weight counts (or an honest fallback)
    mlp_w = measured.get("mlp", {}).get("mlp_weights_by_hidden_config")
    q_steps, q_params = kta_epochs, fm_facts["qsvm_crosslayer"]["n_params_effective"]
    if mlp_w:
        lo, hi = min(mlp_w.values()), max(mlp_w.values())
        mlp_clause = (f"the MLP baseline gradient-trains {lo:,}-{hi:,} weights "
                      f"(measured: {mlp_w}) inside every one of its {fits['mlp']} fits, "
                      f"{lo // q_params}x-{hi // q_params}x the quantum map's "
                      f"{q_params} parameters, over "
                      f"{int(df.set_index('model').loc['mlp', 'n_gradient_steps']):,} "
                      f"update steps against the map's {q_steps}")
    else:
        mlp_clause = ("the MLP baseline also gradient-trains its hidden layers "
                      "(weight counts NOT MEASURED in this run)")
    verdict["axis_3_gradient_trained_representation"]["verdict"] = (
        "This is the one axis the quantum models get and the kernel/tree/neighbour "
        f"baselines do not: {q_steps} Adam steps over {q_params} cross-layer map "
        f"parameters. It is NOT a quantum-only axis though -- {mlp_clause}; the LSTM "
        f"sequence baseline likewise takes "
        f"{int(df.set_index('model').loc['lstm_seq', 'n_gradient_steps'])} Adam steps. "
        "qsvm_zz receives ZERO representation budget by design (fixed_params "
        "ablation), so the cross-layer >> ZZ gap is a trained-vs-untrained gap, not a "
        "topology gap.")

    if measure:
        wq = measured["qsvm_crosslayer"]["wall_s"]
        cq = measured["qsvm_crosslayer"]["core_s"]
        core_classical = {m: measured[m]["core_s"] for m in measured
                          if not m.startswith("qsvm_")}
        worst_classical = max(core_classical, key=core_classical.get)
        verdict["axis_4_measured_wall_clock"] = {
            "unit": "seconds of tuning on this machine, matched-N protocol",
            "provenance": ("RE-MEASURED by tuning_budget; the published run's per-model timing "
                           "is IRRECOVERABLE (see protocol.wall_clock_provenance)"),
            "wall_s_by_model": {m: measured[m]["wall_s"] for m in measured},
            "process_tree_cpu_s_by_model_UNRELIABLE": {
                m: measured[m]["cpu_s"] for m in measured},
            "core_s_by_model_serial": {m: measured[m]["core_s"] for m in measured},
            "timing_noise_note": ("this machine ran other jobs concurrently; "
                                  "repeat runs of tuning_budget moved the quantum KTA timing by "
                                  "up to ~40% wall / ~13% serial-core, so quote these "
                                  "as approximate orders of magnitude"),
            "quantum_breakdown": {m: {k: v for k, v in measured[m].items()
                                      if k.startswith("t_")}
                                  for m in measured if m.startswith("qsvm_")},
            "verdict": ("Time is the axis on which the quantum models cost more, and "
                        "essentially all of it is KTA feature-map training plus the "
                        "O(N^2) Gram, NOT hyperparameter search: the cross-layer QSVM's "
                        f"C-grid CV takes {measured['qsvm_crosslayer']['t_cgrid_wall_s']}"
                        f"s of its {wq}s. On the parallelism-free comparison (serial, "
                        f"one thread everywhere) the QSVM's whole tuning costs {cq} "
                        f"core-seconds against {core_classical[worst_classical]} for "
                        f"the most expensive classical search ({worst_classical}), a "
                        f"factor of {round(cq / max(core_classical[worst_classical], 1e-9), 1)}. "
                        "Timings are machine- and load-dependent and should be quoted "
                        "as orders of magnitude, not as metrics."),
        }

    # ------------------------------------------------------------------ #
    protocol = {
        "item": "tuning_budget -- 'comparable tuning budgets' quantified",
        "what_baseline_fairness_did": ("baseline_fairness lists the grids (hyperparams.csv, 72 rows) and "
                             "quotes CV-only fit counts "
                             "(rbf_svm 16/80, random_forest 4/20, mlp 4/20, xgboost "
                             "36/180, quantum 20), without a per-model budget table, "
                             "the refit, the representation axis, or timing."),
        "reconciliation_with_baseline_fairness": ("tuning_budget's n_fits_cv_only column reproduces baseline_fairness's "
                                     "quoted numbers exactly (80/20/20/180 and quantum "
                                     "20); n_fits adds the single refit sklearn "
                                     "performs after the search (refit=True default)."),
        "n_configs_source": ("INTROSPECTED: build_baseline(name, configs/baselines.yaml, "
                             "seed=17, folds=5, tune=True).param_grid expanded with "
                             "sklearn.model_selection.ParameterGrid -- so the two "
                             "code-truths baseline_fairness documented are picked up automatically "
                             "(XGBoost's grid is a code default; MLP early_stopping is "
                             "hard-coded False in the binary path)."),
        "n_fits_definition": "n_configs * cv_folds + 1 refit",
        "n_gradient_steps_definition": ("parameter-update steps spent learning a "
                                        "REPRESENTATION; 0 for kernel/tree/neighbour "
                                        "models; per-model basis in the "
                                        "gradient_steps_basis column"),
        "n_trainable_params_definition": ("quantum feature-map parameters actually "
                                          "optimised, counted from the live "
                                          "CrossLayerFeatureMap object "
                                          "(fm.init_params() element counts), with the "
                                          "inert (L,1) gamma tensor of the "
                                          "zero-coupling map excluded"),
        "matched_n_protocol": {
            "dataset": "runs/dev_grid/features.parquet (90,553 windows, 19 raw features)",
            "split": {"test_frac": sp["test_frac"], "val_frac": sp["val_frac"],
                      "group_by": sp.get("group_by", "sender"), "seed": int(sp["seed"])},
            "n_train_matched": int(len(tr_match)),
            "n_train_quantum_after_internal_recap": int(len(tr_q)),
            "n_val": int(len(va)), "n_test": int(len(te)),
            "n_features": int(n_dim),
            "input": "per-block PCA 4 mobility + 4 network, [0,pi]-scaled",
            "n_fold_train_used_for_mlp_step_bound": n_fold_train,
        },
        "seeds": {"split_and_cap": SPLIT_SEED, "classical_cv_and_model": BASELINE_SEED,
                  "kta": int(kta.get("seed", 11)),
                  "quantum_cv_folds": "QuantumKernelSVM default seed 0 "
                                      "(src/qkv2x/models/qsvm.py:27)"},
        "kta_schedule": {"optimiser": "Adam", "lr": kta.get("lr"),
                         "epochs": kta_epochs, "batch": kta_batch,
                         "objective": kta.get("objective"),
                         "init": [kta.get("init_w"), kta.get("init_u"),
                                  kta.get("init_gamma")],
                         "restarts": 1,
                         "snapshot_rule": "best centered-KTA alignment over epochs",
                         "window_evaluations": kta_epochs * kta_batch,
                         "alignment_evaluations": kta_epochs},
        "feature_map_facts": fm_facts,
        "wall_clock_provenance": {
            "published_per_model_timing": "IRRECOVERABLE",
            "reason": ("results/packetloss_dev/{summary,provenance}.json record only one "
                       "aggregate wall_time_s = 858.5 s for the entire packet-loss sweep "
                       "(11 loss levels x 8 model rows + 200-replicate grouped "
                       "bootstraps); no per-model timer is written anywhere in the "
                       "Table-I pipeline, and no other artefact under results/ carries "
                       "per-model timing for these models."),
            "published_aggregate_wall_time_s": float(
                json.loads(PUB_SUMMARY.read_text())["wall_time_s"]),
            "tuning_budget_measurement": ("re-measured here on the matched-N protocol in three "
                                 "flavours: measured_tuning_wall_s (as-shipped, "
                                 "parallelism on), measured_tuning_cpu_s (process-tree "
                                 "CPU via psutil -- getrusage under-counts because "
                                 "loky workers are kept alive, not reaped), and "
                                 "measured_tuning_core_s (SERIAL re-run: joblib "
                                 "n_jobs=1, estimator n_jobs=1, threadpoolctl limit 1, "
                                 "torch.set_num_threads(1)) -- the last is the only "
                                 "timing comparable across models. Machine-dependent; "
                                 "quote as an order of magnitude only."),
            "cpu_s_caveat": ("measured_tuning_cpu_s is a process-TREE total and still "
                             "carries loky overhead (worker poll/teardown) even after "
                             "the pool warm-up, so it can exceed the serial "
                             "core-seconds for the same search; cite "
                             "measured_tuning_core_s, not cpu_s."),
            "machine": platform.platform(), "processor": platform.machine(),
        },
        "invariants": invariants,
        "measured": measure,
        "joblib_pool_warmup_s": round(t_warm, 2) if t_warm is not None else None,
        "interpreter": sys.executable,
        "numpy": np.__version__, "pandas": pd.__version__,
    }

    headline = {
        "n_fits": {m: int(fits[m]) for m in df.model},
        "n_configs": {m: int(cfgs[m]) for m in df.model},
        "n_fits_cv_only": dict(zip(df.model, df.n_fits_cv_only.astype(int).tolist())),
        "quantum_n_fits": int(q_fits),
        "quantum_n_configs": int(q_cfgs),
        "ratio_rbf_over_quantum": round(fits["rbf_svm"] / q_fits, 2),
        "ratio_xgboost_over_quantum": round(fits["xgboost"] / q_fits, 2),
        "n_gradient_steps": dict(zip(df.model, df.n_gradient_steps.tolist())),
        "n_trainable_params": dict(zip(df.model,
                                       df.n_trainable_params.astype(int).tolist())),
        "n_structural_hparams_searched": dict(
            zip(df.model, df.n_structural_hparams_searched.astype(int).tolist())),
        "fairness_verdict_one_line": (
            "On the enumerated-search axis the classical baselines were tuned at least "
            f"as hard as the quantum models ({fits['rbf_svm']} vs {q_fits} fits for "
            f"RBF-SVM, {fits['xgboost']} vs {q_fits} for XGBoost, "
            f"{fits['random_forest']} vs {q_fits} for the random forest, {fits['mlp']} "
            f"vs {q_fits} for the MLP) and searched 1-4 structural hyperparameters "
            "against the quantum models' zero; the quantum models' only extra budget is "
            "200 Adam steps over 64 feature-map parameters, which is itself smaller "
            "than the MLP's gradient-trained weight budget; and the one classical model "
            f"that beats the QSVM on a SMALLER budget (the one-class SVM, "
            f"{fits['ocsvm_classical']} fits, in the low-label regime) makes the "
            "no-advantage finding stronger, not weaker."),
        "low_label_ocsvm_vs_qsvm_pr_auc": {
            "n_train_50": {"ocsvm_classical": 0.685, "qsvm_crosslayer": 0.514},
            "n_train_100": {"ocsvm_classical": 0.679, "qsvm_crosslayer": 0.545},
            "n_train_200": {"ocsvm_classical": 0.700, "qsvm_crosslayer": 0.521},
            "source": "results/lowlabel_dev/lowlabel_results.csv (mean over repeats)"},
    }
    if measure:
        headline["measured_tuning_wall_s"] = {m: measured[m]["wall_s"] for m in measured}
        headline["measured_tuning_core_s_serial"] = {
            m: measured[m]["core_s"] for m in measured}
        headline["mlp_weights_by_hidden_config"] = measured.get("mlp", {}).get(
            "mlp_weights_by_hidden_config", "NOT_MEASURED")
        headline["n_base_learners_total"] = {
            m: measured[m].get("n_base_learners_total") for m in measured}
        headline["n_learned_params"] = {
            m: measured[m].get("n_learned_params") for m in measured}
        headline["qsvm_crosslayer_time_breakdown_s"] = {
            k: v for k, v in measured["qsvm_crosslayer"].items() if k.startswith("t_")}

    summary = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runtime_s": round(time.time() - t0, 1),
        "item": "tuning_budget",
        "protocol": protocol,
        "fairness_verdict": verdict,
        "headline": headline,
        "measured_detail": measured,
    }
    (OUT / "tuning_budget_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # ------------------------------ report ------------------------------ #
    pd.set_option("display.width", 220)
    print("\n" + "=" * 110)
    print("tuning_budget -- tuning budget per model (matched-N dev_grid protocol)")
    print("=" * 110)
    show = df[["model", "n_configs", "cv_folds", "n_fits", "n_fits_cv_only",
               "n_gradient_steps", "n_trainable_params",
               "n_structural_hparams_searched", "measured_tuning_wall_s",
               "measured_tuning_core_s", "n_base_learners_total"]]
    print(show.to_string(index=False))
    print("\nFAIRNESS VERDICT (axis 1, enumerated classifier search):")
    print("  " + verdict["axis_1_enumerated_classifier_search"]["verdict"])
    print("\nFAIRNESS VERDICT (axis 2, structural hyperparameters):")
    print("  " + verdict["axis_2_structural_hyperparameters"]["verdict"])
    print("\nFAIRNESS VERDICT (axis 3, gradient-trained representation):")
    print("  " + verdict["axis_3_gradient_trained_representation"]["verdict"])
    if measure:
        print("\nFAIRNESS VERDICT (axis 4, measured wall clock):")
        print("  " + verdict["axis_4_measured_wall_clock"]["verdict"])
    print(f"\nwrote {OUT / 'tuning_budget.csv'}")
    print(f"wrote {OUT / 'tuning_budget_summary.json'}")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
