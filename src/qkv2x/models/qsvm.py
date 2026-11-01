"""Precomputed-kernel SVM classifiers.

These are kernel-agnostic: they take Gram matrices (train-train and test-train),
so the SAME class serves the quantum fidelity kernel, the projected kernel, and
any classical precomputed kernel. C / nu are selected by group-aware CV using
PR-AUC (the primary metric for rare attacks).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.svm import SVC, OneClassSVM


def _cv_indices(y, groups, n_splits, seed):
    y = np.asarray(y)
    if groups is not None and len(np.unique(groups)) >= n_splits:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(len(y)), y, groups))
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y))


class QuantumKernelSVM:
    def __init__(self, C_grid=(0.1, 1.0, 10.0, 100.0), cv_folds=5,
                 class_weight="balanced", seed=0):
        self.C_grid = list(C_grid)
        self.cv_folds = int(cv_folds)
        self.class_weight = class_weight
        self.seed = int(seed)
        self.best_C_ = None
        self.cv_score_ = None
        self.svc_ = None

    def fit(self, K_train: np.ndarray, y: np.ndarray, groups=None):
        y = np.asarray(y).astype(int)
        if len(np.unique(y)) < 2:
            raise ValueError("QuantumKernelSVM.fit needs both classes present.")
        folds = _cv_indices(y, groups, self.cv_folds, self.seed)
        best = (-np.inf, self.C_grid[0])
        for C in self.C_grid:
            scores = []
            for tr, va in folds:
                if len(np.unique(y[tr])) < 2:
                    continue
                svc = SVC(C=C, kernel="precomputed", class_weight=self.class_weight)
                svc.fit(K_train[np.ix_(tr, tr)], y[tr])
                s = svc.decision_function(K_train[np.ix_(va, tr)])
                if len(np.unique(y[va])) < 2:
                    continue
                scores.append(average_precision_score(y[va], s))
            mean = float(np.mean(scores)) if scores else -np.inf
            if mean > best[0]:
                best = (mean, C)
        self.cv_score_, self.best_C_ = best[0], best[1]
        self.svc_ = SVC(C=self.best_C_, kernel="precomputed",
                        class_weight=self.class_weight)
        self.svc_.fit(K_train, y)
        return self

    def decision_function(self, K_test_train: np.ndarray) -> np.ndarray:
        return self.svc_.decision_function(K_test_train)

    def predict(self, K_test_train: np.ndarray) -> np.ndarray:
        return self.svc_.predict(K_test_train)


class QuantumKernelMultiSVM:
    """Multi-class precomputed-kernel SVM for attack-TYPE identification.

    sklearn's precomputed-kernel SVC is natively multi-class (one-vs-one), so the
    SAME quantum Gram serves the multi-class confusion-matrix evaluation. C is chosen
    by group-aware CV using macro-F1 (the binary PR-AUC scorer of QuantumKernelSVM is
    undefined for >2 classes). Degenerate folds (a class missing from the fold's train
    split is fine for OvO; folds with <2 classes are skipped)."""

    def __init__(self, C_grid=(1.0,), cv_folds=3, class_weight="balanced", seed=0):
        self.C_grid = list(C_grid)
        self.cv_folds = int(cv_folds)
        self.class_weight = class_weight
        self.seed = int(seed)
        self.best_C_ = None
        self.cv_score_ = None
        self.svc_ = None
        self.classes_ = None

    def fit(self, K_train: np.ndarray, y: np.ndarray, groups=None):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("QuantumKernelMultiSVM.fit needs >= 2 classes present.")
        folds = _cv_indices(y, groups, self.cv_folds, self.seed)
        best = (-np.inf, self.C_grid[0])
        for C in self.C_grid:
            scores = []
            for tr, va in folds:
                if len(np.unique(y[tr])) < 2 or len(va) == 0:
                    continue
                svc = SVC(C=C, kernel="precomputed", class_weight=self.class_weight,
                          decision_function_shape="ovr")
                svc.fit(K_train[np.ix_(tr, tr)], y[tr])
                pred = svc.predict(K_train[np.ix_(va, tr)])
                scores.append(f1_score(y[va], pred, average="macro"))
            mean = float(np.mean(scores)) if scores else -np.inf
            if mean > best[0]:
                best = (mean, C)
        self.cv_score_, self.best_C_ = best[0], best[1]
        self.svc_ = SVC(C=self.best_C_, kernel="precomputed",
                        class_weight=self.class_weight, decision_function_shape="ovr")
        self.svc_.fit(K_train, y)
        return self

    def predict(self, K_test_train: np.ndarray) -> np.ndarray:
        return self.svc_.predict(K_test_train)


class QuantumKernelOneClassSVM:
    """Few-label option: train on benign windows only, flag novelty.

    nu is selected on a labelled validation set (benign+attack) by PR-AUC; if none
    is given, nu defaults to the first value in the grid.
    """

    def __init__(self, nu_grid=(0.01, 0.05, 0.10), seed=0):
        self.nu_grid = list(nu_grid)
        self.seed = int(seed)
        self.best_nu_ = None
        self.ocsvm_ = None

    def fit(self, K_benign_benign: np.ndarray,
            K_val_benign: np.ndarray | None = None, y_val: np.ndarray | None = None):
        best_nu = self.nu_grid[0]
        if K_val_benign is not None and y_val is not None and len(np.unique(y_val)) == 2:
            best_score = -np.inf
            for nu in self.nu_grid:
                oc = OneClassSVM(kernel="precomputed", nu=nu)
                oc.fit(K_benign_benign)
                score = -oc.decision_function(K_val_benign)  # higher => more anomalous
                ap = average_precision_score(y_val, score)
                if ap > best_score:
                    best_score, best_nu = ap, nu
        self.best_nu_ = best_nu
        self.ocsvm_ = OneClassSVM(kernel="precomputed", nu=best_nu)
        self.ocsvm_.fit(K_benign_benign)
        return self

    def decision_function(self, K_test_benign: np.ndarray) -> np.ndarray:
        """Anomaly score: higher => more likely misbehaving."""
        return -self.ocsvm_.decision_function(K_test_benign)

    def predict(self, K_test_benign: np.ndarray) -> np.ndarray:
        """1 = misbehaving (novelty), 0 = benign."""
        return (self.ocsvm_.predict(K_test_benign) == -1).astype(int)
