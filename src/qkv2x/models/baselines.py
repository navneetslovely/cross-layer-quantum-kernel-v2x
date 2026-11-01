"""Classical baselines: RBF-SVM, Random Forest, KNN+bagging, MLP.

All operate on the SAME feature vectors as the quantum kernel (the reduced,
[0,pi]-scaled coordinates) and, optionally, the full pre-reduction feature set.
Each returns an sklearn Pipeline exposing predict / predict_proba.
"""
from __future__ import annotations

from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

BASELINE_NAMES = ["rbf_svm", "random_forest", "knn_bagging", "mlp", "xgboost"]


def _xgb_classifier(seed: int, **kw):
    """Gradient-boosted trees (SOTA tabular baseline). Imported lazily so the
    rest of the zoo works without the optional xgboost dependency."""
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:                      # pragma: no cover
        raise ImportError("xgboost is required for the 'xgboost' baseline "
                          "(pip install xgboost)") from exc
    return XGBClassifier(random_state=seed, n_jobs=-1, eval_metric="logloss",
                         tree_method="hist", **kw)


def _cv(seed, folds=5):
    return StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)


def build_baseline(name: str, cfg: dict, seed: int = 0, folds: int | None = None,
                   tune: bool = True):
    """Return an estimator ready to `.fit(X, y)`.

    `tune=True` → GridSearchCV (StratifiedKFold over `folds`). `tune=False` → a plain
    single-config pipeline with NO inner CV — used at the smallest low-label budgets where
    there are too few positives to cross-validate (otherwise stratified folds get 0–1
    positives and SVC/AP-scoring fail). MLP `early_stopping` is always off (its internal
    10%-stratified split also needs several positives and is non-essential here). Callers
    pass an `folds` clamped to the minority-class count; see run_model.
    NB: no `probability=` on SVC — the kwarg is deprecated in sklearn 1.9 (TD-7); scores
    come from `decision_function` via `baseline_scores`.
    """
    folds = int(cfg.get("cv_folds", 5)) if folds is None else max(2, int(folds))
    if name == "rbf_svm":
        rc = cfg.get("rbf_svm", {})
        pipe = Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf",
                                     class_weight=rc.get("class_weight", "balanced")))])
        if not tune:
            pipe.set_params(svc__C=float(_first(rc.get("C_grid"), [1.0])),
                            svc__gamma=_first(rc.get("gamma_grid"), ["scale"]))
            return pipe
        grid = {"svc__C": rc.get("C_grid", [0.1, 1, 10, 100]),
                "svc__gamma": rc.get("gamma_grid", ["scale", 0.01, 0.1, 1.0])}
        return GridSearchCV(pipe, grid, scoring="average_precision",
                            cv=_cv(seed, folds), n_jobs=-1)
    if name == "random_forest":
        rc = cfg.get("random_forest", {})
        est = RandomForestClassifier(class_weight=rc.get("class_weight", "balanced"),
                                     random_state=seed, n_jobs=-1)
        if not tune:
            est.set_params(n_estimators=int(_first(rc.get("n_estimators"), [300])),
                           max_depth=_first(rc.get("max_depth"), [None]))
            return est
        grid = {"n_estimators": rc.get("n_estimators", [200, 500]),
                "max_depth": rc.get("max_depth", [None, 20])}
        return GridSearchCV(est, grid, scoring="average_precision",
                            cv=_cv(seed, folds), n_jobs=-1)
    if name == "knn_bagging":
        rc = cfg.get("knn_bagging", {})
        base = KNeighborsClassifier()
        bag = BaggingClassifier(estimator=base,
                                n_estimators=int(rc.get("n_estimators", 10)),
                                max_samples=float(rc.get("max_samples", 0.8)),
                                random_state=seed, n_jobs=-1)
        pipe = Pipeline([("scale", StandardScaler()), ("bag", bag)])
        if not tune:
            pipe.set_params(bag__estimator__n_neighbors=int(_first(rc.get("k"), [9])))
            return pipe
        grid = {"bag__estimator__n_neighbors": rc.get("k", [5, 9, 15])}
        return GridSearchCV(pipe, grid, scoring="average_precision",
                            cv=_cv(seed, folds), n_jobs=-1)
    if name == "mlp":
        rc = cfg.get("mlp", {})
        mlp = MLPClassifier(max_iter=int(rc.get("max_iter", 400)),
                            early_stopping=False, random_state=seed)
        pipe = Pipeline([("scale", StandardScaler()), ("mlp", mlp)])
        if not tune:
            pipe.set_params(mlp__hidden_layer_sizes=tuple(_first(rc.get("hidden"), [[64, 32]])),
                            mlp__alpha=float(_first(rc.get("alpha"), [1e-4])))
            return pipe
        grid = {"mlp__hidden_layer_sizes": [tuple(h) for h in rc.get("hidden", [[64, 32]])],
                "mlp__alpha": rc.get("alpha", [1e-4, 1e-3])}
        return GridSearchCV(pipe, grid, scoring="average_precision",
                            cv=_cv(seed, folds), n_jobs=-1)
    if name == "xgboost":
        rc = cfg.get("xgboost", {})
        est = _xgb_classifier(seed)
        if not tune:
            est.set_params(n_estimators=int(_first(rc.get("n_estimators"), [400])),
                           max_depth=int(_first(rc.get("max_depth"), [6])),
                           learning_rate=float(_first(rc.get("learning_rate"), [0.1])))
            return est
        grid = {"n_estimators": rc.get("n_estimators", [200, 400]),
                "max_depth": rc.get("max_depth", [4, 6, 8]),
                "learning_rate": rc.get("learning_rate", [0.05, 0.1]),
                # class imbalance: neg/pos ratio is data-dependent; expose a small grid
                "scale_pos_weight": rc.get("scale_pos_weight", [1.0, 3.0, 5.0])}
        return GridSearchCV(est, grid, scoring="average_precision",
                            cv=_cv(seed, folds), n_jobs=-1)
    raise ValueError(f"unknown baseline {name}")


def _first(seq, default):
    """First grid value (or the default) — used for the single-config multi-class fit."""
    v = seq if seq else default
    return v[0] if isinstance(v, (list, tuple)) and len(v) else v


def build_multiclass(name: str, cfg: dict, seed: int = 0):
    """Return a plain MULTI-CLASS estimator (attack-type identification, ).

    Unlike build_baseline, this does NOT wrap in GridSearchCV with the binary
    'average_precision' scorer (which is undefined for >2 classes). It uses a single
    sensible config (the first of each grid) at a multi-class-native sklearn model,
    so .fit(X, y_multi)/.predict(X) yield attack-type labels for a confusion matrix.
    """
    if name == "rbf_svm":
        rc = cfg.get("rbf_svm", {})
        return Pipeline([("scale", StandardScaler()),
                         ("svc", SVC(kernel="rbf",
                                     C=float(_first(rc.get("C_grid"), [1.0])),
                                     gamma=_first(rc.get("gamma_grid"), ["scale"]),
                                     class_weight=rc.get("class_weight", "balanced"),
                                     decision_function_shape="ovr"))])
    if name == "random_forest":
        rc = cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=int(_first(rc.get("n_estimators"), [300])),
            max_depth=_first(rc.get("max_depth"), [None]),
            class_weight=rc.get("class_weight", "balanced"),
            random_state=seed, n_jobs=-1)
    if name == "knn_bagging":
        rc = cfg.get("knn_bagging", {})
        base = KNeighborsClassifier(n_neighbors=int(_first(rc.get("k"), [9])))
        bag = BaggingClassifier(estimator=base,
                                n_estimators=int(rc.get("n_estimators", 10)),
                                max_samples=float(rc.get("max_samples", 0.8)),
                                random_state=seed, n_jobs=-1)
        return Pipeline([("scale", StandardScaler()), ("bag", bag)])
    if name == "mlp":
        rc = cfg.get("mlp", {})
        return Pipeline([("scale", StandardScaler()),
                         ("mlp", MLPClassifier(
                             hidden_layer_sizes=tuple(_first(rc.get("hidden"), [[64, 32]])),
                             alpha=float(_first(rc.get("alpha"), [1e-4])),
                             max_iter=int(rc.get("max_iter", 400)),
                             early_stopping=rc.get("early_stopping", True),
                             random_state=seed))])
    if name == "xgboost":
        rc = cfg.get("xgboost", {})
        return _xgb_classifier(
            seed,
            n_estimators=int(_first(rc.get("n_estimators"), [400])),
            max_depth=int(_first(rc.get("max_depth"), [6])),
            learning_rate=float(_first(rc.get("learning_rate"), [0.1])))
    raise ValueError(f"unknown multiclass baseline {name}")


def baseline_scores(fitted, X):
    """Uniform decision scores (higher => more malicious) across baseline types."""
    est = fitted.best_estimator_ if isinstance(fitted, GridSearchCV) else fitted
    if hasattr(est, "predict_proba"):
        return est.predict_proba(X)[:, 1]
    if hasattr(est, "decision_function"):
        return est.decision_function(X)
    return est.predict(X).astype(float)
