"""Detection metrics. PR-AUC is primary (attacks are rare)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """False-positive rate at the threshold achieving >= target TPR."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    pos = scores[y_true == 1]
    neg = scores[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thr = np.quantile(pos, 1.0 - target_tpr)  # threshold s.t. ~target_tpr of pos pass
    return float(np.mean(neg >= thr))


def detection_report(y_true: np.ndarray, scores: np.ndarray,
                     y_pred: np.ndarray | None = None, target_tpr: float = 0.95) -> dict:
    """Binary detection metrics from decision scores (higher => more malicious)."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    if y_pred is None:
        y_pred = (scores >= np.median(scores)).astype(int)
    out = {
        "n": int(len(y_true)),
        "n_pos": int((y_true == 1).sum()),
        "prevalence": float(np.mean(y_true)),
    }
    if out["n_pos"] == 0 or out["n_pos"] == out["n"]:
        # degenerate split; AUCs undefined
        out.update({"pr_auc": float("nan"), "roc_auc": float("nan")})
    else:
        out["pr_auc"] = float(average_precision_score(y_true, scores))
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
        out["fpr_at_tpr"] = fpr_at_tpr(y_true, scores, target_tpr)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0)
    out.update({
        "precision": float(p), "recall": float(r), "f1": float(f1),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    })
    return out


def macro_report(y_true, y_pred, labels=None) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0, labels=labels)
    return {
        "macro_precision": float(p),
        "macro_recall": float(r),
        "macro_f1": float(f1),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": list(labels) if labels is not None else sorted(set(map(str, y_true))),
    }


def _safe_auc(y: np.ndarray, s: np.ndarray, kind: str) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s)) if kind == "pr" else float(roc_auc_score(y, s))


def grouped_bootstrap_ci(y_true, scores, groups, n_boot: int = 200, seed: int = 0,
                         alpha: float = 0.05, metrics=("pr_auc", "roc_auc")) -> dict:
    """Confidence intervals via GROUPED bootstrap.

    Resamples whole transmitters (`groups`) with replacement so that the correlated
    windows of one physical sender stay together — the honest unit of resampling for
    these data. Returns {metric: {mean, lo, hi}} at the (1-alpha) level. This is how we
    get error bars on the single-split H2/RQ3 point estimates."""
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores, dtype=float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    acc = {m: [] for m in metrics}
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(g == gg)[0] for gg in pick])
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        if "pr_auc" in acc:
            acc["pr_auc"].append(_safe_auc(yb, sb, "pr"))
        if "roc_auc" in acc:
            acc["roc_auc"].append(_safe_auc(yb, sb, "roc"))
    out = {}
    for m, vals in acc.items():
        v = np.asarray([x for x in vals if np.isfinite(x)], dtype=float)
        if v.size == 0:
            out[m] = {"mean": float("nan"), "lo": float("nan"), "hi": float("nan"),
                      "n_boot": 0}
        else:
            out[m] = {"mean": float(v.mean()),
                      "lo": float(np.quantile(v, alpha / 2)),
                      "hi": float(np.quantile(v, 1 - alpha / 2)),
                      "n_boot": int(v.size)}
    return out


def holm_bonferroni(pvals: dict) -> dict:
    """Holm–Bonferroni step-down adjustment for a FAMILY of comparisons.

    Input: {comparison_name: raw_p}. Output: adjusted p-values (monotone,
    clipped at 1). Controls the family-wise error rate without independence
    assumptions — the standard correction for the multiple quantum-vs-classical
    comparisons reported per experiment."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        running = max(running, (m - rank) * float(p))
        adj[name] = min(1.0, running)
    return adj


def paired_bootstrap_auc_diff(y_true, scores_a, scores_b, groups, n_boot: int = 200,
                              seed: int = 0, metric: str = "roc_auc") -> dict:
    """Significance of an AUC gap between two models on the SAME windows.

    Grouped PAIRED bootstrap: each replicate resamples transmitters and recomputes
    AUC(a)-AUC(b) on the same resample (so the comparison is paired). Returns the
    observed delta, a 95% CI on the delta, and a two-sided bootstrap p-value (the gap
    is significant at 0.05 if the CI excludes 0 / p<0.05). Grouped resampling is used
    instead of DeLong because windows are not iid (correlated within a transmitter)."""
    y = np.asarray(y_true).astype(int)
    sa = np.asarray(scores_a, dtype=float)
    sb = np.asarray(scores_b, dtype=float)
    g = np.asarray(groups)
    uniq = np.unique(g)
    rng = np.random.default_rng(seed)
    kind = "pr" if metric == "pr_auc" else "roc"
    obs = _safe_auc(y, sa, kind) - _safe_auc(y, sb, kind)
    diffs = []
    for _ in range(int(n_boot)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([np.where(g == gg)[0] for gg in pick])
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        diffs.append(_safe_auc(yb, sa[idx], kind) - _safe_auc(yb, sb[idx], kind))
    diffs = np.asarray([d for d in diffs if np.isfinite(d)], dtype=float)
    if diffs.size == 0:
        return {"metric": metric, "delta": float(obs), "lo": float("nan"),
                "hi": float("nan"), "p_two_sided": float("nan"), "n_boot": 0}
    p = 2.0 * min(float(np.mean(diffs <= 0)), float(np.mean(diffs >= 0)))
    return {"metric": metric, "delta": float(obs),
            "lo": float(np.quantile(diffs, 0.025)),
            "hi": float(np.quantile(diffs, 0.975)),
            "p_two_sided": float(min(p, 1.0)), "n_boot": int(diffs.size)}


def detection_latency(
    t_window_end: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    onset: np.ndarray | None = None,
) -> dict:
    """Mean time from first malicious window to first correct detection, per group.

    If `onset` (attack onset time per window) is provided, latency is measured from
    onset; otherwise from the first malicious window's end time.
    """
    t = np.asarray(t_window_end, dtype=float)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    g = np.asarray(groups)
    latencies = []
    n_attacked = 0          # groups that contain at least one malicious window
    for grp in np.unique(g):
        m = g == grp
        if not np.any(y[m] == 1):
            continue
        n_attacked += 1
        tm = t[m]
        ym = y[m]
        pm = p[m]
        order = np.argsort(tm)
        tm, ym, pm = tm[order], ym[order], pm[order]
        first_atk_t = tm[ym == 1][0] if onset is None else float(np.min(onset[m][ym == 1]))
        detected = np.where((ym == 1) & (pm == 1))[0]
        if detected.size == 0:
            continue
        latencies.append(float(tm[detected[0]] - first_atk_t))
    if n_attacked == 0:
        return {"detection_latency_mean": float("nan"),
                "detection_latency_median": float("nan"),
                "detected_fraction": float("nan"), "n_attacked_groups": 0}
    # detected_fraction = attacked groups EVER correctly flagged / attacked groups
    detected_fraction = len(latencies) / n_attacked
    if not latencies:
        return {"detection_latency_mean": float("nan"),
                "detection_latency_median": float("nan"),
                "detected_fraction": 0.0, "n_attacked_groups": int(n_attacked)}
    return {
        "detection_latency_mean": float(np.mean(latencies)),
        "detection_latency_median": float(np.median(latencies)),
        "detected_fraction": float(detected_fraction),
        "n_attacked_groups": int(n_attacked),
    }
