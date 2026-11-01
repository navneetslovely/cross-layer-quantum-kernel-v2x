"""Matplotlib helpers for the experiment figures (Agg backend; file output only)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _save(fig, out_path):
    """Save a figure as PNG and, alongside, a vector PDF for the paper."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)


def degradation_curves(df, x_col, y_col, group_col, out_path, title,
                       overlay=None, ci=None):
    """One curve per model: y vs x (e.g. PR-AUC vs packet loss). If `ci=(lo_col,
    hi_col)` and those columns exist, draw a shaded 95% confidence band per model."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    has_ci = ci is not None and ci[0] in df.columns and ci[1] in df.columns
    for name, sub in df.groupby(group_col):
        sub = sub.sort_values(x_col)
        line, = ax.plot(sub[x_col], sub[y_col], marker="o", label=str(name))
        if has_ci:
            ax.fill_between(sub[x_col], sub[ci[0]], sub[ci[1]], alpha=0.18,
                            color=line.get_color(), linewidth=0)
    if overlay is not None:
        ox, oy, olabel = overlay
        ax.plot(ox, oy, "k--", lw=1.5, label=olabel)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def learning_curves(df, out_path, title, x_col="n_train", y_col="pr_auc",
                    group_col="model"):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, sub in df.groupby(group_col):
        agg = sub.groupby(x_col)[y_col].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg[x_col], agg["mean"], yerr=agg["std"], marker="o",
                    capsize=3, label=str(name))
    ax.set_xscale("log")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def grouped_bars(df, x, group, value, out_path, title, ylabel):
    """Grouped bar chart: categories on `x`, one bar per `group` (e.g. per model)."""
    import numpy as np
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    cats = list(dict.fromkeys(df[x].tolist()))          # preserve first-seen order
    groups = list(dict.fromkeys(df[group].tolist()))
    nb = max(len(groups), 1)
    width = 0.8 / nb
    xpos = np.arange(len(cats))
    for gi, g in enumerate(groups):
        sub = df[df[group] == g]
        vals = [float(sub[sub[x] == c][value].mean()) if (sub[x] == c).any() else np.nan
                for c in cats]
        ax.bar(xpos + (gi - (nb - 1) / 2) * width, vals, width, label=str(g))
    ax.set_xticks(xpos)
    ax.set_xticklabels([str(c) for c in cats])
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_path)


def bar_per_attack(d: dict, value_key, out_path, title, ylabel):
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    attacks = list(d.keys())
    vals = [d[a][value_key] for a in attacks]
    if not attacks:
        ax.text(0.5, 0.5, "no attack class met the per-class window floor",
                ha="center", va="center", transform=ax.transAxes, fontsize=9,
                color="#888888")
    ax.bar(range(len(attacks)), vals, color="#534AB7")
    ax.set_xticks(range(len(attacks)))
    ax.set_xticklabels(attacks, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    _save(fig, out_path)
