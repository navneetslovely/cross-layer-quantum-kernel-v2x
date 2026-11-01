"""Shared utilities for the experiment-backlog study scripts .

The point of this module is *permanence*: every study writes a rich, self-describing
``metadata.json`` (plus the repo's standard ``provenance.json``) capturing the
hyperparameters, train/val/test split sizes & positive counts, complexity counters
(N, Gram O(N^2) size, qubits, statevector dim, circuit depth, #params, #circuits/shots,
approx FLOPs/memory), wall-time, git hash, platform, and library versions -- so the
results can be re-plotted and re-analysed later WITHOUT re-running anything.
"""
from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "results"


def now_iso() -> str:
    return _dt.datetime.now().isoformat(timespec="seconds")


def git_info() -> dict:
    def _run(args):
        try:
            return subprocess.check_output(
                args, cwd=REPO, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except Exception:
            return None

    return {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
    }


def lib_versions() -> dict:
    out = {}
    for m in ["numpy", "scipy", "sklearn", "pandas", "pennylane", "torch", "xgboost"]:
        try:
            out[m] = getattr(__import__(m), "__version__", "?")
        except Exception:
            out[m] = None
    return out


def quantum_complexity(
    n_train: int,
    n_qubits: int,
    layers: int,
    n_edges: int,
    shots: int | None = None,
    n_test: int = 0,
) -> dict:
    """Complexity counters for a quantum-kernel run, for later cost analysis."""
    dim = 2 ** int(n_qubits)
    n_params = 2 * int(layers) * int(n_qubits) + int(layers) * int(n_edges)
    return {
        "n_train": int(n_train),
        "n_test": int(n_test),
        "n_qubits": int(n_qubits),
        "statevector_dim": int(dim),
        "layers": int(layers),
        "n_entangler_edges": int(n_edges),
        "n_trainable_params": int(n_params),
        "gram_train_entries": int(n_train) ** 2,
        "gram_test_entries": int(n_train) * int(n_test),
        "fidelity_time_flops_approx": int(n_train) ** 2 * int(dim),
        "gram_mem_bytes_approx": int(n_train) ** 2 * 8,
        "shots": shots,
        "n_fidelity_circuits_if_hardware": (
            int(n_train) * (int(n_train) - 1) // 2 if shots else None
        ),
    }


def split_meta(y, idx: dict) -> dict:
    """Train/val/test split metadata: sizes, positive counts, prevalence."""
    y = np.asarray(y)
    out = {}
    for name, ii in idx.items():
        if ii is None:
            continue
        yy = y[np.asarray(ii)]
        out[name] = {
            "n": int(len(yy)),
            "n_pos": int(yy.sum()),
            "prevalence": float(yy.mean()) if len(yy) else 0.0,
        }
    return out


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (Path,)):
        return str(o)
    return str(o)


def write_metadata(
    out_dir,
    *,
    study: str,
    started_iso: str,
    t0: float,
    config_path: str | None = None,
    hyperparams: dict | None = None,
    splits: dict | None = None,
    complexity: dict | None = None,
    results_summary: dict | None = None,
    notes: str | None = None,
    extra: dict | None = None,
) -> dict:
    """Write {out_dir}/metadata.json (rich, structured) and the repo's provenance.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "study": study,
        "timestamp": started_iso,
        "wall_time_s": round(time.time() - t0, 3),
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git": git_info(),
        "lib_versions": lib_versions(),
        "config_path": config_path,
        "hyperparams": hyperparams or {},
        "splits": splits or {},
        "complexity": complexity or {},
        "results_summary": results_summary or {},
        "notes": notes,
    }
    if extra:
        meta.update(extra)
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, default=_json_default))
    # Also emit the repo-standard provenance.json (pip_freeze + config text) for consistency.
    try:
        from qkv2x.utils.provenance import capture_provenance

        capture_provenance(
            str(out),
            timestamp=started_iso,
            config_path=config_path,
            extra={
                "study": study,
                "wall_time_s": meta["wall_time_s"],
                "splits": meta["splits"],
                "complexity": meta["complexity"],
                "hyperparams": meta["hyperparams"],
            },
        )
    except Exception as e:  # provenance is best-effort; metadata.json is the source of truth
        meta["provenance_error"] = str(e)
        (out / "metadata.json").write_text(
            json.dumps(meta, indent=2, default=_json_default)
        )
    return meta


def save_table(df, out_dir, name: str):
    """Save a results table as CSV; return the path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / name
    df.to_csv(p, index=False)
    return p
