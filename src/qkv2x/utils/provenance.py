"""Capture run provenance next to results (reproducibility / paper artifact appendix).

Writes a provenance.json into each experiment's out_dir with the exact code + env
state and timing, so every reported number can be traced to a commit, package set,
seeds, and wall-clock."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def _run(cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60,
                              cwd=str(Path(__file__).resolve().parents[3])).stdout.strip()
    except Exception as exc:  # noqa: BLE001 — provenance must never crash the run
        return f"<unavailable: {exc}>"


def capture_provenance(out_dir, *, timestamp: str, config_path: str | None = None,
                       extra: dict | None = None) -> dict:
    """Write {out_dir}/provenance.json. `timestamp` is passed in by the caller (ISO
    string) so the value is explicit. Returns the provenance dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    git = {
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "dirty": _run(["git", "status", "--porcelain"]) != "",
    }
    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"])
    prov = {
        "timestamp": timestamp,
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "git": git,
        "pip_freeze": pip_freeze.splitlines() if "<unavailable" not in pip_freeze else pip_freeze,
        "config_path": config_path,
    }
    if config_path:
        try:
            prov["config"] = Path(config_path).read_text()
        except Exception as exc:  # noqa: BLE001
            prov["config"] = f"<unavailable: {exc}>"
    if extra:
        prov.update(extra)
    (out_dir / "provenance.json").write_text(json.dumps(prov, indent=2, default=str))
    return prov
