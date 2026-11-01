"""YAML config loading with light validation and dotted access."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open("r") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config {path} did not parse to a mapping")
    return data


def require(cfg: dict, *keys: str) -> Any:
    """Fetch a nested key, raising a clear error if missing (no silent defaults)."""
    node: Any = cfg
    trail = []
    for k in keys:
        trail.append(k)
        if not isinstance(node, dict) or k not in node:
            raise KeyError(f"missing required config key: {'.'.join(trail)}")
        node = node[k]
    return node
