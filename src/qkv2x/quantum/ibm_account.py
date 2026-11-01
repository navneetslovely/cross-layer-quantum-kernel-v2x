"""Credential resolver for the IBM Quantum runners.

No credentials ship with this repository. Provide your own token in ONE of these ways
(checked in this order):

  1. environment variables  QISKIT_IBM_TOKEN  (+ optional QISKIT_IBM_INSTANCE, QISKIT_IBM_CHANNEL)
  2. a saved Qiskit account -- QiskitRuntimeService.save_account(token=..., channel=...)

The IBM open plan meters a 28-day rolling window of QPU time per account; `pick_account`
refuses to start a job that does not fit in the remaining allocation. The token is never
logged; `describe()` returns a redacted provenance string that is safe to store in results.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_STORE = Path.home() / ".qiskit" / "qiskit-ibm.json"
DEFAULT_CHANNEL = "ibm_quantum_platform"
_PLACEHOLDERS = {"", "YOUR_IBM_QUANTUM_API_TOKEN", "YOUR_INSTANCE_CRN_OR_LEAVE_EMPTY"}


def _clean(v):
    v = (v or "").strip()
    return None if v in _PLACEHOLDERS else v


def _from_env():
    tok = _clean(os.environ.get("QISKIT_IBM_TOKEN"))
    if tok:
        return {"source": "env", "name": "QISKIT_IBM_TOKEN", "token": tok,
                "instance": _clean(os.environ.get("QISKIT_IBM_INSTANCE")),
                "channel": os.environ.get("QISKIT_IBM_CHANNEL", DEFAULT_CHANNEL)}
    return None


def _from_store():
    if not _STORE.exists():
        return []
    accts = json.load(open(_STORE))
    items = [(n, a) for n, a in accts.items() if a.get("token")]
    items.sort(key=lambda kv: (not kv[1].get("is_default_account", False),
                               kv[1].get("channel") != DEFAULT_CHANNEL))
    return [{"source": "qiskit_store", "name": n, "token": a["token"],
             "instance": a.get("instance"),
             "channel": a.get("channel", DEFAULT_CHANNEL)} for n, a in items]


def candidates() -> list[dict]:
    out = [c for c in (_from_env(),) if c]
    out.extend(_from_store())
    return out


def describe(cand: dict) -> str:
    """Redacted provenance safe to store in results JSON (never the token)."""
    inst = cand.get("instance") or ""
    tail = ("..." + inst[-12:]) if len(inst) > 12 else (inst or "(none)")
    return f"{cand['source']}:{cand['name']} channel={cand['channel']} instance={tail}"


def _connect(c):
    from qiskit_ibm_runtime import QiskitRuntimeService
    kw = {"channel": c["channel"], "token": c["token"]}
    if c.get("instance"):
        kw["instance"] = c["instance"]
    return QiskitRuntimeService(**kw)


def get_service(verbose: bool = True):
    """Return (service, provenance_str). Tries each candidate, skipping bad tokens."""
    errs = []
    for c in candidates():
        try:
            svc = _connect(c)
            svc.backends(operational=True, simulator=False)  # force a real call
            if verbose:
                print(f"[ibm_account] using {describe(c)}")
            return svc, describe(c)
        except Exception as e:  # noqa: BLE001 - report all, expose none
            errs.append(f"{describe(c)} -> {type(e).__name__}: {str(e)[:120]}")
    raise RuntimeError(
        "No usable IBM Quantum credential. Tried:\n  " + "\n  ".join(errs)
        + "\nSet QISKIT_IBM_TOKEN to your own token, or save an account with "
          "QiskitRuntimeService.save_account(...).")


def pick_account(need_s: float, verbose: bool = True):
    """Return (service, provenance, snapshot) for a credential with >= need_s of QPU time left."""
    fits, tried = [], []
    for c in candidates():
        try:
            svc = _connect(c)
            snap = remaining_seconds(svc)
            rem = snap.get("remaining_s")
            tried.append((c["name"], rem))
            if isinstance(rem, (int, float)) and rem >= need_s:
                fits.append((rem, c, svc, snap))
        except Exception as e:  # noqa: BLE001
            tried.append((c["name"], f"{type(e).__name__}"))
    if not fits:
        raise RuntimeError(
            f"No credential has {need_s:.0f}s of QPU time free. Balances: {tried}. "
            "The open plan meters a 28-day rolling window, so wait for old jobs to age out.")
    fits.sort(key=lambda t: t[0])
    rem, c, svc, snap = fits[0]
    if verbose:
        print(f"[ibm_account] {describe(c)} | remaining {rem:.0f}s (need {need_s:.0f}s)")
    return svc, describe(c), snap


def remaining_seconds(service) -> dict:
    """Allocation snapshot (the open plan uses a 28-day rolling window)."""
    try:
        u = service.usage() or {}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "reason": f"{type(e).__name__}: {e}"[:160]}
    return {"available": True,
            "limit_s": u.get("usage_limit_seconds"),
            "consumed_s": u.get("usage_consumed_seconds"),
            "remaining_s": u.get("usage_remaining_seconds"),
            "limit_reached": u.get("usage_limit_reached"),
            "usage_period": u.get("usage_period"),
            "window": "28-day rolling"}
