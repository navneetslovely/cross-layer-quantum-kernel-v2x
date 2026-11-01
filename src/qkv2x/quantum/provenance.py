"""Hardware-provenance instrumentation for QPU runs.

Records the 18 hardware-reporting fields (``PROVENANCE_FIELDS``) for every
QPU run in one place, so individual runners do not have to write them down.

``capture_run_provenance`` is a **pure function of the objects a runner already
has** (backend, ISA circuits, pass-manager settings, job). It needs no network
connection of its own -- everything it reads is either local (circuits, module
versions) or already cached on the backend object -- so it is import-safe and
unit-testable offline against fake backends.

Two invariants that the tests enforce and that the paper relies on:

* **Never omit.** The returned dict always contains all 18 provenance keys
  (``PROVENANCE_FIELDS``). A field that genuinely cannot be obtained is set to
  the literal string ``IRRECOVERABLE`` and the reason is recorded under
  ``provenance["irrecoverable"][field]``.
* **Never fabricate.** No value is defaulted, assumed, back-solved or carried
  over from another run. Every number is read from the object handed in.

Calibration figures (T1/T2/gate error/readout error) are aggregated over the
physical qubits **actually used** by the routed circuit, i.e. the union of the
qubits touched by any instruction in the ISA circuit -- see
``patch_qubits`` and ``physical_qubit_ids`` in the output.

Usage in a runner (purely additive, never fatal)::

    from qkv2x.quantum.provenance import capture_run_provenance, safe_capture

    out["provenance"] = safe_capture(
        backend=backend, isa_circuits=isa, logical_circuits=circs,
        pass_manager_settings={"optimization_level": 3,
                               "seed_transpiler": args.seed,
                               "initial_layout": layout},
        job=job, shots=args.shots,
    )

``safe_capture`` never raises: a provenance failure must never kill a run that
has already spent paid QPU seconds.
"""
from __future__ import annotations

import importlib
import platform
import sys
import traceback
from datetime import datetime, timezone
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "IRRECOVERABLE",
    "PROVENANCE_SCHEMA_VERSION",
    "PROVENANCE_FIELDS",
    "capture_run_provenance",
    "safe_capture",
    "software_fingerprint",
    "missing_fields",
]

#: Sentinel written for any field that cannot be recovered. Never omit a key,
#: never guess a value -- write this and a reason instead.
IRRECOVERABLE = "IRRECOVERABLE"

PROVENANCE_SCHEMA_VERSION = "1.0"

#: The 18 provenance fields recorded for every hardware run.
#: ``capture_run_provenance`` guarantees every one of these keys is present.
PROVENANCE_FIELDS: tuple[str, ...] = (
    "physical_qubit_ids",          # 1  physical qubit IDs (final layout)
    "logical_2q_gate_count",       # 2  logical (pre-routing) 2q gate count
    "routed_2q_gate_count",        # 3  routed 2q gate count
    "routed_2q_depth",             # 4  routed 2q depth
    "routed_total_depth",          # 5  routed total (full) depth
    "median_2q_error",             # 6  median 2q gate error of the used patch
    "median_readout_error",        # 7  median readout error of the used patch
    "min_t1_us",                   # 8  min T1 of the used patch
    "min_t2_us",                   # 9  min T2 of the used patch
    "backend_name",                # 10 backend name
    "shots",                       # 11 shots
    "calibration_last_update_date",  # 12 calibration timestamp
    "backend_version",             # 13 backend version / processor revision
    "software_versions",           # 14 qiskit / qiskit-ibm-runtime / ... at run time
    "transpiler_settings",         # 15 optimization level, seed, initial layout
    "resilience_settings",         # 16 error suppression / mitigation as resolved
    "job_id",                      # 17 job id
    "qpu_usage_seconds",           # 18 QPU usage seconds
)

#: Modules whose installed version is recorded with every run.
_TRACKED_MODULES = ("qiskit", "qiskit_ibm_runtime", "qiskit_aer", "numpy",
                    "scipy", "sklearn", "torch", "pennylane")

_TWO_Q = 2


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class _Recorder:
    """Collects values, turning any failure into IRRECOVERABLE + a reason."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.reasons: dict[str, str] = {}

    def set(self, field: str, fn, *, missing: str) -> Any:
        """Store ``fn()``; on exception or ``None`` store IRRECOVERABLE."""
        try:
            val = fn()
        except Exception as exc:  # noqa: BLE001 - provenance must never raise
            self.fail(field, f"{type(exc).__name__}: {exc}")
            return IRRECOVERABLE
        if val is None:
            self.fail(field, missing)
            return IRRECOVERABLE
        self.values[field] = val
        return val

    def fail(self, field: str, reason: str) -> None:
        self.values[field] = IRRECOVERABLE
        self.reasons[field] = reason


def _as_list(obj: Any) -> list:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if isinstance(obj, Iterable) and not hasattr(obj, "count_ops"):
        return list(obj)
    return [obj]


def _two_qubit_ops(circuit) -> int:
    """Count instructions acting on exactly 2 qubits (barriers excluded)."""
    n = 0
    for inst in circuit.data:
        op = getattr(inst, "operation", None)
        name = getattr(op, "name", "")
        if name in ("barrier", "delay"):
            continue
        if len(inst.qubits) == _TWO_Q:
            n += 1
    return n


def _two_qubit_depth(circuit) -> int:
    return int(circuit.depth(lambda instr: len(instr.qubits) == _TWO_Q))


def _active_physical_qubits(circuit) -> list[int]:
    """Physical indices touched by any real instruction in a routed circuit."""
    idx = circuit.find_bit
    used: set[int] = set()
    for inst in circuit.data:
        name = getattr(getattr(inst, "operation", None), "name", "")
        if name in ("barrier", "delay"):
            continue
        for q in inst.qubits:
            used.add(int(idx(q).index))
    return sorted(used)


def _final_layout(circuit) -> list[int] | None:
    layout = getattr(circuit, "layout", None)
    if layout is None:
        return None
    for attr in ("final_index_layout", "initial_index_layout"):
        fn = getattr(layout, attr, None)
        if callable(fn):
            try:
                return [int(q) for q in fn()]
            except Exception:  # noqa: BLE001
                continue
    return None


def _iso(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return value


# --------------------------------------------------------------------------- #
# software fingerprint
# --------------------------------------------------------------------------- #
def software_fingerprint(modules: Sequence[str] = _TRACKED_MODULES) -> dict[str, str]:
    """Installed versions of the run-relevant stack. Never raises.

    A module that is not installed is reported as ``"not-installed"`` (that is a
    fact, not a missing value); a module whose version cannot be determined is
    reported as IRRECOVERABLE.
    """
    out: dict[str, str] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for name in modules:
        try:
            mod = importlib.import_module(name)
        except Exception:  # noqa: BLE001
            out[name] = "not-installed"
            continue
        ver = getattr(mod, "__version__", None)
        out[name] = str(ver) if ver else IRRECOVERABLE
    return out


# --------------------------------------------------------------------------- #
# backend calibration
# --------------------------------------------------------------------------- #
def _backend_properties(backend):
    fn = getattr(backend, "properties", None)
    return fn() if callable(fn) else None


def _patch_calibration(backend, patch: Sequence[int]) -> tuple[dict[str, Any], dict[str, str]]:
    """Per-qubit T1/T2/readout + patch-internal 2q errors, for ``patch`` only.

    Returns ``(values, reasons)``. T1/T2 are converted to microseconds; the
    source (BackendProperties vs Target.qubit_properties) is recorded so the
    number can be traced.
    """
    vals: dict[str, Any] = {}
    reasons: dict[str, str] = {}
    if not patch:
        for f in ("min_t1_us", "min_t2_us", "median_readout_error", "median_2q_error"):
            vals[f] = IRRECOVERABLE
            reasons[f] = "no physical qubit patch could be determined from the ISA circuit"
        vals["per_qubit"] = {}
        return vals, reasons

    props = None
    try:
        props = _backend_properties(backend)
    except Exception as exc:  # noqa: BLE001
        reasons["backend_properties"] = f"{type(exc).__name__}: {exc}"

    per_qubit: dict[str, dict[str, Any]] = {}
    t1s, t2s, ros = [], [], []
    source = "IRRECOVERABLE"
    for q in patch:
        rec: dict[str, Any] = {"t1_us": IRRECOVERABLE, "t2_us": IRRECOVERABLE,
                               "readout_error": IRRECOVERABLE}
        if props is not None:
            try:
                rec["t1_us"] = float(props.t1(q)) * 1e6
                rec["t2_us"] = float(props.t2(q)) * 1e6
                rec["readout_error"] = float(props.readout_error(q))
                source = "BackendProperties"
            except Exception:  # noqa: BLE001
                pass
        if rec["t1_us"] is IRRECOVERABLE or rec["t1_us"] == IRRECOVERABLE:
            try:
                qp = backend.qubit_properties(q)
                rec["t1_us"] = float(qp.t1) * 1e6
                rec["t2_us"] = float(qp.t2) * 1e6
                source = "Target.qubit_properties"
            except Exception:  # noqa: BLE001
                pass
        per_qubit[str(q)] = rec
        if isinstance(rec["t1_us"], float):
            t1s.append(rec["t1_us"])
        if isinstance(rec["t2_us"], float):
            t2s.append(rec["t2_us"])
        if isinstance(rec["readout_error"], float):
            ros.append(rec["readout_error"])

    vals["per_qubit"] = per_qubit
    vals["calibration_source"] = source

    def _agg(field: str, seq, fn, what: str):
        if seq:
            vals[field] = float(fn(seq))
        else:
            vals[field] = IRRECOVERABLE
            reasons[field] = f"backend exposed no {what} for any qubit in the used patch"

    _agg("min_t1_us", t1s, min, "T1")
    _agg("min_t2_us", t2s, min, "T2")
    _agg("median_readout_error", ros, median, "readout error")

    # median 2q error over edges whose BOTH endpoints are in the patch
    errs: list[float] = []
    edges: list[list[int]] = []
    try:
        target = backend.target
        pset = set(int(q) for q in patch)
        for name in target.operation_names:
            inst = target.get(name, None)
            if not inst:
                continue
            for qargs, ip in inst.items():
                if qargs is None or len(qargs) != _TWO_Q:
                    continue
                if int(qargs[0]) in pset and int(qargs[1]) in pset:
                    err = getattr(ip, "error", None)
                    if err is not None:
                        errs.append(float(err))
                        edges.append([int(qargs[0]), int(qargs[1])])
    except Exception as exc:  # noqa: BLE001
        reasons["median_2q_error"] = f"{type(exc).__name__}: {exc}"
    if errs:
        vals["median_2q_error"] = float(median(errs))
        vals["n_2q_edges_in_patch"] = len(errs)
        vals["patch_2q_edges"] = edges
    else:
        vals["median_2q_error"] = IRRECOVERABLE
        reasons.setdefault(
            "median_2q_error",
            "no 2-qubit instruction in the backend target has both endpoints "
            "inside the used patch, or the target carries no error values")
    return vals, reasons


# --------------------------------------------------------------------------- #
# job metadata
# --------------------------------------------------------------------------- #
def _job_usage_seconds(job) -> float | None:
    """QPU usage seconds, trying every API surface, never guessing."""
    fn = getattr(job, "usage", None)
    if callable(fn):
        u = fn()
        if isinstance(u, (int, float)):
            return float(u)
        if isinstance(u, Mapping):
            for k in ("quantum_seconds", "seconds", "usage_seconds"):
                if isinstance(u.get(k), (int, float)):
                    return float(u[k])
    mfn = getattr(job, "metrics", None)
    if callable(mfn):
        m = mfn() or {}
        usage = m.get("usage") or {}
        for k in ("quantum_seconds", "seconds"):
            if isinstance(usage.get(k), (int, float)):
                return float(usage[k])
        if isinstance(m.get("usage_seconds"), (int, float)):
            return float(m["usage_seconds"])
    return None


def _resolved_options(job, override: Any) -> Any:
    """Resilience / error-suppression options as ACTUALLY resolved for the job."""
    if override is not None:
        return override
    inputs = getattr(job, "inputs", None)
    if isinstance(inputs, Mapping):
        opts = inputs.get("options")
        if opts is not None:
            return _jsonable(opts)
        # SamplerV2 puts twirling/dynamical-decoupling under pubs-level options
        return _jsonable({k: v for k, v in inputs.items()
                          if k in ("options", "resilience_level", "twirling",
                                   "dynamical_decoupling")}) or None
    return None


def _jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    for attr in ("_asdict",):
        fn = getattr(obj, attr, None)
        if callable(fn):
            return _jsonable(fn())
    d = getattr(obj, "__dict__", None)
    if isinstance(d, dict) and d:
        return {str(k): _jsonable(v) for k, v in d.items() if not str(k).startswith("_")}
    return str(obj)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def capture_run_provenance(
    backend=None,
    isa_circuits=None,
    pass_manager_settings: Mapping[str, Any] | None = None,
    job=None,
    extra: Mapping[str, Any] | None = None,
    *,
    logical_circuits=None,
    shots: int | None = None,
    resilience_settings: Any = None,
) -> dict[str, Any]:
    """Capture every provenance field knowable at run time.

    Parameters
    ----------
    backend
        Any Qiskit ``BackendV2`` (real ``IBMBackend`` or a fake). Only cached /
        local attributes are read -- no job is ever submitted from here.
    isa_circuits
        The routed (post-transpilation) circuit or list of circuits actually
        submitted. Routed gate counts, depths and the physical patch come from
        the FIRST circuit (every kernel-pair circuit in this project is
        structurally identical); ``n_isa_circuits`` records how many there were.
    pass_manager_settings
        The settings handed to ``generate_preset_pass_manager`` --
        ``optimization_level``, ``seed_transpiler``, ``initial_layout``.
    job
        The runtime job (already submitted / completed). Used only for
        ``job_id``, usage seconds and resolved options.
    extra
        Free-form dict merged into the output under ``extra`` (e.g. map name,
        n_train/n_test). Never overrides a provenance field.
    logical_circuits
        The pre-routing circuits, so the logical 2q gate count is recorded
        rather than reconstructed after the fact.
    shots
        Shots per circuit as actually requested.
    resilience_settings
        Explicit override when the runner knows the resolved options.

    Returns
    -------
    dict
        Contains all of :data:`PROVENANCE_FIELDS` plus ``schema_version``,
        ``captured_at``, ``irrecoverable`` (field -> reason) and ``extra``.
        Missing values are the literal string :data:`IRRECOVERABLE`.
    """
    rec = _Recorder()

    isa_list = _as_list(isa_circuits)
    log_list = _as_list(logical_circuits)
    isa0 = isa_list[0] if isa_list else None
    log0 = log_list[0] if log_list else None

    # --- routed circuit structure ------------------------------------------
    no_isa = "no ISA (routed) circuit was passed to capture_run_provenance"
    if isa0 is None:
        for f in ("physical_qubit_ids", "routed_2q_gate_count",
                  "routed_2q_depth", "routed_total_depth"):
            rec.fail(f, no_isa)
        patch: list[int] = []
    else:
        patch = []
        try:
            patch = _active_physical_qubits(isa0)
        except Exception as exc:  # noqa: BLE001
            rec.fail("physical_qubit_ids", f"{type(exc).__name__}: {exc}")
        layout = None
        try:
            layout = _final_layout(isa0)
        except Exception:  # noqa: BLE001
            layout = None
        if layout:
            rec.values["physical_qubit_ids"] = layout
        elif patch:
            rec.values["physical_qubit_ids"] = patch
            rec.reasons["physical_qubit_ids_note"] = (
                "circuit carried no TranspileLayout; reporting the physical "
                "qubits actually operated on instead of the final layout")
        else:
            rec.fail("physical_qubit_ids",
                     "routed circuit exposed neither a TranspileLayout nor any "
                     "instruction from which physical qubits could be read")
        rec.set("routed_2q_gate_count", lambda: _two_qubit_ops(isa0), missing=no_isa)
        rec.set("routed_2q_depth", lambda: _two_qubit_depth(isa0), missing=no_isa)
        rec.set("routed_total_depth", lambda: int(isa0.depth()), missing=no_isa)

    if log0 is None:
        rec.fail("logical_2q_gate_count",
                 "no pre-routing (logical) circuit was passed to "
                 "capture_run_provenance; pass logical_circuits=")
    else:
        rec.set("logical_2q_gate_count", lambda: _two_qubit_ops(log0),
                missing="logical circuit contained no countable instructions")

    # --- backend identity + calibration ------------------------------------
    if backend is None:
        for f in ("backend_name", "backend_version", "calibration_last_update_date",
                  "min_t1_us", "min_t2_us", "median_2q_error", "median_readout_error"):
            rec.fail(f, "no backend object was passed to capture_run_provenance")
        cal_extra: dict[str, Any] = {}
    else:
        rec.set("backend_name", lambda: str(getattr(backend, "name", None) or None),
                missing="backend object exposes no .name")
        rec.set("backend_version", lambda: _backend_version(backend),
                missing="backend exposes neither backend_version nor a "
                        "processor_type/revision in its configuration")
        rec.set("calibration_last_update_date",
                lambda: _calibration_timestamp(backend),
                missing="backend.properties() returned no last_update_date "
                        "(simulators and some fakes have no calibration record)")
        cal_vals, cal_reasons = _patch_calibration(backend, patch)
        for f in ("min_t1_us", "min_t2_us", "median_2q_error", "median_readout_error"):
            v = cal_vals.get(f, IRRECOVERABLE)
            if v == IRRECOVERABLE:
                rec.fail(f, cal_reasons.get(f, "not exposed by the backend"))
            else:
                rec.values[f] = v
        rec.reasons.update({k: v for k, v in cal_reasons.items()
                            if k not in ("min_t1_us", "min_t2_us",
                                         "median_2q_error", "median_readout_error")})
        cal_extra = {k: v for k, v in cal_vals.items()
                     if k in ("per_qubit", "calibration_source",
                              "n_2q_edges_in_patch", "patch_2q_edges")}

    # --- run settings -------------------------------------------------------
    pms = dict(pass_manager_settings or {})
    if pms:
        rec.values["transpiler_settings"] = {
            "optimization_level": pms.get("optimization_level", IRRECOVERABLE),
            "seed_transpiler": pms.get("seed_transpiler", IRRECOVERABLE),
            "initial_layout": _jsonable(pms.get("initial_layout", None)),
            **{k: _jsonable(v) for k, v in pms.items()
               if k not in ("optimization_level", "seed_transpiler", "initial_layout")},
        }
    else:
        rec.fail("transpiler_settings",
                 "runner did not pass pass_manager_settings=")

    if shots is None:
        rec.fail("shots", "shots were not passed to capture_run_provenance")
    else:
        rec.values["shots"] = int(shots)

    rec.set("resilience_settings", lambda: _resolved_options(job, resilience_settings),
            missing="no options were set on the job; the library defaults of the "
                    "recorded qiskit-ibm-runtime version applied "
                    "(no resilience level, no mitigation requested)")

    # --- job ----------------------------------------------------------------
    if job is None:
        rec.fail("job_id", "no job object was passed (offline / simulator run)")
        rec.fail("qpu_usage_seconds", "no job object was passed (offline / simulator run)")
    else:
        rec.set("job_id", lambda: str(job.job_id()),
                missing="job object exposes no job_id()")
        rec.set("qpu_usage_seconds", lambda: _job_usage_seconds(job),
                missing="job exposed neither usage() nor metrics()['usage'] with "
                        "a numeric quantum-seconds field")

    rec.values["software_versions"] = software_fingerprint()

    out: dict[str, Any] = {f: rec.values.get(f, IRRECOVERABLE) for f in PROVENANCE_FIELDS}
    out["schema_version"] = PROVENANCE_SCHEMA_VERSION
    out["captured_at"] = datetime.now(timezone.utc).isoformat()
    out["patch_qubits"] = patch or IRRECOVERABLE
    out["n_isa_circuits"] = len(isa_list) if isa_list else IRRECOVERABLE
    out["calibration_detail"] = cal_extra
    out["irrecoverable"] = dict(rec.reasons)
    out["extra"] = _jsonable(dict(extra or {}))
    return out


def _backend_version(backend) -> str | None:
    bv = getattr(backend, "backend_version", None)
    conf = None
    try:
        cfn = getattr(backend, "configuration", None)
        conf = cfn() if callable(cfn) else None
    except Exception:  # noqa: BLE001
        conf = None
    proc = getattr(conf, "processor_type", None) if conf is not None else None
    if bv is None and proc is None:
        return None
    parts = []
    if bv is not None:
        parts.append(f"backend_version={bv}")
    if proc:
        parts.append(f"processor={_jsonable(proc)}")
    return "; ".join(str(p) for p in parts)


def _calibration_timestamp(backend) -> str | None:
    props = _backend_properties(backend)
    if props is None:
        return None
    ts = getattr(props, "last_update_date", None)
    return _iso(ts) if ts is not None else None


def missing_fields(prov: Mapping[str, Any]) -> list[str]:
    """Provenance fields that came back IRRECOVERABLE (for a runner to print)."""
    return [f for f in PROVENANCE_FIELDS if prov.get(f, IRRECOVERABLE) == IRRECOVERABLE]


def safe_capture(**kwargs) -> dict[str, Any]:
    """``capture_run_provenance`` that can never raise.

    A paid QPU run must never lose its results because provenance logging
    failed. On any internal error this returns a minimal dict that still has
    all 18 provenance keys, every one marked IRRECOVERABLE, plus the traceback.
    """
    try:
        return capture_run_provenance(**kwargs)
    except BaseException as exc:  # noqa: BLE001 - fail-safe by design
        reason = f"provenance capture itself failed: {type(exc).__name__}: {exc}"
        out: dict[str, Any] = {f: IRRECOVERABLE for f in PROVENANCE_FIELDS}
        out["schema_version"] = PROVENANCE_SCHEMA_VERSION
        out["captured_at"] = datetime.now(timezone.utc).isoformat()
        out["capture_failed"] = True
        out["irrecoverable"] = {f: reason for f in PROVENANCE_FIELDS}
        out["traceback"] = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__))[-4000:]
        out["extra"] = {}
        return out
