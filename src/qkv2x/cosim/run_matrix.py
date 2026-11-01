"""Co-simulation sweep driver.

Expands the Cartesian sweep declared in a sim YAML and, for every (sweep point,
seed), constructs ONE ns-3 command line and runs the qkv2x-cosim example. This is
the only place that shells out to ns-3; everything downstream reads the CSVs it
produces. No data is fabricated — each invocation is a full SUMO+ns-3 run.

Environment:
  NS3_DIR  path to the ns-3.42 tree where qkv2x-cosim is built (has ./ns3)
  SUMO_HOME  SUMO install/build with libtraci (sumo on PATH)
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from ..config import load_yaml, require

# sweep key -> how it is applied to the command.
#   "arg:<name>"          -> --<name> <value>
#   "pair:<a>,<b>"        -> sets BOTH --<a>=value and --<b>=value (one knob, two args)
#   "sumo:--scale"        -> appended to sumoExtraArgs as "--scale <value>"
SWEEP_KEY_MAP = {
    "tx_power_dbm": "arg:txPowerDbm",
    "shadowing_sigma_db": "arg:shadowSigmaDb",
    "path_loss_exponent": "arg:plExp",
    "density_scale": "sumo:--scale",
    "bsm_rate_hz": "arg:bsmRateHz",
    "sybil_power_mode": "arg:sybilPowerMode",  # ablation axis: same|offset|random
    # --- attack-INTENSITY axes ---------
    # falsification magnitude knobs; sweep one with a single-attack mix to measure
    # detectability as the attack gets subtler/stronger.
    "const_offset_m": "pair:constOffsetMin,constOffsetMax",  # fixed offset magnitude [m]
    "random_offset_radius": "arg:randomOffsetRadius",        # random-offset radius [m]
    "drift_speed": "arg:driftSpeed",                         # position-drift rate [m/s]
    "flooding_factor": "arg:floodingFactor",                 # x nominal BSM rate
    "sybil_ghosts": "arg:sybilGhosts",                       # # of fake identities
    # Data-falsification magnitude axis: the heading-jitter
    # amplitude [deg] the attacker injects into its claimed motion. Small jitter is
    # benign-like (stealthy), large jitter is kinematically implausible (detectable);
    # sweep it with a single-attack data_falsification mix to trace detectability vs
    # intensity for a second, non-positional attack family.
    "data_fals_heading_jitter": "arg:dataFalsHeadingJitter",  # claimed-heading jitter [deg]
}

# Sybil power-control mode label -> ns-3 integer code (matches AttackProfile).
_SYBIL_MODE_CODES = {"same": 0, "offset": 1, "random": 2}


def _sybil_mode_to_int(v) -> int:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    return _SYBIL_MODE_CODES.get(str(v).lower(), 0)


def _point_id(point: dict[str, Any]) -> str:
    s = ",".join(f"{k}={point[k]}" for k in sorted(point))
    h = hashlib.sha1(s.encode()).hexdigest()[:8]
    keypart = "_".join(f"{k}{point[k]}" for k in sorted(point))
    return f"{keypart}__{h}".replace(".", "p")


def _mix_spec(mix: dict[str, int]) -> str:
    return ",".join(f"{k}:{v}" for k, v in mix.items() if v and v > 0)


def expand_points(cfg: dict) -> list[dict[str, Any]]:
    sweep = cfg.get("sweep", {}) or {}
    if not sweep:
        return [{}]
    keys = list(sweep.keys())
    grids = [sweep[k] for k in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*grids)]


def build_command(cfg: dict, point: dict[str, Any], seed: int, out_dir: Path,
                  ns3_dir: Path) -> list[str]:
    scen = require(cfg, "scenario")
    ch = require(cfg, "channel")
    bsm = require(cfg, "bsm")
    atk = require(cfg, "attacks")
    params = atk.get("params", {})

    sumo_extra = list(scen.get("sumo_extra_args", []))

    base_tx = ch["tx_power_dbm"]
    base_sigma = ch["shadowing_sigma_db"]
    base_eta = ch["path_loss_exponent"]
    base_rate = bsm["rate_hz"]

    overrides: dict[str, Any] = {}
    for k, v in point.items():
        if k not in SWEEP_KEY_MAP:
            raise KeyError(f"sweep key '{k}' has no mapping in SWEEP_KEY_MAP")
        mapping = SWEEP_KEY_MAP[k]
        if mapping.startswith("arg:"):
            overrides[mapping[4:]] = v
        elif mapping.startswith("pair:"):
            for argname in mapping[5:].split(","):
                overrides[argname] = v
        elif mapping.startswith("sumo:"):
            # Emit as a single space-free "key=value" token. ns-3's `./ns3 run`
            # wrapper word-splits forwarded program args, so "--scale 1.0" would
            # arrive split (the value lost) and SUMO would mis-parse it.
            sumo_extra += [f"{mapping[5:]}={v}"]

    # Resolve the SUMO binary to an absolute path under $SUMO_HOME/bin when
    # possible. libtraci's Simulation::start() spawns this binary directly and
    # does not reliably search PATH (notably on macOS), so a bare "sumo" can fail
    # to launch and the cosim aborts with "Could not connect to TraCI server".
    sumo_bin = cfg.get("sumo_binary", "sumo")
    sumo_home = os.environ.get("SUMO_HOME", "")
    if sumo_home and not os.path.isabs(sumo_bin):
        cand = Path(sumo_home) / "bin" / sumo_bin
        if cand.exists():
            sumo_bin = str(cand)

    args = {
        "sumoBinary": sumo_bin,
        # absolute so SUMO finds the config regardless of ns-3's working dir
        "sumoCfg": str(Path(scen["sumocfg"]).resolve()),
        "stepLen": scen["step_length"],
        "warmup": scen["warmup"],
        "duration": scen["duration"],
        "beginTime": scen.get("begin", 0.0),
        "nodePool": cfg["node_pool"],
        "truthEvery": scen.get("truth_every_steps", 1),
        "rsuCount": cfg["rsu"]["count"],
        "rsuMargin": cfg["rsu"]["margin"],
        "rsuZ": cfg["rsu"]["z"],
        "txPowerDbm": overrides.get("txPowerDbm", base_tx),
        "plExp": overrides.get("plExp", base_eta),
        "refLossDb": ch["ref_loss_db"],
        "refDist": ch["ref_distance"],
        "shadowSigmaDb": overrides.get("shadowSigmaDb", base_sigma),
        "shadowTauS": ch["shadowing_tau_s"],
        "nakagami": "true" if ch.get("nakagami", False) else "false",
        "rxSensDbm": ch["rx_sensitivity_dbm"],
        "ccaThreshDbm": ch["cca_threshold_dbm"],
        "dataRate": ch["datarate"],
        "bsmRateHz": overrides.get("bsmRateHz", base_rate),
        "bsmSizeBytes": bsm["size_bytes"],
        "attackFraction": atk["fraction"],
        "onsetMin": atk["onset_uniform"][0],
        "onsetMax": atk["onset_uniform"][1],
        "mixSpec": _mix_spec(atk["mix"]),
        "constOffsetMin": overrides.get("constOffsetMin", params.get("const_offset_min", 50.0)),
        "constOffsetMax": overrides.get("constOffsetMax", params.get("const_offset_max", 200.0)),
        "randomOffsetRadius": overrides.get("randomOffsetRadius",
                                            params.get("random_offset_radius", 100.0)),
        "driftSpeed": overrides.get("driftSpeed", params.get("drift_speed", 2.0)),
        "replayDelay": params.get("replay_delay", 5.0),
        "dataFalsSpeedMin": params.get("data_fals_speed_min", 0.0),
        "dataFalsSpeedMax": params.get("data_fals_speed_max", 2.5),
        "dataFalsHeadingJitter": overrides.get("dataFalsHeadingJitter",
                                               params.get("data_fals_heading_jitter", 60.0)),
        "eventualStopDelay": params.get("eventual_stop_delay", 10.0),
        "sybilGhosts": overrides.get("sybilGhosts", params.get("sybil_ghosts", 3)),
        "sybilSpacing": params.get("sybil_spacing", 8.0),
        "sybilPowerMode": _sybil_mode_to_int(
            overrides.get("sybilPowerMode", params.get("sybil_power_mode", "same"))),
        "sybilPowerOffsetDb": params.get("sybil_power_offset_db", 3.0),
        "floodingFactor": overrides.get("floodingFactor", params.get("flooding_factor", 10.0)),
        "seed": seed,
        "outDir": str(Path(out_dir).resolve()),
    }
    if sumo_extra:
        args["sumoExtraArgs"] = " ".join(str(x) for x in sumo_extra)

    # Pass program args AFTER `--` as separate argv elements (no shell splitting),
    # so values containing spaces (e.g. sumoExtraArgs="--scale 1.0") and paths with
    # spaces survive intact.
    prog_args = [f"--{k}={v}" for k, v in args.items()]
    # Invoke the ns-3 launcher with the SAME interpreter running this CLI (e.g. the
    # qkv2x env's python) rather than relying on its `#!/usr/bin/env python3` shebang.
    # On macOS that shebang can resolve to Homebrew's Python 3.14, whose argparse
    # rejects ns-3.42's launcher ("action 'store_true' is not valid for positional
    # arguments"), making every co-sim fail before SUMO even starts. ns-3.42 is happy
    # under Python <=3.12/3.13 (qkv2x is 3.11).
    return [sys.executable, str(ns3_dir / "ns3"), "run", "qkv2x-cosim", "--"] + prog_args


def _run_one(cmd: list[str], out_dir: Path, ns3_dir: Path, sumo_home: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["SUMO_HOME"] = sumo_home
    env["PATH"] = f"{sumo_home}/bin:" + env.get("PATH", "")
    log = out_dir / "cosim.log"
    with log.open("w") as fh:
        proc = subprocess.run(cmd, cwd=str(ns3_dir), env=env, stdout=fh,
                              stderr=subprocess.STDOUT)
    ok = proc.returncode == 0 and (out_dir / "rx.csv").exists()
    return {"out_dir": str(out_dir), "returncode": proc.returncode, "ok": ok}


def run_matrix(config_path: str) -> dict:
    cfg = load_yaml(config_path)
    ns3_dir = Path(os.environ.get("NS3_DIR", "")).expanduser()
    sumo_home = os.environ.get("SUMO_HOME", "")
    if not ns3_dir or not (ns3_dir / "ns3").exists():
        raise EnvironmentError(
            "NS3_DIR must point at the ns-3.42 tree containing ./ns3 (with qkv2x-cosim built)."
        )
    if not sumo_home:
        raise EnvironmentError("SUMO_HOME must be set (SUMO build with libtraci).")

    out_root = Path(cfg["out_root"]).resolve()
    seeds = cfg["seeds"]
    n_jobs = int(cfg.get("n_jobs", 1))
    points = expand_points(cfg)

    jobs = []
    manifest = {"config": config_path, "scenario": cfg["scenario"]["name"], "points": []}
    for point in points:
        pid = _point_id(point) if point else "base"
        for seed in seeds:
            out_dir = (out_root / pid / f"seed{seed}").resolve()
            cmd = build_command(cfg, point, seed, out_dir, ns3_dir)
            jobs.append((cmd, out_dir))
            manifest["points"].append(
                {"point": point, "seed": seed, "out_dir": str(out_dir)})

    out_root.mkdir(parents=True, exist_ok=True)
    with (out_root / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)

    results = []
    if n_jobs <= 1:
        for cmd, out_dir in jobs:
            results.append(_run_one(cmd, out_dir, ns3_dir, sumo_home))
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            futs = [ex.submit(_run_one, cmd, out_dir, ns3_dir, sumo_home)
                    for cmd, out_dir in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())

    failed = [r for r in results if not r["ok"]]
    summary = {"total": len(results), "ok": len(results) - len(failed), "failed": failed}
    with (out_root / "run_summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2)
    if failed:
        raise RuntimeError(
            f"{len(failed)}/{len(results)} co-sim points failed; see */cosim.log and "
            f"{out_root/'run_summary.json'}."
        )
    return summary
