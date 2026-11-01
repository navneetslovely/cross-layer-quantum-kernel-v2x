#!/usr/bin/env bash
# Attack-intensity co-sim sweeps for drift and data_falsification.
# Full pipeline per attack: cosim (SUMO+ns-3) -> features -> intensity experiment.
# Long-running (co-sim is the bottleneck); launch in the background.
set -uo pipefail
cd "$(dirname "$0")/.."
export SUMO_HOME="${SUMO_HOME:-$HOME/opt/sumo}"
export NS3_DIR="${NS3_DIR:-$HOME/opt/ns-3.42}"
PY=python

for atk in drift datafalsification; do
  echo "===== ${atk}: COSIM ($(date +%H:%M:%S)) ====="
  $PY -m qkv2x.cli cosim --config "configs/sim/intensity_${atk}.yaml" || { echo "COSIM FAILED: ${atk}"; exit 1; }
  echo "===== ${atk}: FEATURES ($(date +%H:%M:%S)) ====="
  $PY -m qkv2x.cli features --runs "runs/intensity_${atk}" --config configs/features.yaml || { echo "FEATURES FAILED: ${atk}"; exit 1; }
  echo "===== ${atk}: EXPERIMENT ($(date +%H:%M:%S)) ====="
  $PY -m qkv2x.cli experiment --config "configs/experiments/intensity_${atk}.yaml" || { echo "EXPERIMENT FAILED: ${atk}"; exit 1; }
  echo "===== ${atk}: DONE ($(date +%H:%M:%S)) ====="
done
echo "ALL DONE ($(date +%H:%M:%S))"
