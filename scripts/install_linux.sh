#!/usr/bin/env bash
# End-to-end toolchain install for Linux / WSL (Ubuntu); optional CUDA Torch for the GPU engine.
#
# Builds SUMO from source (for libtraci C++), ns-3.42 with the qkv2x contrib
# module, and a CUDA Python environment (Torch + PennyLane) for the GPU kernels.
#
# Usage:
#  bash scripts/install_linux.sh            # full install
#  PREFIX=$HOME/opt bash scripts/install_linux.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/opt}"
NS3_VERSION="3.42"
JOBS="${JOBS:-$(nproc)}"
mkdir -p "$PREFIX"

echo "==> [1/5] APT build dependencies (sudo)"
sudo apt-get update
sudo apt-get install -y \
  git cmake g++ ninja-build python3 python3-dev python3-venv python3-pip \
  libxerces-c-dev libgdal-dev libproj-dev libfox-1.6-dev libgl2ps-dev \
  libavformat-dev libswscale-dev libopenscenegraph-dev swig \
  libeigen3-dev pkg-config wget ca-certificates

echo "==> [2/5] Build SUMO from source (provides libtraci C++)"
if [[ ! -d "$PREFIX/sumo" ]]; then
  git clone --depth 1 --branch v1_20_0 https://github.com/eclipse-sumo/sumo "$PREFIX/sumo"
fi
cd "$PREFIX/sumo"
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$JOBS"
export SUMO_HOME="$PREFIX/sumo"
export PATH="$SUMO_HOME/bin:$PATH"
if ! ls "$SUMO_HOME"/bin/libtracicpp* >/dev/null 2>&1; then
  echo "ERROR: libtracicpp not built under $SUMO_HOME/bin" >&2; exit 1
fi
echo "  SUMO_HOME=$SUMO_HOME (libtraci OK)"

echo "==> [3/5] Fetch ns-$NS3_VERSION and link the qkv2x module"
if [[ ! -d "$PREFIX/ns-3.42" ]]; then
  wget -q "https://www.nsnam.org/releases/ns-allinone-$NS3_VERSION.tar.bz2" -O "$PREFIX/ns3.tar.bz2"
  tar -xjf "$PREFIX/ns3.tar.bz2" -C "$PREFIX"
  mv "$PREFIX/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION" "$PREFIX/ns-3.42"
fi
NS3_DIR="$PREFIX/ns-3.42"
rm -rf "$NS3_DIR/contrib/qkv2x"
ln -s "$REPO/cosim/ns3/contrib/qkv2x" "$NS3_DIR/contrib/qkv2x"

echo "==> [4/5] Configure + build ns-3 (examples enabled)"
cd "$NS3_DIR"
SUMO_HOME="$SUMO_HOME" ./ns3 configure --enable-examples --build-profile=optimized
SUMO_HOME="$SUMO_HOME" ./ns3 build qkv2x-cosim

echo "==> [5/5] Python environment (CUDA Torch + PennyLane)"
cd "$REPO"
python3 -m venv "$PREFIX/qkv2x-venv"
# shellcheck disable=SC1091
source "$PREFIX/qkv2x-venv/bin/activate"
pip install --upgrade pip wheel
pip install -e ".[dev]"
# CUDA 12.1 wheels (override TORCH_CUDA for your GPU, or install the CPU build)
pip install torch --index-url "${TORCH_CUDA:-https://download.pytorch.org/whl/cu121}"
# Optional GPU statevector backend (cuQuantum); safe to skip if it fails.
pip install pennylane-lightning-gpu custatevec-cu12 || \
  echo "  (lightning.gpu optional backend not installed; default.qubit+Torch CUDA still uses the GPU)"

cat > "$REPO/env.sh" <<ENV
# Source this before running qkv2x on this machine.
export SUMO_HOME="$SUMO_HOME"
export NS3_DIR="$NS3_DIR"
export PATH="\$SUMO_HOME/bin:\$PATH"
source "$PREFIX/qkv2x-venv/bin/activate"
ENV

echo
echo "DONE. Next:"
echo "  source env.sh"
echo "  bash cosim/sumo/scenarios/dev_grid/make_dev_grid.sh"
echo "  python -m qkv2x.cli cosim --config configs/sim/dev_grid.yaml"
