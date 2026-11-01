#!/usr/bin/env bash
# Toolchain install for macOS (Apple Silicon or Intel).
#
# SUMO is built from source via Homebrew deps (for libtraci C++); ns-3.42 is built
# with the qkv2x module; the Python env uses the CPU build of Torch.
# The Mac is good for co-sim + development; the heavy quantum sweeps belong on the
# Everything in the paper runs on CPU.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PREFIX="${PREFIX:-$HOME/opt}"
NS3_VERSION="3.42"
JOBS="${JOBS:-$(sysctl -n hw.ncpu)}"
mkdir -p "$PREFIX"

command -v brew >/dev/null 2>&1 || { echo "Install Homebrew first: https://brew.sh"; exit 1; }

echo "==> [1/5] Homebrew dependencies (incl. GUI: fox + gl2ps + XQuartz)"
brew install cmake ninja xerces-c proj gdal fox gl2ps eigen swig python@3.11 wget || true
# XQuartz (X11) is required to display sumo-gui / netedit on macOS.
brew install --cask xquartz || true

echo "==> [2/5] Build SUMO from source (libtraci + sumo-gui)"
if [[ ! -d "$PREFIX/sumo" ]]; then
  git clone --depth 1 --branch v1_20_0 https://github.com/eclipse-sumo/sumo "$PREFIX/sumo"
fi
cd "$PREFIX/sumo"
# -DCMAKE_DISABLE_FIND_PACKAGE_GDAL=TRUE: SUMO 1.20 does not compile against
# Homebrew's GDAL 3.13 (OGRSpatialReference const-ness change). GDAL is only used
# for OSM/raster geo-import, which the dev_grid scenario does not need. The full
# build (incl. sumo-gui) succeeds with it disabled.
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_DISABLE_FIND_PACKAGE_GDAL=TRUE
cmake --build build -j "$JOBS"
export SUMO_HOME="$PREFIX/sumo"
export PATH="$SUMO_HOME/bin:$PATH"
ls "$SUMO_HOME"/bin/libtracicpp* >/dev/null 2>&1 || { echo "ERROR: libtraci not built"; exit 1; }

echo "==> [3/5] Fetch ns-$NS3_VERSION and link qkv2x"
if [[ ! -d "$PREFIX/ns-3.42" ]]; then
  wget -q "https://www.nsnam.org/releases/ns-allinone-$NS3_VERSION.tar.bz2" -O "$PREFIX/ns3.tar.bz2"
  tar -xjf "$PREFIX/ns3.tar.bz2" -C "$PREFIX"
  mv "$PREFIX/ns-allinone-$NS3_VERSION/ns-$NS3_VERSION" "$PREFIX/ns-3.42"
fi
NS3_DIR="$PREFIX/ns-3.42"
rm -rf "$NS3_DIR/contrib/qkv2x"
ln -s "$REPO/cosim/ns3/contrib/qkv2x" "$NS3_DIR/contrib/qkv2x"

echo "==> [4/5] Configure + build ns-3"
cd "$NS3_DIR"
SUMO_HOME="$SUMO_HOME" ./ns3 configure --enable-examples --build-profile=optimized
SUMO_HOME="$SUMO_HOME" ./ns3 build qkv2x-cosim

echo "==> [5/5] Python environment (conda env qkv2x — matches README.md)"
cd "$REPO"
# Use the canonical conda env (Python 3.11) so env.sh is identical no matter how you
# installed. Torch is installed separately and the package is registered with --no-deps.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda env list | grep -qE '^\s*qkv2x\s' || conda env create -f environment.yml
conda activate qkv2x
pip install -e . --no-deps
pip install torch   # CPU build is sufficient

cat > "$REPO/env.sh" <<ENV
# Source each shell: cd <repo> && source env.sh
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate qkv2x
export SUMO_HOME="$SUMO_HOME"
export NS3_DIR="$NS3_DIR"
export PATH="\$SUMO_HOME/bin:\$PATH"
# quantum.device=auto resolves to CPU.
# if you ever change that.
ENV

echo
echo "DONE. Next:"
echo "  source env.sh"
echo "  bash scripts/run_paper.sh        # full dev_grid results (CPU-only is fine)"
