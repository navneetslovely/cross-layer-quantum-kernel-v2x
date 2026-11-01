# Source this in every new terminal:  cd <repo> && source env.sh
# Activates the conda env and points at your SUMO + ns-3.42 toolchain.
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate qkv2x

# Set these to your own installation paths (see README.md, Setup).
export SUMO_HOME="${SUMO_HOME:-/path/to/sumo}"
export NS3_DIR="${NS3_DIR:-/path/to/ns-3.42}"
export PATH="$SUMO_HOME/bin:$PATH"

# The full pipeline runs on CPU; no GPU is required.
