#!/usr/bin/env bash
# Fetch the VeReMi dataset (van der Heijden et al., 2018) into data/veremi/original/.
# The data ship as GitHub release tarballs.
#
# These are large external downloads; this script only retrieves the published
# data. No data is generated here.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$HERE/data/veremi"
mkdir -p "$DEST/original"

echo "VeReMi original:"
echo "  Repository: https://github.com/VeReMi-dataset/VeReMi"
echo "  Releases (per-density tarballs): https://github.com/VeReMi-dataset/VeReMi/releases"
echo

if command -v git >/dev/null 2>&1 && [[ ! -d "$DEST/original/VeReMi" ]]; then
  echo "[veremi] cloning original repo metadata (data ships as release assets)…"
  git clone --depth 1 https://github.com/VeReMi-dataset/VeReMi.git "$DEST/original/VeReMi" || true
fi

cat <<'NOTE'

ACTION REQUIRED (one-time, large files):
  1) Download one or more VeReMi *original* release tarballs from the Releases page
     and extract them under data/veremi/original/ so each simulation directory
     (containing a GroundTruth*.json) is discoverable.

Then run:
  python -m qkv2x.cli experiment --config configs/experiments/veremi.yaml
The loader (qkv2x.data.veremi) auto-discovers every simulation directory that
contains a GroundTruth file under data/veremi/.
NOTE
