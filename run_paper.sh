#!/usr/bin/env bash
# Regenerates the data figures of the paper (Figs. 2-4) from full_paper_data/ and results/ (no datasets needed).
set -euo pipefail
cd "$(dirname "$0")"
python scripts/render_figures.py
python scripts/studies/fig_crossdataset.py
echo "done: figures are in figures/"
