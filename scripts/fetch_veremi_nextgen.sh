#!/bin/bash
# Fetch the VeReMi NextGen subset used by configs/experiments/veremi_nextgen.yaml
#. Source: Zenodo record 19665762 (Hermann et al., VNC 2026).
# Downloads into data/veremi_nextgen/dl/ (gitignored). ~1.7 GB for the default
# scenario/attack subset. Resume-safe (curl -C -); re-run on failure.
set -u
cd "$(dirname "$0")/.."
mkdir -p data/veremi_nextgen/dl
cd data/veremi_nextgen/dl
base="https://zenodo.org/records/19665762/files"

scenarios="InTAS_highway_2 InTAS_urban_2"
attacks="constantPositionOffset randomPositionOffset trafficCongestionSybil \
dosAttack randomSpeedOffset dataReplay timeDelayAttack"

for sc in $scenarios; do
  files="${sc}.zip"
  for a in $attacks; do files="$files ${sc}_${a}.zip"; done
  for f in $files; do
    [ -s "$f" ] && { echo "have  $f"; continue; }
    echo "fetch $f"
    curl -L -C - --retry 5 --speed-limit 10240 --speed-time 30 \
         -o "$f.part" "$base/$f?download=1" && mv "$f.part" "$f"
  done
done
echo "done -> $(pwd)"
