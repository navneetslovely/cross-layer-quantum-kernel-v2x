#!/usr/bin/env bash
# Fetch the HCRL Car-Hacking Dataset (Korea University, Hacking & Countermeasure
# Research Lab — Prof. Huy Kang Kim). CAN-bus IDS: DoS / Fuzzy / Spoofing(gear,RPM)
# message-injection attacks + an attack-free baseline.
#
#  Official page:  https://ocslab.hksecurity.net/Datasets/car-hacking-dataset
#  Lab index:  https://ocslab.hksecurity.net/Datasets
#  Cite (required by the lab): Seo, Song, Kim, "GIDS: GAN based Intrusion Detection
#  System for In-Vehicle Network," IEEE PST 2018. DOI 10.1109/PST.2018.8514157.
#
# Access: PUBLIC for academic use, but the lab asks that you CITE the paper, and the
#  download links live on the dataset page (they are sometimes served via on-page /
#  redirected links rather than a stable flat URL). This script ATTEMPTS the direct
#  file URLs; if the host layout has changed it prints precise manual instructions
#  and exits 0 (it never fails silently and never fabricates a file).
#
# Files (CSV; cols: Timestamp,CAN ID,DLC,DATA[0..7],Flag  where Flag R=normal T=attack):
#  DoS_dataset.csv  Fuzzy_dataset.csv  gear_dataset.csv  RPM_dataset.csv
#  normal_run_data.txt  (attack-free baseline, no Flag column)
# Total ≈ 1.5–2.5 GB uncompressed (~17.5M CAN frames). Writes under data/hcrl_carhacking/.
set -u
cd "$(dirname "$0")/../.."
DEST="data/hcrl_carhacking"
mkdir -p "$DEST"

PAGE="https://ocslab.hksecurity.net/Datasets/car-hacking-dataset"
# Best-known direct file locations on the HCRL host (the page links to these names).
BASE="https://ocslab.hksecurity.net/sites/default/files"
FILES="DoS_dataset.csv Fuzzy_dataset.csv gear_dataset.csv RPM_dataset.csv normal_run_data.txt"

manual_note() {
  cat <<EOF

================ MANUAL STEP REQUIRED (HCRL Car-Hacking) ================
The automatic download did not retrieve all files (the HCRL page may serve them
via on-page links rather than the flat URL this script tried). Do this once:

  1. Open:  $PAGE
  2. Download these files from the links on that page:
        DoS_dataset.csv  Fuzzy_dataset.csv  gear_dataset.csv
        RPM_dataset.csv  normal_run_data.txt
  3. Place them (uncompressing any .zip) in:
        $(pwd)/$DEST/
  4. Re-run this script to verify they are present.

  License/terms: public for academic use; you MUST cite
    Seo, Song, Kim, "GIDS," IEEE PST 2018 (DOI 10.1109/PST.2018.8514157).
========================================================================
EOF
}

got_all=1
if command -v curl >/dev/null 2>&1; then
  for f in $FILES; do
    out="$DEST/$f"
    if [ -s "$out" ]; then echo "have  $f"; continue; fi
    echo "try   $f"
    # -f: fail (non-2xx) without writing an HTML error page as the "data".
    if curl -fL -C - --retry 3 --connect-timeout 30 -A "Mozilla/5.0" \
            -o "$out.part" "$BASE/$f" && [ -s "$out.part" ]; then
      mv "$out.part" "$out"; echo "  ok  $f ($(wc -c < "$out" | tr -d ' ') bytes)"
    else
      rm -f "$out.part"; echo "  miss $f (not at $BASE/$f)"; got_all=0
    fi
  done
else
  echo "[fetch] curl not found — manual download required."; got_all=0
fi

if [ "$got_all" -eq 1 ]; then
  echo "done -> $DEST  (cite: Seo, Song, Kim, IEEE PST 2018, DOI 10.1109/PST.2018.8514157)"
else
  manual_note
fi
exit 0
