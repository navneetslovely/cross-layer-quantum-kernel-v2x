#!/usr/bin/env bash
# Fetch the ROAD (Real ORNL Automotive Dynamometer) CAN Intrusion Dataset.
#
#  Source (official, OPEN):  https://zenodo.org/records/10462796
#  Paper: Verma et al., "A comprehensive guide to CAN IDS data and introduction
#  of the ROAD dataset," PLOS ONE 19(1):e0296879 (2024). arXiv:2012.14600.
#  License: CC BY 4.0 (attribution required).
#
# Access: PUBLIC DIRECT DOWNLOAD — no form, no email, no login. A single
#  road.zip (556,718,276 bytes ≈ 531 MiB). Contains raw can-utils candump *.log
#  captures plus CAN-D signal-translated CSVs (Label,Time,ID,Signal_*), 33 attack
#  captures (~30 min) + 12 ambient captures (~3 h) = ~3.5 h of one vehicle's CAN.
#
# Style mirrors scripts/fetch_veremi_nextgen.sh: idempotent (skips an already-good
# file), resume-safe (curl -C -), verifies the published MD5, and unzips in place.
# Writes under data/road/ (gitignored). Re-run on failure.
set -u
cd "$(dirname "$0")/../.."          # repo root  (scripts/datasets/ -> ../../)
DEST="data/road"
DL="$DEST/dl"
mkdir -p "$DL"

URL="https://zenodo.org/records/10462796/files/road.zip?download=1"
FILE="$DL/road.zip"
WANT_MD5="cab184cfc2fe12c0834bc46188c0f330"
WANT_BYTES="556718276"

md5_of() {  # portable md5 (macOS `md5 -q` / Linux `md5sum`)
  if command -v md5sum >/dev/null 2>&1; then md5sum "$1" | awk '{print $1}';
  elif command -v md5 >/dev/null 2>&1; then md5 -q "$1";
  else echo ""; fi
}

# ---- already have a verified copy? ------------------------------------------
if [ -s "$FILE" ]; then
  have_md5="$(md5_of "$FILE")"
  if [ -n "$have_md5" ] && [ "$have_md5" = "$WANT_MD5" ]; then
    echo "have (md5 ok)  $FILE"
  elif [ -z "$have_md5" ]; then
    echo "have (md5 tool missing, not re-verified)  $FILE"
  else
    echo "checksum mismatch on existing $FILE (got $have_md5) -> re-downloading"
    rm -f "$FILE"
  fi
fi

# ---- download ----------------------------------------------------------------
if [ ! -s "$FILE" ]; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "[fetch] curl not found. Download manually:"
    echo "          $URL"
    echo "  and place road.zip at $FILE"
    exit 0
  fi
  echo "fetch road.zip (~531 MiB) -> $FILE"
  # NOTE: do NOT spoof a browser User-Agent here — Zenodo flags spoofed UAs as bot
  # traffic and returns 403 "unusual traffic". curl's default UA downloads fine.
  curl -L -C - --retry 5 --connect-timeout 30 --speed-limit 10240 --speed-time 30 \
       -o "$FILE.part" "$URL" && mv "$FILE.part" "$FILE"
  if [ ! -s "$FILE" ]; then
    echo "[fetch] download did not complete. Re-run this script (it resumes)."
    echo "  Or download in a browser: $URL"
    exit 0
  fi
fi

# ---- verify ------------------------------------------------------------------
got_md5="$(md5_of "$FILE")"
got_bytes="$(wc -c < "$FILE" | tr -d ' ')"
if [ -n "$got_md5" ] && [ "$got_md5" != "$WANT_MD5" ]; then
  echo "[fetch] MD5 MISMATCH: got $got_md5 want $WANT_MD5 — file is corrupt; delete $FILE and re-run."
  exit 1
fi
[ "$got_bytes" = "$WANT_BYTES" ] || echo "[fetch] WARN size $got_bytes != expected $WANT_BYTES"
echo "[fetch] verified road.zip (md5 ${got_md5:-skipped}, $got_bytes bytes)"

# ---- unzip (idempotent) ------------------------------------------------------
if command -v unzip >/dev/null 2>&1; then
  if [ -z "$(find "$DEST" -mindepth 1 -maxdepth 2 -name '*.log' -o -name '*.csv' 2>/dev/null | head -1)" ]; then
    echo "[fetch] unzipping into $DEST/"
    unzip -q -o "$FILE" -d "$DEST"
  else
    echo "[fetch] already unzipped under $DEST/ (found .log/.csv) — skipping unzip"
  fi
else
  echo "[fetch] unzip not found — leaving archive at $FILE (extract manually into $DEST/)"
fi

echo "done -> $DEST  (cite: Verma et al., PLOS ONE 2024; DOI 10.5281/zenodo.10462796; CC BY 4.0)"
