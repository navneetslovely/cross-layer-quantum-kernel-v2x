#!/usr/bin/env bash
# Fetch CIC-IoV-2024 (CICIoV2024) — Canadian Institute for Cybersecurity, UNB.
# IoV / CAN-bus IDS collected on a REAL 2019 Ford vehicle: DoS + 4 spoofing
# variants (RPM, SPEED, STEERING_WHEEL, GAS) vs benign.
#
#  Official page:  https://www.unb.ca/cic/datasets/iov-dataset-2024.html
#  CIC index/terms: https://www.unb.ca/cic/datasets/index.html
#  Cite (required): Neto, Taslimasa, Dadkhah, Iqbal, Xiong, Rahman, Ghorbani,
#     "CICIoV2024: Advancing realistic IDS approaches against DoS and spoofing
#  attack in IoV CAN bus," Internet of Things 26:101209 (2024).
#  DOI 10.1016/j.iot.2024.101209.
#
# ACCESS = FORM-GATED. CIC datasets are FREE with no login/password, but each
# dataset page has a download button at the bottom that opens a short data-collection
# form; you complete the form to reach the files. There is NO confirmed open-directory
# mirror for IoV-2024 (unlike CIC-IoT-2023). So this script CANNOT auto-download it —
# it prints precise instructions and exits 0 (it does not fail silently).
#
# Format once downloaded: CSV in three numeric encodings (decimal / hexadecimal /
# binary subfolders). Columns: ID, DATA_0..DATA_7, label (binary), category
# (Benign/DoS/Spoofing), specific_class (6-class). ~1.41M rows; tens of MB.
# Place the extracted CSVs under data/cic_iov_2024/.
set -u
cd "$(dirname "$0")/../.."
DEST="data/cic_iov_2024"
mkdir -p "$DEST"

PAGE="https://www.unb.ca/cic/datasets/iov-dataset-2024.html"

# If the user has already placed the data, just confirm and stop.
if [ -n "$(find "$DEST" -type f \( -name '*.csv' -o -name '*.zip' \) 2>/dev/null | head -1)" ]; then
  echo "have  CIC-IoV-2024 files already present under $DEST/:"
  find "$DEST" -type f \( -name '*.csv' -o -name '*.zip' \) 2>/dev/null | sed 's/^/   /'
  echo "done -> $DEST  (cite: Neto et al., Internet of Things 2024, DOI 10.1016/j.iot.2024.101209)"
  exit 0
fi

cat <<EOF

================ MANUAL STEP REQUIRED (CIC-IoV-2024) ====================
CIC-IoV-2024 is FREE but FORM-GATED — it cannot be scripted without first
completing the CIC download form. Do this once:

  1. Open:  $PAGE
  2. Scroll to the bottom and click the red DOWNLOAD button.
  3. Fill the short data-collection form (no fee, no account/password).
  4. Download the dataset archive (CSV; decimal / hexadecimal / binary folders).
  5. Extract it into:
        $(pwd)/$DEST/
     (keep the decimal/ hexadecimal/ binary/ subfolders as provided).
  6. Re-run this script to verify the files are present.

  Schema: ID, DATA_0..DATA_7, label, category, specific_class.
  Classes: Benign, DoS, Spoofing-RPM, Spoofing-SPEED,
           Spoofing-STEERING_WHEEL, Spoofing-GAS.
  License/terms (CIC): redistribution allowed WITH citation of the dataset and
    the paper — Neto et al., Internet of Things 26:101209 (2024),
    DOI 10.1016/j.iot.2024.101209.

  (Optional, non-authoritative) third-party mirrors exist on Kaggle/GitHub if you
  accept non-official provenance; the UNB form above is the canonical source.
========================================================================
EOF
exit 0
