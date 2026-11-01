#!/usr/bin/env bash
# Build the small development grid scenario: a real SUMO network + demand,
# used for fast iteration and end-to-end runs.
# Requires SUMO_HOME (and $SUMO_HOME/bin on PATH).
set -euo pipefail

if [[ -z "${SUMO_HOME:-}" ]]; then
  echo "ERROR: SUMO_HOME is not set." >&2
  exit 1
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

GRID_NUMBER="${GRID_NUMBER:-6}"     # 6x6 junctions
GRID_LENGTH="${GRID_LENGTH:-150}"   # m between junctions
END="${END:-400}"                   # s of demand
PERIOD="${PERIOD:-0.6}"             # mean s between vehicle insertions (lower => denser)
SEED="${SEED:-42}"

echo "[dev_grid] netgenerate ${GRID_NUMBER}x${GRID_NUMBER} grid"
netgenerate --grid \
  --grid.number "$GRID_NUMBER" \
  --grid.length "$GRID_LENGTH" \
  --default.speed 13.89 \
  --no-turnarounds \
  --tls.guess true \
  --output-file dev_grid.net.xml

echo "[dev_grid] randomTrips -> routes"
python3 "$SUMO_HOME/tools/randomTrips.py" \
  -n dev_grid.net.xml \
  -o dev_grid.trips.xml \
  -r dev_grid.rou.xml \
  --begin 0 --end "$END" --period "$PERIOD" \
  --seed "$SEED" \
  --validate \
  --vehicle-class passenger \
  --min-distance 200 \
  --fringe-factor 5

cat > dev_grid.sumocfg <<XML
<configuration>
  <input>
    <net-file value="dev_grid.net.xml"/>
    <route-files value="dev_grid.rou.xml"/>
  </input>
  <time>
    <begin value="0"/>
    <end value="${END}"/>
  </time>
  <processing>
    <time-to-teleport value="-1"/>
    <collision.action value="warn"/>
  </processing>
  <report>
    <no-step-log value="true"/>
    <duration-log.disable value="true"/>
  </report>
</configuration>
XML

echo "[dev_grid] done -> $HERE/dev_grid.sumocfg"
