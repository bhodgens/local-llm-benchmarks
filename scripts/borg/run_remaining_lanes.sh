#!/bin/bash
# Run the remaining borg lanes STRICTLY one at a time (a competing lane's
# stop_servers() kills the other's server, which is what corrupted an earlier run).
set -u
exec 9>/root/bench/.lane.lock
flock -n 9 || { echo "another lane is running; refusing"; exit 1; }

PY=/root/hfenv/bin/python3
LOG=/root/bench/lanes-remaining.log
: > "$LOG"

for M in ds41-flash glm53-flash flashnext; do
  echo "[$(date +%H:%M:%S)] === $M ===" >> "$LOG"
  "$PY" /root/bench/bench_borg_lanes.py --models "$M" --n 256 >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] === $M done rc=$? ===" >> "$LOG"
done

"$PY" /root/bench/bench_borg_lanes.py --table >> "$LOG" 2>&1
touch /root/bench/LANES_DONE
echo "ALL LANES DONE" >> "$LOG"
