#!/bin/bash
# Re-run every registry model through the corrected lane (warmup request discarded,
# first-request prefill recorded separately). Serialized: one lane at a time.
set -u
exec 9>/root/bench/.lane.lock
flock -n 9 || { echo "another lane holds the lock"; exit 1; }
PY=/root/hfenv/bin/python3
{
  echo "[$(date +%H:%M:%S)] full re-run start"
  for M in kolibri-1 qwen38-27b qwen38-27b-vision laguna-xs21 ds4-flash ds41-flash glm53-flash flashnext; do
    echo "[$(date +%H:%M:%S)] --- $M ---"
    "$PY" /root/bench/bench_borg_lanes.py --models "$M" --n 256
  done
  echo "[$(date +%H:%M:%S)] full re-run done"
  "$PY" /root/bench/bench_borg_lanes.py --table
} > /root/bench/lane-full-rerun.out 2>&1
touch /root/bench/FULL_RERUN_DONE
