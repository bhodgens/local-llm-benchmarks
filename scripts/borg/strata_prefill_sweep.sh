#!/bin/bash
# Does Strata's short-prompt prefill improve with a bigger prompt chunk?
# Arm A: current config (--prefill auto => chunks up to 8192)
# Arm B: STRATA_PREFILL_AUTO_MAX=32768 (chunks up to 32768)
# Each arm measures COLD prefill at ~0.5K / 2.5K / 8K / 29K tokens (unique text each).
set -u
PY=/root/hfenv/bin/python3
OUT=/root/bench/results/strata-prefill-sweep.jsonl
LOG=/root/bench/strata-prefill-sweep.log
: > "$OUT"; : > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

stop() { ps -eo pid,cmd | grep -E "[e]ngine/strata|[s]erve/server.py" | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done; sleep 6; }

wait_up() { for i in $(seq 1 90); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok"' && { log "up in $((i*5))s"; return 0; }; sleep 5; done; log "NOT UP"; return 1; }

run_arm() {
  local tag="$1"; shift
  stop
  ( cd /root/strata && env "$@" setsid nohup ./run-iq3_s.sh > "/root/bench/results/strata-$tag.log" 2>&1 < /dev/null & )
  wait_up || return 1
  for W in 130 660 2100 7600; do
    $PY /root/bench/gen_probe.py --port 8080 --model qwen3.8-flash-next-iq3_s \
        --words "$W" --max-tokens 32 --reasoning-off --label "$tag-$W" --out "$OUT"
  done
}

run_arm "chunk8192"  STRATA_PREFILL_AUTO_MAX=8192
run_arm "chunk32768" STRATA_PREFILL_AUTO_MAX=32768
stop
log "SWEEP DONE"
cat "$OUT" >> "$LOG"
