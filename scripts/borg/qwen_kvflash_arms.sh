#!/bin/bash
# Qwen 3.8 27B KVFlash arms.
# A: ctx 131072 + --kvflash auto  -> does the pool cost anything at our normal setting?
# B: ctx 262144 + --kvflash 8192  -> does a bounded pool let a 256K context run fast (Laguna-style)?
set -u
PY=/root/hfenv/bin/python3
OUT=/root/bench/results/qwen-kvflash.jsonl
log() { echo "[$(date +%H:%M:%S)] $*" >> /root/bench/kvflash.log; }

stop() { ps -eo pid,cmd | grep -E "[l]uce_server" | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done; sleep 5; }

wait_up() {
  for i in $(seq 1 72); do
    curl -s -m 3 http://127.0.0.1:8901/health | grep -q '"ok"' && { log "up after $((i*5))s"; return 0; }
    sleep 5
  done
  log "NOT UP"; return 1
}

arm() {
  local ctx="$1" kvf="$2" tag="$3"
  stop
  ( cd /root/lucebox && nohup /root/bench/serve_qwen3827b_kvflash.sh "$ctx" "$kvf" \
      > "/root/bench/results/qwen-kvflash-$tag.log" 2>&1 </dev/null & )
  wait_up || return 1
  # small-prompt decode + prefill (comparable to the Phase 1 baseline)
  $PY /root/bench/gen_probe.py --port 8901 --model luce --words 500 --max-tokens 256 --label "$tag-2k" --out "$OUT"
  # long-ish prompt
  $PY /root/bench/gen_probe.py --port 8901 --model luce --words 3000 --max-tokens 256 --label "$tag-11k" --out "$OUT"
  if [ "$ctx" = "262144" ]; then
    # push a genuinely long prompt at the 256K config
    $PY /root/bench/gen_probe.py --port 8901 --model luce --words 12000 --max-tokens 128 --label "$tag-45k" --out "$OUT"
  fi
}

arm 131072 auto kvflash-auto-131k
arm 262144 8192 kvflash-pool-262k
log "KVFLASH ARMS DONE"
cat "$OUT"
