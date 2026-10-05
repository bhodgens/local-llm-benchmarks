#!/bin/bash
# Qwen 3.8 27B kvflash: controlled 3-arm comparison.
#   arm 1 control   : no kvflash, ctx 131072, f16 KV, block 16   (matches the Phase 1 ledger)
#   arm 2 kvflash   : --kvflash auto, ctx 131072
#   arm 3 kvflash256: --kvflash 8192, ctx 262144                 (does a bounded pool hold speed at depth?)
# Each arm runs: gen_probe @ ~2K and ~11K, then parity.py (HumanEval-10 greedy => the 228.7 ledger).
set -u
PY=/root/hfenv/bin/python3
OUT=/root/bench/results/qwen-kvflash-control.jsonl
log() { echo "[$(date +%H:%M:%S)] $*" >> /root/bench/kvflash-ctl.log; }

stop() { ps -eo pid,cmd | grep -E "[l]uce_server" | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done; sleep 5; }

wait_up() { for i in $(seq 1 72); do curl -s -m 3 http://127.0.0.1:8901/health | grep -q '"ok"' && { log "up after $((i*5))s"; return 0; }; sleep 5; done; log "NOT UP"; return 1; }

run_arm() {
  local tag="$1" script="$2"; shift 2
  stop
  ( cd /root/lucebox && nohup "$script" "$@" > "/root/bench/results/qwen-$tag.log" 2>&1 </dev/null & )
  wait_up || return 1
  $PY /root/bench/gen_probe.py --port 8901 --model luce --words 500  --max-tokens 256 --label "$tag-2k"  --out "$OUT"
  $PY /root/bench/gen_probe.py --port 8901 --model luce --words 3000 --max-tokens 256 --label "$tag-11k" --out "$OUT"
  $PY /root/bench/parity.py --port 8901 --max-tokens 256 --warmup 1 --out "/root/bench/results/qwen-parity-$tag.json" 2>&1 | tail -1
}

# arm 1: control (no kvflash) - uses the original Phase 1 serve script
run_arm "control-131k" /root/bench/serve_qwen3827b.sh f16 16
# arm 2: kvflash auto at 131K
run_arm "kvflash-131k" /root/bench/serve_qwen3827b_kvflash.sh 131072 auto
# arm 3: kvflash pool at 262K, plus a long prompt
run_arm "kvflash-262k" /root/bench/serve_qwen3827b_kvflash.sh 262144 8192
$PY /root/bench/gen_probe.py --port 8901 --model luce --words 12000 --max-tokens 128 --label "kvflash-262k-45k" --out "$OUT"

log "CTL DONE"
cat "$OUT"
