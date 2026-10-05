#!/bin/bash
# DS4 prefill knob sweep: does --chunk (and prompt size) explain the gap to lucebox's 788 @2K?
# Arms: (prompt_words, chunk) pairs. One restart per arm.
set -u
PY=/root/hfenv/bin/python3
OUT=/root/bench/results/ds4-prefill-arms.jsonl
log() { echo "[$(date +%H:%M:%S)] $*" >> /root/bench/prefill-sweep.log; }

stop() {
  ps -eo pid,cmd | grep luce_server | grep -v grep | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done
  sleep 6
}

start_chunk() {
  local chunk="$1"
  stop
  ( cd /root/lucebox && nohup /root/bench/serve_ds4flash.sh "$chunk" > /root/bench/results/ds4flash-chunk${chunk}.log 2>&1 </dev/null & )
  for i in $(seq 1 60); do
    if curl -s -m 3 http://127.0.0.1:8902/health | grep -q '"ok"'; then log "chunk $chunk healthy after $((i*5))s"; return 0; fi
    sleep 5
  done
  log "chunk $chunk NOT healthy"; return 1
}

# arm: chunk 2048 (profile default) at a small prompt, for direct comparison with lucebox's 2K
start_chunk 2048 && $PY /root/bench/gen_probe.py --port 8902 --model luce --words 500 --max-tokens 64 --label "chunk2048-~2k"  --out "$OUT"
# arm: chunk 4096 at ~2K
start_chunk 4096 && $PY /root/bench/gen_probe.py --port 8902 --model luce --words 500 --max-tokens 64 --label "chunk4096-~2k"  --out "$OUT"
# arm: chunk 8192 at ~2K (the Strix profile's value)
start_chunk 8192 && $PY /root/bench/gen_probe.py --port 8902 --model luce --words 500 --max-tokens 64 --label "chunk8192-~2k"  --out "$OUT"
# best chunk, longer prompt
start_chunk 8192 && $PY /root/bench/gen_probe.py --port 8902 --model luce --words 3000 --max-tokens 64 --label "chunk8192-~11k" --out "$OUT"
log "SWEEP DONE"
cat "$OUT"
