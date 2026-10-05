#!/bin/bash
# Find the lowest --n-cpu-moe that still loads (more GPU experts = faster, until OOM).
set -u
PY=/root/hfenv/bin/python3
OUT=/root/bench/results/kolibri-ncm-limit.jsonl
LOG=/root/bench/kolibri-ncm-limit.log
: > "$OUT"; : > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
stop() { ps -eo pid,cmd | grep -E "[l]lama.cpp-kolibri/build-hip" | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done; sleep 5; }
wait_up() { for i in $(seq 1 48); do curl -s -m 3 http://127.0.0.1:8902/health | grep -q '"ok"' && return 0; sleep 5; done; return 1; }

for NCM in 26 24 22; do
  stop
  ( setsid nohup /root/bench/serve_kolibri.sh "$NCM" > "/root/bench/results/kolibri-ncm$NCM.log" 2>&1 < /dev/null & )
  if wait_up; then
    VRAM=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "GPU\[0\]" -A1 | grep Used | head -1 | awk '{print $NF}')
    $PY /root/bench/gen_probe.py --port 8902 --model Kolibri-1-Q4_K_M.gguf \
        --words 30 --max-tokens 256 --label "ncm$NCM-decode" --out "$OUT"
    echo "{\"n_cpu_moe\": $NCM, \"vram_used_bytes\": \"$VRAM\"}" >> "$OUT"
  else
    ERR=$(grep -iE "out of memory|failed to allocate|error" "/root/bench/results/kolibri-ncm$NCM.log" | tail -1 | cut -c1-120)
    echo "{\"n_cpu_moe\": $NCM, \"load_failed\": true, \"error\": \"$ERR\"}" >> "$OUT"
  fi
done
stop
log "LIMIT SWEEP DONE"
cat "$OUT" >> "$LOG"
touch /root/bench/KOLIBRI_LIMIT_DONE
