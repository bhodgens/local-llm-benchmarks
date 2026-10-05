#!/bin/bash
# Kolibri-1: does moving some expert layers onto the R9700 beat all-CPU experts?
# --n-cpu-moe N keeps the MoE tensors of N layers on the CPU; a smaller N puts more
# experts on the GPU. The model is 47.5 GB against 32 GB of VRAM, so only part fits.
set -u
PY=/root/hfenv/bin/python3
OUT=/root/bench/results/kolibri-ncpumoe.jsonl
LOG=/root/bench/kolibri-ncpumoe.log
: > "$OUT"; : > "$LOG"
log() { echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

stop() { ps -eo pid,cmd | grep -E "[l]lama.cpp-kolibri/build-hip" | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done; sleep 5; }

wait_up() { for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8902/health | grep -q '"ok"' && { log "up in $((i*5))s"; return 0; }; sleep 5; done; log "NOT UP"; return 1; }

for NCM in 999 40 28 16; do
  stop
  ( setsid nohup /root/bench/serve_kolibri.sh "$NCM" > "/root/bench/results/kolibri-ncm$NCM.log" 2>&1 < /dev/null & )
  wait_up || { echo "{\"n_cpu_moe\":$NCM,\"error\":\"not up\"}" >> "$OUT"; continue; }
  # short-prompt decode probe (256 out) and a ~2.5K prompt for prefill
  $PY /root/bench/gen_probe.py --port 8902 --model Kolibri-1-Q4_K_M.gguf \
      --words 30 --max-tokens 256 --label "ncm$NCM-decode" --out "$OUT"
  $PY /root/bench/gen_probe.py --port 8902 --model Kolibri-1-Q4_K_M.gguf \
      --words 660 --max-tokens 32 --label "ncm$NCM-prefill" --out "$OUT"
done
stop
log "SWEEP DONE"
cat "$OUT" >> "$LOG"
touch /root/bench/KOLIBRI_SWEEP_DONE
