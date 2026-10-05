#!/bin/bash
# DS4 prefill-mode arms: does the documented hetero recipe beat the profile we ran?
# A: --ds4-prefill exact (profile default is sparse)
# B: exact + the documented TP env vars (GROUPED_MMVQ, BATCH_SPLIT_COPIES, budget 11700)
set -u
PY=/root/hfenv/bin/python3
OUT=/root/bench/results/ds4-prefill-mode.jsonl
log() { echo "[$(date +%H:%M:%S)] $*" >> /root/bench/prefill-mode.log; }
MODEL=/root/models/ds4/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf
DRAFT=/root/models/ds4/draft/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf

stop() { ps -eo pid,cmd | grep luce_server | grep -v grep | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done; sleep 6; }

wait_up() { for i in $(seq 1 60); do curl -s -m 3 http://127.0.0.1:8902/health | grep -q '"ok"' && { log "up after $((i*5))s"; return 0; }; sleep 5; done; log "NOT UP"; return 1; }

arm_a() {
  stop
  ( cd /root/lucebox && nohup /root/lucebox/server/build-hip/luce_server "$MODEL" \
      --draft "$DRAFT" --profile ds4-r9700-strix --ds4-prefill exact \
      --host 127.0.0.1 --port 8902 > /root/bench/results/ds4-exact.log 2>&1 </dev/null & )
  wait_up && { $PY /root/bench/gen_probe.py --port 8902 --model luce --words 500 --max-tokens 64 --label "exact-2k" --out "$OUT"; \
               $PY /root/bench/gen_probe.py --port 8902 --model luce --words 3000 --max-tokens 64 --label "exact-11k" --out "$OUT"; }
}

arm_b() {
  stop
  ( cd /root/lucebox && nohup env LUCE_EXPERT_BUDGET_MB=11700 LUCE_DS4_TP_BATCH_SPLIT_COPIES=1 LUCE_DS4_TP_GROUPED_MMVQ=1 \
      /root/lucebox/server/build-hip/luce_server "$MODEL" \
      --draft "$DRAFT" --profile ds4-r9700-strix --ds4-prefill exact \
      --host 127.0.0.1 --port 8902 > /root/bench/results/ds4-tpgrouped.log 2>&1 </dev/null & )
  wait_up && { $PY /root/bench/gen_probe.py --port 8902 --model luce --words 500 --max-tokens 64 --label "tpgrouped-2k" --out "$OUT"; \
               $PY /root/bench/gen_probe.py --port 8902 --model luce --words 3000 --max-tokens 64 --label "tpgrouped-11k" --out "$OUT"; }
}

arm_a
arm_b
log "MODES DONE"
cat "$OUT"
