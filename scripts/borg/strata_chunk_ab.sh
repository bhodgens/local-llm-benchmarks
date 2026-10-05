#!/bin/bash
# A/B: Strata prompt-chunk cap vs decode speed. Bigger chunks lend expert-cache
# memory to prefill buffers ("auto" only goes as large as the cache can lend),
# so this checks whether the prefill gain costs decode.
set -u
PY=/root/hfenv/bin/python3
OUT=/root/bench/results/strata-chunk-ab.jsonl
: > "$OUT"
stop() { ps -eo pid,cmd | grep -E "[e]ngine/strata|[s]erve/server.py" | awk '{print $1}' | while read p; do kill "$p" 2>/dev/null; done; sleep 6; }
wait_up() { for i in $(seq 1 72); do curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok"' && return 0; sleep 5; done; return 1; }

for CAP in 8192 32768; do
  stop
  ( setsid nohup env STRATA_PREFILL_AUTO_MAX=$CAP /root/strata/run-iq3_s.sh \
      > "/root/bench/results/strata-cap$CAP.log" 2>&1 < /dev/null & )
  wait_up || { echo "{\"cap\":$CAP,\"error\":\"not up\"}" >> "$OUT"; continue; }
  # decode probe: short unique prompt, 256 tokens out, three times
  for i in 1 2 3; do
    $PY /root/bench/gen_probe.py --port 8080 --model qwen3.8-flash-next-iq3_s \
        --words 20 --max-tokens 256 --reasoning-off --label "cap$CAP-dec$i" --out "$OUT"
  done
  # prefill probe at ~2.5K
  $PY /root/bench/gen_probe.py --port 8080 --model qwen3.8-flash-next-iq3_s \
      --words 660 --max-tokens 32 --reasoning-off --label "cap$CAP-prefill" --out "$OUT"
  # long prefill at ~28K to see the big-chunk win
  $PY /root/bench/gen_probe.py --port 8080 --model qwen3.8-flash-next-iq3_s \
      --words 7600 --max-tokens 32 --reasoning-off --label "cap$CAP-prefill28k" --out "$OUT"
done
stop
touch /root/bench/STRATA_AB_DONE
