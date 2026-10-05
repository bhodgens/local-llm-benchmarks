#!/bin/bash
# PLE I/O arm sweep for Strata on borg: direct (SSD, default) vs mmap (page cache) vs ram (locked).
# Each arm: restart Strata, wait healthy, cold-prefill probe at ~8K tokens, record, stop.
set -u
PY=/root/hfenv/bin/python3
CFG=/root/strata/strata-iq3_s.json
OUT=/root/bench/results/ple-arms.jsonl
: > "$OUT"

stop_strata() {
  pkill -f "strata/serve/server.py" 2>/dev/null
  pkill -f "engine/strata" 2>/dev/null
  sleep 6
}

set_ple_io() {
  local mode="$1"
  $PY - "$CFG" "$mode" <<'PYEOF'
import json, sys
cfg_path, mode = sys.argv[1], sys.argv[2]
cfg = json.load(open(cfg_path))
args = [a for a in cfg["args"] if a != "--ple-io"]
i = args.index("--ple-io")+1 if "--ple-io" in args else None
args = [a for a in args if a not in ("direct","mmap","ram") or True]
# remove any stale ple-io value token
clean = []
skip = False
for a in args:
    if skip:
        skip = False
        continue
    if a == "--ple-io":
        skip = True
        continue
    if a in ("direct", "mmap", "ram"):
        continue
    clean.append(a)
clean += ["--ple-io", mode]
cfg["args"] = clean
json.dump(cfg, open(cfg_path, "w"), indent=1)
print("ple-io set to", mode)
PYEOF
}

wait_health() {
  for i in $(seq 1 90); do
    if curl -s -m 3 http://127.0.0.1:8080/health | grep -q '"ok"'; then echo "healthy after $((i*5))s"; return 0; fi
    sleep 5
  done
  echo "NOT HEALTHY"; return 1
}

for MODE in direct mmap ram; do
  echo "=== arm: $MODE ==="
  stop_strata
  set_ple_io "$MODE"
  ( cd /root/strata && nohup ./run-iq3_s.sh > /root/bench/results/strata-$MODE.log 2>&1 </dev/null & )
  if ! wait_health; then
    echo "{\"label\":\"$MODE\",\"error\":\"server not healthy\"}" >> "$OUT"
    continue
  fi
  # cold prefill probe at ~8K, then a second at ~32K
  $PY /root/bench/prefill_probe.py --port 8080 --tokens 8000  --label "$MODE-8k"  >> "$OUT" 2>&1
  $PY /root/bench/prefill_probe.py --port 8080 --tokens 32000 --label "$MODE-32k" >> "$OUT" 2>&1
done
stop_strata
echo "ARMS DONE"
cat "$OUT"
