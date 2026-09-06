#!/bin/bash
# Whittle chain: waits for Carnice, converts BF16, runs all 3 configs, restores services.
set -x
LOGDIR=/tmp/coding-bench/logs
mkdir -p "$LOGDIR"

# 1. Wait for Carnice chain (PID as arg 1)
while [ -n "$1" ] && kill -0 "$1" 2>/dev/null; do sleep 60; done
echo "=== CARNICE DONE $(date) — starting Whittle chain ==="

# 2. Convert BF16 safetensors -> GGUF (CPU-bound)
bash /home/caimlas/llm-benchmarks/scripts/convert_whittle.sh > "$LOGDIR/whittle_convert.log" 2>&1
tail -5 "$LOGDIR/whittle_convert.log"

# 3. Free safetensors
rm -rf /home/files/llms/whittle-bf16
df -h /home | tail -1

# 4. Free V100
sudo -n systemctl stop caimlas-bonsai comfyui
sleep 5

cd /home/caimlas/llm-benchmarks

# 5. BF16 native (V100)
echo "=== WHITTLE BF16 RUN $(date) ==="
PYTHONUNBUFFERED=1 /home/caimlas/bench-venv/bin/python scripts/run_whittle.py bf16 > "$LOGDIR/whittle_bf16_run.log" 2>&1
tail -5 "$LOGDIR/whittle_bf16_run.log"

# 6. Q4_K_M (V100) — pass filter that matches only the non-3060 entry
echo "=== WHITTLE Q4KM V100 RUN $(date) ==="
PYTHONUNBUFFERED=1 /home/caimlas/bench-venv/bin/python scripts/run_whittle.py Q4_K_M > "$LOGDIR/whittle_q4km_run.log" 2>&1
tail -5 "$LOGDIR/whittle_q4km_run.log"

# 7. Restore bonsai as user sim for 3060 run
sudo -n systemctl start caimlas-bonsai
sleep 30
for i in $(seq 1 60); do
  curl -s --max-time 3 http://localhost:8081/health | grep -q '"ok"' && break
  sleep 5
done
echo "=== bonsai user sim up ==="

# 8. Q4_K_M (3060)
echo "=== WHITTLE Q4KM 3060 RUN $(date) ==="
PYTHONUNBUFFERED=1 /home/caimlas/bench-venv/bin/python scripts/run_whittle.py 3060 > "$LOGDIR/whittle_3060_run.log" 2>&1
tail -5 "$LOGDIR/whittle_3060_run.log"

# 9. Restore comfyui
sudo -n systemctl start comfyui
echo "=== WHITTLE CHAIN COMPLETE $(date) ==="
curl -s --max-time 3 http://localhost:8081/health && echo " bonsai up"
