#!/bin/bash
# BTL-4 Compact on the Strix Halo iGPU (gfx1151) — the "second model" next to
# resident flashnext (which owns the R9700). Boots on ROCm1 via llama.cpp dual-arch
# build (gfx1151+gfx1201, NO_PEER_COPY=ON). HIP_VISIBLE_DEVICES=1 pins it to the iGPU.
# Flags per the model card: --jinja, reasoning-format deepseek, q8 KV, temp 1.0.
set -u
export HIP_VISIBLE_DEVICES=1
export GGML_CUDA_NO_PEER_COPY=ON
LOG=/root/bench/btl4-server.log
sudo pkill -f "llama-server" 2>/dev/null; sleep 3
setsid nohup /root/llama.cpp-glm/build-hip/bin/llama-server \
  -m /root/models/BTL-4-Compact-IQ2_XXS.gguf \
  -ngl 999 \
  --port 9912 -c 32768 -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --jinja \
  --reasoning-format deepseek \
  --temp 1.0 --top-p 0.95 --top-k 20 \
  > $LOG 2>&1 < /dev/null &
UP=0
for i in $(seq 1 120); do
  curl -s -m 3 http://127.0.0.1:9912/health | grep -q ok && { UP=1; echo "up after $((i*5))s"; break; }
  sleep 5
done
[ $UP = 0 ] && { echo LOAD_FAILED; tail -6 $LOG; exit 1; }
echo "BTL4_UP"

# correctness gate
R=$(curl -s -m 120 http://127.0.0.1:9912/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"btl4","messages":[{"role":"user","content":"What is the capital of France? Answer with just the city name."}],"max_tokens":32,"temperature":0}')
echo "gate: $(echo "$R" | grep -oE '"content":"[^"]*"' | head -1 | cut -c13-50)"

# decode: 3x 256 tokens
for i in 1 2 3; do
  curl -s -m 300 http://127.0.0.1:9912/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"btl4","messages":[{"role":"user","content":"Write a detailed plan for a weekend hiking trip."}],"max_tokens":256,"temperature":0.7}' \
    > /tmp/btl4-gen$i.json
  python3 -c "
import json
d = json.load(open('/tmp/btl4-gen$i.json'))
u = (d.get('usage') or {}).get('timings') or {}
print('server decode:', u.get('predicted_per_second'), 'tok/s')"
done
echo BTL4_DONE
