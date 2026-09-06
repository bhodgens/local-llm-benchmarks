#!/bin/bash
# Final corrected Whittle sequence
set -x
cd /home/caimlas/llm-benchmarks

echo "=== WHITTLE Q4KM V100 REDO2 $(date) ==="
PYTHONUNBUFFERED=1 /home/caimlas/bench-venv/bin/python scripts/run_whittle.py Q4_K_M > /tmp/coding-bench/logs/whittle_q4km_redo2.log 2>&1
tail -4 /tmp/coding-bench/logs/whittle_q4km_redo2.log

echo "=== WHITTLE BF16 CONVERT+RUN $(date) ==="
bash /home/caimlas/llm-benchmarks/scripts/convert_whittle.sh > /tmp/coding-bench/logs/whittle_convert2.log 2>&1
tail -3 /tmp/coding-bench/logs/whittle_convert2.log

if [ -s /home/files/llms/Whittle-MoE-27B-A18B-v2.1-BF16.gguf ]; then
  rm -rf /home/files/llms/whittle-bf16
  echo "safetensors freed"
  PYTHONUNBUFFERED=1 /home/caimlas/bench-venv/bin/python scripts/run_whittle.py bf16 > /tmp/coding-bench/logs/whittle_bf16_run2.log 2>&1
  tail -4 /tmp/coding-bench/logs/whittle_bf16_run2.log
fi

echo "=== LFM user sim on 8081 for 3060 run ==="
CUDA_VISIBLE_DEVICES=1 /home/caimlas/git/llama.cpp/build/bin/llama-server \
  --model /home/files/llms/LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf \
  --flash-attn on --host 0.0.0.0 --port 8081 --gpu-layers 99 --ctx-size 131072 \
  --ubatch-size 512 --threads 4 --threads-batch 4 \
  --cache-type-k q8_0 --cache-type-v q8_0 --parallel 2 --temp 0.0 -n 4096 &
USIM=$!
until curl -s --max-time 3 http://127.0.0.1:8081/health | grep -q '"ok"'; do sleep 5; done

echo "=== WHITTLE 3060 RUN2 $(date) ==="
PYTHONUNBUFFERED=1 /home/caimlas/bench-venv/bin/python scripts/run_whittle.py 3060 > /tmp/coding-bench/logs/whittle_3060_run2.log 2>&1
tail -4 /tmp/coding-bench/logs/whittle_3060_run2.log

kill $USIM 2>/dev/null
sleep 3
sudo -n systemctl start caimlas-bonsai comfyui
sleep 25
curl -s --max-time 3 http://localhost:8081/health && echo " === services restored ==="
