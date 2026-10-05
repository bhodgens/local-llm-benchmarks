#!/bin/bash
# Serve Kolibri-1 (Aleph Alpha 78B MoE) on borg with the community-patched llama.cpp.
# Architecture 'kolibri1' is not in stock llama.cpp: this binary is the patched build
# (git apply --3way of kolibri1-llama.cpp.patch onto 836d571; verified in libllama.so).
# 47.5 GB Q4_K_M: more than the R9700's 32 GB, so attention runs on the GPU with
# expert tensors offloaded to the Strix Halo's memory via --n-cpu-moe.
# Override the expert-offload count with $1 (default 999 = all experts on CPU).
set -u
NCPUMOE="${1:-999}"
EXTRA=()
[ -n "$NCPUMOE" ] && EXTRA=(--n-cpu-moe "$NCPUMOE")
exec /root/llama.cpp-kolibri/build-hip/bin/llama-server \
  -m /root/models/kolibri/Kolibri-1-Q4_K_M.gguf \
  --host 127.0.0.1 --port 8902 \
  --ctx-size 32768 \
  --n-gpu-layers 999 \
  "${EXTRA[@]}" \
  --jinja \
  --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --temp 1.0 --top-p 0.97 --top-k 128 \
  --threads 16
