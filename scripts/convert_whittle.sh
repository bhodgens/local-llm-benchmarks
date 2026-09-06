#!/bin/bash
set -e
echo "=== Converting Whittle BF16 safetensors -> GGUF $(date) ==="
cd /home/caimlas/git/llama.cpp
/home/caimlas/bench-venv/bin/python convert_hf_to_gguf.py /home/files/llms/whittle-bf16 \
  --outtype bf16 --no-mtp --outfile /home/files/llms/Whittle-MoE-27B-A18B-v2.1-BF16.gguf
echo "=== verify ==="
timeout 120 python3 /tmp/gguf_mtp_check.py /home/files/llms/Whittle-MoE-27B-A18B-v2.1-BF16.gguf 2>/dev/null | grep -E 'arch|block_count|expert' || true
ls -lh /home/files/llms/Whittle-MoE-27B-A18B-v2.1-BF16.gguf
echo "=== CONVERSION DONE $(date) ==="
