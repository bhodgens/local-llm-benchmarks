#!/bin/bash
# Convert AEON ULTIMATE UNCENSORED BF16 -> GGUF -> Q4_K_M, then clean up.
# Runs on CPU (conversion is RAM-bound). ~55GB model needs ~60GB RAM headroom.
set -e
SRC=/home/files/llms/aeon-uncensored-bf16
LLMS=/home/files/llms
PY=/home/caimlas/bench-venv/bin/python
CONVERTER=/home/caimlas/git/llama.cpp/convert_hf_to_gguf.py
QUANTIZER=/home/caimlas/git/llama.cpp/build/bin/llama-quantize

echo "=== 1. Convert to BF16 GGUF (text-only, keep MTP head) $(date) ==="
cd /home/caimlas/git/llama.cpp
$PY $CONVERTER "$SRC" --outtype bf16 --outfile "$LLMS/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16.gguf"

echo "=== 2. Verify GGUF metadata $(date) ==="
timeout 120 python3 /tmp/gguf_mtp_check.py "$LLMS/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16.gguf" 2>/dev/null | grep -E 'arch|block_count|nextn|tensors with' || true

echo "=== 3. Quantize to Q4_K_M $(date) ==="
$QUANTIZER "$LLMS/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16.gguf" "$LLMS/Qwen3.8-27B-AEON-Ultimate-Uncensored-Q4_K_M.gguf" Q4_K_M

echo "=== 4. Verify quantized file $(date) ==="
ls -lh "$LLMS/Qwen3.8-27B-AEON-Ultimate-Uncensored-Q4_K_M.gguf"
timeout 120 python3 /tmp/gguf_mtp_check.py "$LLMS/Qwen3.8-27B-AEON-Ultimate-Uncensored-Q4_K_M.gguf" 2>/dev/null | grep -E 'arch|block_count|nextn|tensors with' || true

echo "=== 5. Cleanup intermediates $(date) ==="
rm -rf "$SRC"
rm -f "$LLMS/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16.gguf"

echo "=== PIPELINE COMPLETE $(date) ==="
df -h /home | tail -1
