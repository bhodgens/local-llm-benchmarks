#!/bin/bash
# Serve Qwen 3.8 27B Vision on the R9700: Phase 1 profile + mmproj projector.
set -u
KV="${1:-f16}"
BLOCK="${2:-16}"
cd /root/lucebox
exec env HIP_VISIBLE_DEVICES=0 /root/lucebox/server/build-hip/luce_server \
  /root/models/qwen38v/Qwen3.8-27B-IQ4_XS-pure.gguf \
  --mmproj /root/models/qwen38v/Qwen3.8-27B-mmproj-Q8_0.gguf \
  --draft /root/models/draft/Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --target-device hip:0 \
  --draft-device hip:0 \
  --draft-block-size "$BLOCK" \
  --max-ctx 131072 \
  --cache-type-k "$KV" \
  --cache-type-v "$KV" \
  --host 127.0.0.1 --port 8902
