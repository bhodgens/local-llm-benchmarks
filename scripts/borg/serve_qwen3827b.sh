#!/bin/bash
# Serve Qwen3.8-27B on the R9700 with the lucebox published profile.
# Usage: serve_qwen3827b.sh [kv:f16|q8_0] [block:8|16]
set -u
KV="${1:-f16}"
BLOCK="${2:-16}"
cd /root/lucebox
exec env HIP_VISIBLE_DEVICES=0 server/build-hip/luce_server \
  /root/models/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft /root/models/draft/Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --target-device hip:0 \
  --draft-device hip:0 \
  --draft-block-size "$BLOCK" \
  --max-ctx 131072 \
  --cache-type-k "$KV" \
  --cache-type-v "$KV" \
  --host 127.0.0.1 --port 8901
