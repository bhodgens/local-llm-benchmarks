#!/bin/bash
# Serve Qwen 3.8 27B with KVFlash options for the kvflash experiment.
# Usage: serve_qwen3827b_kvflash.sh <max_ctx> <kvflash>   e.g. 131072 auto | 262144 8192
set -u
CTX="${1:-131072}"
KVFLASH="${2:-auto}"
EXTRA=()
[ -n "$KVFLASH" ] && EXTRA=(--kvflash "$KVFLASH")
cd /root/lucebox
exec env HIP_VISIBLE_DEVICES=0 /root/lucebox/server/build-hip/luce_server \
  /root/models/Qwen3.8-27B-UD-IQ4_XS.gguf \
  --draft /root/models/draft/Qwen3.8-27B-DFlash2-Q8_0.gguf \
  --target-device hip:0 --draft-device hip:0 \
  --draft-block-size 16 \
  --max-ctx "$CTX" \
  --cache-type-k f16 --cache-type-v f16 \
  "${EXTRA[@]}" \
  --host 127.0.0.1 --port 8901
