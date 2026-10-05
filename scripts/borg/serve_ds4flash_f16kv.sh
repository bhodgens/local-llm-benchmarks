#!/bin/bash
cd /root/lucebox
exec /root/lucebox/server/build-hip/luce_server   /root/models/ds4/DeepSeek-V4-Flash-0731-ROCMFPX-MIX-STRIX.gguf   --draft /root/models/ds4/draft/DeepSeek-V4-Flash-0731-DSpark-draft-Q4RMFP4-denseF16.gguf   --profile ds4-r9700-strix   --ds4-fused-verify-f16-kv   --host 127.0.0.1 --port 8902
