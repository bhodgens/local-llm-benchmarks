#!/bin/bash
# Serve DeepSeek V4.1 Flash on borg: lucebox reference profile (R9700 + Strix + SSD streaming).
# Run from /root/lucebox per DS41.md. Dense/hot experts + drafter on R9700,
# second expert stack on Strix, rest streamed from NVMe.
set -u
cd /root/lucebox
exec /root/lucebox/server/build-hip/luce_server \
  /root/models/ds41/DeepSeek-V4.1-Flash-ROCMFP2S.gguf \
  --draft /root/models/ds41/draft/DeepSeek-V4.1-Flash-DSpark-draft-MXFP4-Q8.gguf \
  --profile ds41-lucebox \
  --host 127.0.0.1 --port 8902
