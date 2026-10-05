#!/bin/bash
cd /root/lucebox
exec env HIP_VISIBLE_DEVICES=0 /root/lucebox/server/build-hip/luce_server   /root/models/laguna/Laguna-XS-2.1-Q4_K_M.gguf   --draft /root/models/laguna/draft/laguna-xs21-dflash-q4.gguf   --target-device hip:0   --kvflash auto   --max-ctx 131072   --host 127.0.0.1 --port 8902
