#!/bin/bash
# GLM-5.3-Flash on Strix Halo via paoai-strix-engine (Vulkan), MTP draft depth 4.
exec env GGML_VK_VISIBLE_DEVICES=1 /root/paoai-strix-engine/build-vk/bin/llama-server   -m /root/models/glm53flash/GLM-5.3-Flash-PaoAI-ROCmFP4-STRIX-BALANCED.gguf   -ngl 999 -c 65536 --parallel 1   --spec-type draft-mtp --spec-draft-n-max 4   -fa on --cache-type-k q8_0 --cache-type-v q8_0   --host 127.0.0.1 --port 8902
