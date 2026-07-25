#!/bin/bash
set -e

LLMS_DIR=/home/files/llms
HF_BASE="https://huggingface.co"

download() {
    local url="$1"
    local dest="$2"
    if [ -f "$dest" ]; then
        echo "SKIP: $(basename $dest) already exists ($(du -h "$dest" | cut -f1))"
        return 0
    fi
    echo "=== Downloading $(basename $dest) ==="
    wget -c --progress=dot:giga -q --show-progress "$url" -O "$dest" 2>&1
    echo "=== DONE: $(basename $dest) ($(du -h "$dest" | cut -f1)) ==="
}

# 1. Nanbeige4-3B-Thinking-2511 Q8_0 (3.9GB)
download \
    "$HF_BASE/bartowski/Nanbeige_Nanbeige4-3B-Thinking-2511-GGUF/resolve/main/Nanbeige_Nanbeige4-3B-Thinking-2511-Q8_0.gguf" \
    "$LLMS_DIR/Nanbeige4-3B-Thinking-Q8_0.gguf"

# 2. Nanbeige4-3B-Thinking-2511 Q4_K_M (2.3GB)
download \
    "$HF_BASE/bartowski/Nanbeige_Nanbeige4-3B-Thinking-2511-GGUF/resolve/main/Nanbeige_Nanbeige4-3B-Thinking-2511-Q4_K_M.gguf" \
    "$LLMS_DIR/Nanbeige4-3B-Thinking-Q4_K_M.gguf"

# 3. BTL-3-Compact AVQ2 (8.4GB) - may not load on our llama.cpp
download \
    "$HF_BASE/badtheorylabs/BTL-3-Compact/resolve/main/model/BTL-3-Compact-AVQ2.gguf" \
    "$LLMS_DIR/BTL-3-Compact-AVQ2.gguf"

# 4. Laguna-S-2.1 UD-IQ3_S (48.4GB) - HUGE
download \
    "$HF_BASE/unsloth/Laguna-S-2.1-GGUF/resolve/main/Laguna-S-2.1-UD-IQ3_S.gguf" \
    "$LLMS_DIR/Laguna-S-2.1-UD-IQ3_S.gguf"

# 5. Hermes V5 APEX-Compact (16.2GB) - 3060
download \
    "$HF_BASE/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-V5-GGUF/resolve/main/Hermes3.6-35B-A3B-Uncensored-Genesis-V5-APEX-Compact.gguf" \
    "$LLMS_DIR/Hermes3.6-35B-A3B-Uncensored-Genesis-V5-APEX-Compact.gguf"

echo ""
echo "ALL DOWNLOADS COMPLETE"
ls -lh $LLMS_DIR/Nanbeige4-3B* $LLMS_DIR/BTL-3* $LLMS_DIR/Laguna* $LLMS_DIR/Hermes3.6* 2>/dev/null
