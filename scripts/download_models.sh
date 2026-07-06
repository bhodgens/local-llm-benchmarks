#!/bin/bash
set -e

LLMS_DIR=/home/files/llms
HF_BASE="https://huggingface.co"

declare -A MODELS=(
    ["Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf"]="Jackrong/Qwopus3.6-27B-v2-MTP-GGUF"
    ["Qwopus3.5-27B-v3-Q4_K_M.gguf"]="Jackrong/Qwopus3.5-27B-v3-GGUF"
    ["Qwopus3.6-27B-Coder-Compat-MTP-Q4_K_M.gguf"]="Jackrong/Qwopus3.6-27B-Coder-Compat-MTP-GGUF"
    ["Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf"]="Jackrong/Qwopus3.6-35B-A3B-Coder-MTP-GGUF"
    ["RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"]="deadbydawn101/RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP"
)

for fn in "${!MODELS[@]}"; do
    repo="${MODELS[$fn]}"
    dest="$LLMS_DIR/$fn"
    url="$HF_BASE/$repo/resolve/main/$fn"
    
    if [ -f "$dest" ]; then
        echo "SKIP: $fn already exists"
        continue
    fi
    
    echo "=== Downloading $fn from $repo ==="
    echo "URL: $url"
    wget -c --progress=dot:giga -q --show-progress "$url" -O "$dest" 2>&1
    echo "=== DONE: $fn ==="
done

echo "ALL DOWNLOADS COMPLETE"
