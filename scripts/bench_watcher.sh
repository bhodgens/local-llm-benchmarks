#!/bin/bash
# Watch for completed downloads and benchmark them
# Runs the orchestrator every 5 minutes until all models are benchmarked

REQUIRED_MODELS=(
    "Qwopus3.5-27B-v3-Q4_K_M.gguf"
    "Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf"
    "Qwopus3.6-27B-Coder-Compat-MTP-Q4_K_M.gguf"
    "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf"
    "RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf"
)

LLMS_DIR=/home/files/llms
MAX_ITERATIONS=30

for i in $(seq 1 $MAX_ITERATIONS); do
    echo "$(date '+%H:%M:%S') === Iteration $i ==="
    
    # Check which models are still missing
    ALL_DONE=true
    for model in "${REQUIRED_MODELS[@]}"; do
        path="$LLMS_DIR/$model"
        if [ ! -f "$path" ]; then
            echo "  MISSING: $model"
            ALL_DONE=false
        fi
    done
    
    # Run benchmark orchestrator (it will skip already-done and not-yet-downloaded)
    python3 /home/caimlas/llm-benchmarks/scripts/bench_orchestrator.py 2>&1 | tail -5
    
    if [ "$ALL_DONE" = true ]; then
        # Check if all have non-error results
        ERRORS=$(python3 -c "
import json
with open('/home/caimlas/llm-benchmarks/bench_results.json') as f:
    r = json.load(f)
errors = [k for k,v in r.items() if 'error' in v]
print(len(errors))
" 2>/dev/null)
        if [ "$ERRORS" = "0" ]; then
            echo "ALL MODELS BENCHMARKED SUCCESSFULLY"
            break
        else
            echo "  $ERRORS models still have errors, will retry"
        fi
    fi
    
    echo "  Sleeping 300s..."
    sleep 300
done

echo "Watcher complete. Final results:"
cat /home/caimlas/llm-benchmarks/bench_results.json | python3 -m json.tool
