#!/bin/bash
# Wait for Nanbeige benchmark to finish, then run full orchestrator for remaining models
set -e

NANBEIGE_PID=1859730
echo "Waiting for Nanbeige benchmark (PID $NANBEIGE_PID) to finish..."

# Wait for the process to complete
while kill -0 $NANBEIGE_PID 2>/dev/null; do
    sleep 60
done

echo "[$(date '+%H:%M:%S')] Nanbeige benchmark finished."
echo "=== Final Nanbeige results ==="
tail -10 /tmp/coding-bench/nanbeige_bench.log 2>/dev/null

# Verify all downloads are complete
echo "[$(date '+%H:%M:%S')] Verifying downloads..."
for f in Laguna-S-2.1-UD-IQ3_S.gguf Hermes3.6-35B-A3B-Uncensored-Genesis-V5-APEX-Compact.gguf; do
    sz=$(stat -c%s "/home/files/llms/$f" 2>/dev/null || echo "0")
    mb=$((sz / 1048576))
    echo "  $f: ${mb}MB"
done

# Run the full orchestrator (will skip completed Nanbeige models, fail BTL-3 fast, then do Laguna + Hermes)
echo "[$(date '+%H:%M:%S')] Starting full orchestrator for Laguna + Hermes V5..."
cd /home/caimlas/llm-benchmarks
PYTHONUNBUFFERED=1 /home/caimlas/bench-venv/bin/python3 -u scripts/run_new_models.py 2>&1 | tee /tmp/coding-bench/full_orchestrator.log

echo "[$(date '+%H:%M:%S')] Full orchestrator complete!"
