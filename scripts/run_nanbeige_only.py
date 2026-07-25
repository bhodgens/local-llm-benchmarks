#!/usr/bin/env python3
"""
Run only Nanbeige benchmarks while large downloads continue.
Stops gemma (3060), starts Bonsai (V100 user sim), benchmarks 2 models.
"""
import subprocess, json, time, os, sys

# Import the orchestrator functions
sys.path.insert(0, "/home/caimlas/llm-benchmarks/scripts")
from run_new_models import (
    PHASE1_3060, benchmark_model, systemctl, wait_health,
    BONSAI_SERVICE, GEMMA_SERVICE, BONSAI_PORT,
    load_progress, save_progress,
)

# Only run models whose files are fully present
NANBEIGE_MODELS = [
    m for m in PHASE1_3060 
    if "Nanbeige" in m["name"]
]

def main():
    # Verify files exist
    for m in NANBEIGE_MODELS:
        path = f"/home/files/llms/{m['file']}"
        if not os.path.exists(path):
            print(f"ERROR: {m['file']} not found")
            return
        size_gb = os.path.getsize(path) / (1024**3)
        print(f"  {m['file']}: {size_gb:.1f}GB")

    progress = load_progress()
    
    # Check which are already done
    done_names = set()
    for m in progress["models"]:
        lb = m.get("livecodebench", {})
        if lb.get("pass_at_1") is not None:
            done_names.add(m["name"])

    # Stop gemma, start bonsai
    print("\nStopping gemma (free 3060 VRAM)...")
    systemctl("stop", GEMMA_SERVICE)
    
    print("Starting Bonsai (V100 user sim)...")
    systemctl("start", BONSAI_SERVICE)
    if not wait_health(BONSAI_PORT, timeout=180):
        print("WARNING: Bonsai not healthy on :8081!")

    for model in NANBEIGE_MODELS:
        if model["name"] in done_names:
            print(f"SKIP: {model['name']} (already done)")
            continue
        benchmark_model(model, progress, BONSAI_PORT)

    print("\nNanbeige benchmarks complete!")
    for m in progress["models"]:
        if "Nanbeige" in m["name"]:
            lb = m.get("livecodebench", {})
            t2 = m.get("tau2", {})
            tps = m.get("decode_tps")
            print(f"  {m['name']:<45} LCB={lb.get('pass_at_1','?')}  tau2={t2.get('reward','?')}  tok/s={tps}")

if __name__ == "__main__":
    main()
