#!/usr/bin/env python3
"""
One-off tau2 backfill for Qwopus3.8-27B-Flash MTP Q4_K_M.
Its orchestrator (run_qwen38_flash_v100.py) never had a tau2 phase.
Agent on V100 (GPU0):18099, LFM user sim on 3060 (GPU1):8081.
Same protocol as tau2_backfill_3models.py: airline, 15 tasks, seed 42,
concurrency 2, max-steps 30, temp 0.0.
"""
import subprocess, json, time, os, urllib.request, shutil, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Reuse the proven harness from the 3-model backfill
import importlib.util
spec = importlib.util.spec_from_file_location(
    "bf3", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tau2_backfill_3models.py"))
bf3 = importlib.util.module_from_spec(spec)
# prevent its main() from running on import
bf3.__name__ = "bf3"
spec.loader.exec_module(bf3)

CFG = {
    "name": "Qwopus3.8-27B-Flash MTP Q4_K_M",
    "file": "Qwopus3.8-27B-Flash-MTP-Q4_K_M.gguf",
    # original orchestrator server flags (V100, 65K ctx, MTP n=3 for decode speed)
    "gpu": 0,
    "args": ["--gpu-layers", "99", "--ctx-size", "65536", "--parallel", "1",
             "-n", "4096", "--ubatch-size", "512",
             "--threads", "8", "--threads-batch", "8",
             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja",
             "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
}

def main():
    progress = bf3.load_progress()
    log = bf3.log
    log("=== tau2 backfill: Qwopus3.8-27B-Flash (agent V100, user sim 3060) ===")
    usim, ulogf = bf3.start_user_sim(1)
    try:
        # reuse run_agent_tau2 with warm_cache=False (file is 17GB, cached from prior runs)
        bf3.run_agent_tau2(progress, CFG, warm_cache=False)
        save = bf3.save_progress
        save(progress)
    finally:
        bf3.stop_user_sim(usim, ulogf)
    e = bf3.get_entry(progress, CFG["name"])
    t = e.get("tau2", {}) if e else {}
    log(f"RESULT {CFG['name']}: tau2={t.get('reward')} wall={t.get('wall_time_s')}s")

if __name__ == "__main__":
    main()
