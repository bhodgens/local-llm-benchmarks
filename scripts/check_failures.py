#!/usr/bin/env python3
"""
Retry wrapper for model benchmarking.
Watches for failed models and retries with progressively reduced settings.
"""
import json, os, time, subprocess, urllib.request

PROGRESS_FILE = "/tmp/coding-bench/progress.json"
RESULTS_DIR = "/tmp/coding-bench/results"

# Failure-aware retry configs: try progressively smaller settings
RETRY_CONFIGS = [
    # Attempt 1: moderate reduction
    {"ctx_size": "32768", "kv_k": "q4_0", "kv_v": "q4_0", "ubatch": "256", "max_steps": 15, "gpu_layers": "99"},
    # Attempt 2: aggressive reduction  
    {"ctx_size": "16384", "kv_k": "q4_0", "kv_v": "q4_0", "ubatch": "128", "max_steps": 10, "gpu_layers": "99"},
    # Attempt 3: minimal context, partial offload for dense models
    {"ctx_size": "8192", "kv_k": "q4_0", "kv_v": "q4_0", "ubatch": "128", "max_steps": 8, "gpu_layers": "50"},
    # Attempt 4: last resort
    {"ctx_size": "4096", "kv_k": "q4_0", "kv_v": "q4_0", "ubatch": "64", "max_steps": 5, "gpu_layers": "30"},
]

def get_failed_models():
    """Find models with tau2_error or tau2 with reward=None"""
    if not os.path.exists(PROGRESS_FILE):
        return []
    with open(PROGRESS_FILE) as f:
        p = json.load(f)
    failed = []
    for m in p.get("models", []):
        if "tau2_error" in m:
            failed.append({"name": m["name"], "error": m["tau2_error"], "attempts": 0})
        elif "tau2" in m and m["tau2"].get("reward") is None:
            err = m["tau2"].get("error", "unknown")
            failed.append({"name": m["name"], "error": err, "attempts": 0})
    return failed

def get_completed_models():
    """Models with valid tau2 reward"""
    if not os.path.exists(PROGRESS_FILE):
        return set()
    with open(PROGRESS_FILE) as f:
        p = json.load(f)
    done = set()
    for m in p.get("models", []):
        if "tau2" in m and m["tau2"].get("reward") is not None:
            done.add(m["name"])
    return done

def find_model_file(name):
    """Find the GGUF file for a model name"""
    import glob
    # Try to match from existing scripts
    scripts = [
        "/home/caimlas/llm-benchmarks/scripts/run_tau2bench.py",
        "/home/caimlas/llm-benchmarks/scripts/rerun_deepseek_tau2.py",
    ]
    for script in scripts:
        if not os.path.exists(script):
            continue
        with open(script) as f:
            content = f.read()
        # Simple extraction
        if name in content:
            # Find the file: line after this name
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if name in line:
                    for j in range(i, min(i+5, len(lines))):
                        if '"file"' in lines[j] or "'file'" in lines[j]:
                            # Extract filename
                            import re
                            m = re.search(r'"([^"]+\.gguf)"', lines[j])
                            if m:
                                return m.group(1)
    
    # Try direct glob match
    safe = name.split()[0].split("-")[0].lower()
    for f in glob.glob("/home/files/llms/*.gguf"):
        if safe in f.lower():
            return os.path.basename(f)
    return None

def find_model_is_moe(name):
    """Check if model uses cpu-moe"""
    name_lower = name.lower()
    return any(x in name_lower for x in ["35b-a3b", "35b-a1b", "coder-v2-lite"])

if __name__ == "__main__":
    failed = get_failed_models()
    done = get_completed_models()
    
    print("=== FAILURE ANALYSIS ===")
    print("Completed: %d models" % len(done))
    print("Failed: %d models" % len(failed))
    
    for f in failed:
        print("\n  %s" % f["name"])
        print("    Error: %s" % f["error"][:100])
        
        # Determine retry strategy
        err = f["error"].lower()
        if "out of memory" in err or "oom" in err or "alloc" in err or "cuda" in err:
            print("    Cause: OOM")
            print("    Retry configs:")
            for i, cfg in enumerate(RETRY_CONFIGS):
                print("      Attempt %d: ctx=%s, kv=%s, ubatch=%s, layers=%s, steps=%s" % (
                    i+1, cfg["ctx_size"], cfg["kv_k"], cfg["ubatch"], cfg["gpu_layers"], cfg["max_steps"]))
        elif "timeout" in err:
            print("    Cause: Timeout")
            print("    Retry: reduce max_steps and num_tasks")
        elif "died" in err or "failed to create" in err:
            print("    Cause: Server crash")
            print("    Retry: reduce all settings aggressively")
        else:
            print("    Cause: Unknown")
            print("    Error details:", f["error"][:200])
