#!/usr/bin/env python3
"""
Run Aider Polyglot benchmark against a local OpenAI-compatible model.
Tests with 1 exercise first, then scales up.
"""
import subprocess, json, time, os, sys
from pathlib import Path

AIDER_DIR = "/home/caimlas/git/aider"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
EXERCISES_DIR = "polyglot-benchmark"  # Relative to AIDER_DIR
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results", "aider")
LOGS = os.path.join(SCRATCH, "logs")

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

def run_aider_test(model_name, api_base, num_tests=1, edit_format="whole", keywords=None):
    """Run Aider benchmark with a model"""
    safe = model_name.replace(" ", "_").replace("/", "_")
    run_name = f"test_{safe}"
    
    cmd = [
        BENCH_PY, os.path.join(AIDER_DIR, "benchmark", "benchmark.py"),
        run_name,
        "--model", f"openai/{model_name}",
        "--edit-format", edit_format,
        "--threads", "1",
        "--num-tests", str(num_tests),
        "--exercises-dir", EXERCISES_DIR,
    ]
    
    if keywords:
        cmd.extend(["--keywords", keywords])
    
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"
    env["OPENAI_API_BASE"] = api_base
    
    logpath = os.path.join(LOGS, f"{safe}_aider.log")
    
    print(f"  Running Aider benchmark ({num_tests} tests, format={edit_format})...")
    start = time.time()
    
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=3600, env=env, cwd=AIDER_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_rate": None, "wall_time_s": 3600, "error": "timeout"}
    except Exception as e:
        return {"pass_rate": None, "wall_time_s": time.time() - start, "error": str(e)}
    
    # Parse results
    pass_rate = None
    with open(logpath) as f:
        content = f.read()
    
    # Look for pass rate in output
    import re
    # Aider outputs: "Percent passed: 45.0"
    match = re.search(r'Percent passed:\s+([\d.]+)', content)
    if match:
        pass_rate = float(match.group(1)) / 100
    
    # Also check for "X/Y passed"
    if pass_rate is None:
        match = re.search(r'(\d+)/(\d+)\s+passed', content)
        if match:
            passed = int(match.group(1))
            total = int(match.group(2))
            pass_rate = passed / total if total > 0 else 0
    
    return {
        "pass_rate": pass_rate,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
    }

if __name__ == "__main__":
    # Quick smoke test against production server
    model = sys.argv[1] if len(sys.argv) > 1 else "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M"
    api_base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080/v1"
    num_tests = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    
    print(f"Testing Aider with {model} at {api_base} ({num_tests} test)")
    result = run_aider_test(model, api_base, num_tests=num_tests, keywords="python")
    print(f"\nResult: {json.dumps(result, indent=2)}")
