#!/usr/bin/env python3
"""
Benchmark BTL-3 Compact AVQ2 using the BTL-3 fork binary.
Runs on V100 (CUDA0), uses Gemma on 3060 as tau2 user sim.
"""
import subprocess, json, time, os, urllib.request, http.client, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/btl-3/native/llama.cpp/build/bin/llama-server"
MODEL = "/home/files/llms/BTL-3-Compact-AVQ2.gguf"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
GEMMA_PORT = 8080

os.makedirs(LOGS, exist_ok=True)

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

# Start Gemma as user sim on 3060
print("Starting Gemma as tau2 user sim on 3060...")
subprocess.run(["sudo", "systemctl", "start", "caimlas-gemma"], capture_output=True)
time.sleep(5)
for _ in range(40):
    time.sleep(3)
    try:
        r = urllib.request.urlopen(f"http://localhost:{GEMMA_PORT}/health", timeout=3)
        if r.status == 200:
            print("Gemma up")
            break
    except:
        pass

# Start BTL-3 on V100
print("\nStarting BTL-3 Compact on V100...")
cmd = [
    BINARY, "--model", MODEL, "--flash-attn", "on",
    "--gpu-layers", "99", "--ctx-size", "262144",
    "--batch-size", "2048", "--ubatch-size", "512",
    "--host", "0.0.0.0", "--port", str(PORT),
    "--parallel", "2", "--temp", "0.0", "-n", "4096",
    "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
    "--threads", "8", "--threads-batch", "8",
]

env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = "0"
logf = open(os.path.join(LOGS, "BTL-3-Compact_AVQ2_V100_server.log"), "w")
proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)

# Wait for server
server_ok = False
for _ in range(120):
    time.sleep(2)
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
        if json.loads(r.read()).get("status") == "ok":
            print(f"BTL-3 server up!")
            server_ok = True
            break
    except:
        pass
    if proc.poll() is not None:
        logf.close()
        with open(os.path.join(LOGS, "BTL-3-Compact_AVQ2_V100_server.log")) as f:
            err = f.read()[-500:]
        print(f"Server died: {err}")
        break

if not server_ok:
    print("FATAL: Server not ready")
    proc.kill()
    exit(1)

# Load progress
with open(PROGRESS_FILE) as f:
    p = json.load(f)

# Find or create entry
model_name = "BTL-3-Compact AVQ2 (V100)"
mr = None
for m in p["models"]:
    if m["name"] == model_name:
        mr = m
        break
if not mr:
    mr = {"name": model_name, "file": "BTL-3-Compact-AVQ2.gguf", "category": "27B", "gpu": "V100", "thinking": True}
    p["models"].append(mr)

mr["binary"] = "btl-3-fork (CUDA arch 70-real;86-real)"

# 1. tok/s probe
print("\nProbing tok/s...")
try:
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    # Warmup
    conn.request("POST", "/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8, "temperature": 0.0, "stream": False,
    }), {"Content-Type": "application/json"})
    conn.getresponse().read()
    # Timed
    conn.request("POST", "/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content": "Write a detailed essay about computing history from Babbage to modern GPUs."}],
        "max_tokens": 256, "temperature": 0.0, "stream": False,
    }), {"Content-Type": "application/json"})
    start = time.time()
    data = json.loads(conn.getresponse().read())
    elapsed = time.time() - start
    n_tok = data.get("usage", {}).get("completion_tokens", 256)
    tps = n_tok / elapsed if elapsed > 0 else 0
    print(f"  tok/s: {tps:.1f} ({n_tok} tok in {elapsed:.1f}s)")
    mr["decode_tps"] = round(tps, 1)
except Exception as e:
    print(f"  TPS probe failed: {e}")
    mr["decode_tps"] = None
save_progress(p)

# 2. LiveCodeBench
print("\nRunning LiveCodeBench (75 problems, thinking off)...")
output_dir = os.path.join(LCB_DIR, "output", "local_btl3-compact")
if os.path.exists(output_dir):
    import shutil
    shutil.rmtree(output_dir)

lcb_cmd = [
    BENCH_PY, "-m", "lcb_runner.runner.main",
    "--model", "local/btl3-compact",
    "--scenario", "codegeneration",
    "--release_version", "release_latest",
    "--n", "1", "--temperature", "0.0",
    "--max_tokens", "4096",
    "--num_problems", "75",
    "--openai_timeout", "300",
    "--evaluate",
]
lcb_env = dict(os.environ)
lcb_env["OPENAI_KEY"] = "none"
lcb_env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
lcb_env["HF_ALLOW_CODE_EVAL"] = "1"
lcb_env["LCB_DISABLE_THINKING"] = "1"

logpath = os.path.join(LOGS, "BTL-3-Compact_AVQ2_V100_lcb.log")
start = time.time()
try:
    with open(logpath, "w") as lf:
        result = subprocess.run(lcb_cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=36000, env=lcb_env, cwd=LCB_DIR)
    elapsed = time.time() - start
except Exception as e:
    elapsed = time.time() - start
    result = type('', (), {'returncode': -1})()

# Parse from eval JSON
pass1 = None
empty_count = 0
for root, dirs, files in os.walk(output_dir):
    for f in files:
        if f.endswith("_eval.json"):
            with open(os.path.join(root, f)) as rf:
                try:
                    data = json.load(rf)
                    if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                        pass1 = data[0].get("pass@1")
                except: pass

print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
mr["livecodebench"] = {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}
save_progress(p)

# 3. tau2-bench
print("\nRunning tau2-bench (airline, 15 tasks)...")
agent_model_id = "openai/BTL-3-Compact"
tau2_cmd = [
    "uv", "run", "tau2", "run",
    "--domain", "airline",
    "--agent-llm", agent_model_id,
    "--agent-llm-args", json.dumps({
        "api_key": "none",
        "api_base": f"http://127.0.0.1:{PORT}/v1",
        "temperature": 0.0,
    }),
    "--user-llm", "openai/gpt-4o-mini",
    "--user-llm-args", json.dumps({
        "api_key": "none",
        "api_base": f"http://localhost:{GEMMA_PORT}/v1",
    }),
    "--num-tasks", "15", "--num-trials", "1",
    "--max-concurrency", "2", "--max-steps", "30",
    "--max-errors", "5", "--timeout", "300",
    "--seed", "42", "--save-to", "tau2_V100_BTL-3-Compact",
]
tau2_env = dict(os.environ)
tau2_env["OPENAI_API_KEY"] = "none"

logpath2 = os.path.join(LOGS, "BTL-3-Compact_AVQ2_V100_tau2.log")
start = time.time()
try:
    with open(logpath2, "w") as lf:
        result = subprocess.run(tau2_cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=86400, env=tau2_env, cwd=TAU2_DIR)
    elapsed = time.time() - start
except Exception as e:
    elapsed = time.time() - start
    result = type('', (), {'returncode': -1})()

reward = None
with open(logpath2) as f:
    content = f.read()
m = re.search(r'Average Reward\s+([\d.]+)', content)
if m: reward = float(m.group(1))

print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
mr["tau2"] = {"reward": reward, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}
save_progress(p)

# Cleanup
proc.terminate()
try: proc.wait(timeout=10)
except: proc.kill()
logf.close()

print(f"\n{'='*60}")
print(f"BTL-3 Compact AVQ2 COMPLETE")
print(f"  LCB: {pass1}")
print(f"  tau2: {reward}")
print(f"  tok/s: {mr.get('decode_tps')}")
