#!/usr/bin/env python3
"""
Benchmark BTL-3 Full (merged Q4_K_M) on V100 using upstream llama.cpp.
Uses --reasoning off to suppress thinking tokens.
Waits for BTL-3 Compact to finish first.
"""
import subprocess, json, time, os, urllib.request, http.client, re, shutil
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
MODEL = "/home/files/llms/BTL-3-merged-Q4_K_M.gguf"
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

# Wait for BTL-3 Compact to finish
print("Waiting for BTL-3 Compact benchmark to finish...")
while True:
    result = subprocess.run(["ps", "aux"], capture_output=True, text=True)
    if not any("run_btl3_compact_v2" in line and "grep" not in line for line in result.stdout.split('\n')):
        if not any("lcb_runner" in line and "btl3" in line.lower() for line in result.stdout.split('\n') if "grep" not in line):
            break
    time.sleep(30)
print("Compact finished. Starting Full benchmark.")

# Kill any leftover servers on 18099
subprocess.run(["pkill", "-f", "port 18099"], capture_output=True)
time.sleep(3)

# Start server with --reasoning off
print("\nStarting BTL-3 Full Q4_K_M on V100...")
cmd = [
    BINARY, "--model", MODEL, "--flash-attn", "on",
    "--gpu-layers", "99", "--ctx-size", "131072",
    "--batch-size", "2048", "--ubatch-size", "512",
    "--host", "0.0.0.0", "--port", str(PORT),
    "--parallel", "2", "--temp", "0.0", "-n", "4096",
    "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
    "--threads", "8", "--threads-batch", "8",
    "--reasoning", "off",
]

env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = "0"
logf = open(os.path.join(LOGS, "BTL-3-Full_Q4_K_M_V100_server.log"), "w")
proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)

for _ in range(180):
    time.sleep(2)
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
        if json.loads(r.read()).get("status") == "ok":
            print("Server up!")
            break
    except:
        pass
    if proc.poll() is not None:
        logf.close()
        with open(os.path.join(LOGS, "BTL-3-Full_Q4_K_M_V100_server.log")) as f:
            err = f.read()[-500:]
        print(f"Server died: {err}")
        exit(1)

# Ensure Gemma is up as user sim
try:
    urllib.request.urlopen(f"http://localhost:{GEMMA_PORT}/health", timeout=5)
    print("Gemma user sim: OK")
except:
    print("Starting Gemma...")
    subprocess.run(["sudo", "systemctl", "start", "caimlas-gemma"], capture_output=True)
    time.sleep(10)

# Load progress
with open(PROGRESS_FILE) as f:
    p = json.load(f)

model_name = "BTL-3 Full Q4_K_M (V100)"
mr = None
for m in p["models"]:
    if m["name"] == model_name:
        mr = m
        break
if not mr:
    mr = {"name": model_name, "file": "BTL-3-merged-Q4_K_M.gguf", "category": "27B", "gpu": "V100", "thinking": True}
    p["models"].append(mr)
mr["binary"] = "upstream llama.cpp (--reasoning off)"

# 1. tok/s probe
print("\nProbing tok/s...")
try:
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    conn.request("POST", "/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8, "temperature": 0.0, "stream": False,
    }), {"Content-Type": "application/json"})
    conn.getresponse().read()
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
print("\nRunning LiveCodeBench (75 problems, --reasoning off)...")
output_dir = os.path.join(LCB_DIR, "output", "local_btl3-full-q4km")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

lcb_cmd = [
    BENCH_PY, "-m", "lcb_runner.runner.main",
    "--model", "local/btl3-full-q4km",
    "--scenario", "codegeneration",
    "--release_version", "release_latest",
    "--n", "1", "--temperature", "0.0",
    "--max_tokens", "4096",
    "--num_problems", "75",
    "--openai_timeout", "600",
    "--evaluate",
]
lcb_env = dict(os.environ)
lcb_env["OPENAI_KEY"] = "none"
lcb_env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
lcb_env["HF_ALLOW_CODE_EVAL"] = "1"

logpath = os.path.join(LOGS, "BTL-3-Full_Q4_K_M_V100_lcb.log")
start = time.time()
result = None
try:
    with open(logpath, "w") as lf:
        result = subprocess.run(lcb_cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=36000, env=lcb_env, cwd=LCB_DIR)
    elapsed = time.time() - start
except subprocess.TimeoutExpired:
    elapsed = time.time() - start
except Exception as e:
    elapsed = time.time() - start

pass1 = None
for root, dirs, files in os.walk(output_dir):
    for f in files:
        if f.endswith("_eval.json"):
            with open(os.path.join(root, f)) as rf:
                try:
                    data = json.load(rf)
                    if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                        pass1 = data[0].get("pass@1")
                except: pass

rc = result.returncode if result else -1
print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
mr["livecodebench"] = {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": rc}
save_progress(p)

# 3. tau2-bench
print("\nRunning tau2-bench (airline, 15 tasks)...")
tau2_cmd = [
    "uv", "run", "tau2", "run",
    "--domain", "airline",
    "--agent-llm", "openai/BTL-3-Full",
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
    "--seed", "42", "--save-to", "tau2_V100_BTL-3-Full",
]
tau2_env = dict(os.environ)
tau2_env["OPENAI_API_KEY"] = "none"

logpath2 = os.path.join(LOGS, "BTL-3-Full_Q4_K_M_V100_tau2.log")
start = time.time()
result2 = None
try:
    with open(logpath2, "w") as lf:
        result2 = subprocess.run(tau2_cmd, stdout=lf, stderr=subprocess.STDOUT,
                                 timeout=86400, env=tau2_env, cwd=TAU2_DIR)
    elapsed2 = time.time() - start
except:
    elapsed2 = time.time() - start

reward = None
with open(logpath2) as f:
    content = f.read()
m = re.search(r'Average Reward\s+([\d.]+)', content)
if m: reward = float(m.group(1))

rc2 = result2.returncode if result2 else -1
print(f"  tau2 reward: {reward}  ({elapsed2/60:.0f} min)")
mr["tau2"] = {"reward": reward, "wall_time_s": round(elapsed2, 1), "exit_code": rc2}
save_progress(p)

proc.terminate()
try: proc.wait(timeout=10)
except: proc.kill()
logf.close()

print(f"\n{'='*60}")
print(f"BTL-3 Full Q4_K_M COMPLETE")
print(f"  LCB: {pass1}")
print(f"  tau2: {reward}")
print(f"  tok/s: {mr.get('decode_tps')}")
