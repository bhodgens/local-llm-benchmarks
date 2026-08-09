#!/usr/bin/env python3
"""
Benchmark BTL-4 IQ2_XXS on RTX 3060 (12GB, sm_86) using upstream llama.cpp.
Uses --reasoning off to suppress thinking tokens.
Same harness as BTL-3: tok/s probe + LiveCodeBench (75 problems) + tau2-bench (airline, 15 tasks).

Model card recommends:
  --jinja --reasoning-format deepseek -fa on --cache-type-k q8_0 --cache-type-v q8_0
Temperature per card: 1.0, top_p 0.95 (but we use temp=0.0 for deterministic benchmark).
"""
import subprocess, json, time, os, urllib.request, http.client, re, shutil
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
MODEL = "/home/files/llms/badtheorylabs_BTL-4-IQ2_XXS.gguf"
GPU_ID = "1"  # 3060
PORT = 18099
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
GEMMA_PORT = 8081  # Qwythos on V100 (Gemma needs Ampere+, can't use V100)

os.makedirs(LOGS, exist_ok=True)

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

# Kill any leftover servers on 18099
subprocess.run(["pkill", "-f", "port 18099"], capture_output=True)
time.sleep(3)

# Start server with --reasoning off
print("Starting BTL-4 IQ2_XXS on 3060 (GPU1)...")
cmd = [
    BINARY, "--model", MODEL, "--flash-attn", "on",
    "--gpu-layers", "99",
    "--ctx-size", "32768",
    "--batch-size", "2048", "--ubatch-size", "512",
    "--host", "0.0.0.0", "--port", str(PORT),
    "--parallel", "2", "--temp", "0.0", "-n", "4096",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--threads", "6", "--threads-batch", "6",
    "--jinja",
    "--reasoning", "off",
    "--reasoning-format", "deepseek",
]

env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = GPU_ID
logf = open(os.path.join(LOGS, "BTL-4-IQ2_XXS_3060_server.log"), "w")
proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)

for _ in range(240):
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
        with open(os.path.join(LOGS, "BTL-4-IQ2_XXS_3060_server.log")) as f:
            err = f.read()[-1000:]
        print(f"Server died: {err}")
        exit(1)
else:
    print("Server failed to start within 8 minutes")
    proc.terminate()
    logf.close()
    exit(1)

# Record VRAM usage
vram = subprocess.run(
    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", GPU_ID],
    capture_output=True, text=True
)
vram_mib = int(vram.stdout.strip()) if vram.stdout.strip().isdigit() else 0
print(f"  VRAM: {vram_mib} MiB")

# Verify user sim is up (Qwythos on V100 port 8081)
try:
    urllib.request.urlopen(f"http://localhost:{GEMMA_PORT}/health", timeout=5)
    print("User sim (Qwythos on V100): OK")
except:
    print("WARNING: User sim not available, tau2 will fail")

# Load or create progress
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        p = json.load(f)
else:
    p = {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

model_name = "BTL-4 IQ2_XXS (3060)"
mr = None
for m in p["models"]:
    if m["name"] == model_name:
        mr = m
        break
if not mr:
    mr = {"name": model_name, "file": "badtheorylabs_BTL-4-IQ2_XXS.gguf",
          "category": "35B-MoE", "gpu": "3060", "thinking": True}
    p["models"].append(mr)
mr["binary"] = "upstream llama.cpp (--reasoning off)"
mr["vram_mib"] = vram_mib

# 1. tok/s probe
print("\nProbing tok/s...")
try:
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    # warmup
    conn.request("POST", "/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content": "Say hello."}],
        "max_tokens": 8, "temperature": 0.0, "stream": False,
    }), {"Content-Type": "application/json"})
    conn.getresponse().read()
    # measured
    conn.request("POST", "/v1/chat/completions", json.dumps({
        "messages": [{"role": "user", "content":
            "Write a detailed essay about computing history from Babbage to modern GPUs. "
            "Include key milestones, important figures, and technological breakthroughs."}],
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
output_dir = os.path.join(LCB_DIR, "output", "BTL-4-IQ2_XXS")
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

lcb_cmd = [
    BENCH_PY, "-m", "lcb_runner.runner.main",
    "--model", "local/btl4-iq2xxs",
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

logpath = os.path.join(LOGS, "BTL-4-IQ2_XXS_3060_lcb.log")
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
    print(f"  LCB error: {e}")
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
    "--agent-llm", "openai/BTL-4-IQ2XXS",
    "--agent-llm-args", json.dumps({
        "api_key": "none",
        "api_base": f"http://127.0.0.1:{PORT}/v1",
        "temperature": 0.0,
    }),
    "--user-llm", "openai/Qwythos-27B-v1",
    "--user-llm-args", json.dumps({
        "api_key": "none",
        "api_base": f"http://localhost:{GEMMA_PORT}/v1",
    }),
    "--num-tasks", "15", "--num-trials", "1",
    "--max-concurrency", "2", "--max-steps", "30",
    "--max-errors", "5", "--timeout", "300",
    "--seed", "42", "--save-to", "tau2_3060_BTL-4-IQ2XXS",
]
tau2_env = dict(os.environ)
tau2_env["OPENAI_API_KEY"] = "none"

logpath2 = os.path.join(LOGS, "BTL-4-IQ2_XXS_3060_tau2.log")
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
print(f"BTL-4 IQ2_XXS on 3060 COMPLETE")
print(f"  LCB pass@1: {pass1}")
print(f"  tau2 reward: {reward}")
print(f"  tok/s: {mr.get('decode_tps')}")
print(f"  VRAM: {vram_mib} MiB")
