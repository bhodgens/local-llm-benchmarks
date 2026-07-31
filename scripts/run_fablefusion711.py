#!/usr/bin/env python3
"""
Benchmark FableFusion-711 Q4_K_M MTP on V100.
Runs: tok/s probe + LiveCodeBench (75 problems) + tau2-bench (15 airline tasks).
"""
import subprocess, json, time, os, urllib.request, shutil, re, http.client
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp-fermion/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
USER_PORT = 8080  # Gemma 4 12B via tabbyAPI (LFM on 8082 unavailable)
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
RESULTS = os.path.join(SCRATCH, "results")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

MODEL_NAME = "Qwen3.6-27B-FableFusion-711 Q4_K_M MTP"
MODEL_FILE = os.path.join(LLMS_DIR, "gguf", "Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf")
LCB_MODEL = "local/fable-711"
USER_SIM_NAME = "openai/gemma-4-12B-it"  # tabbyAPI on :8080

os.makedirs(LOGS, exist_ok=True)
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)

print(f"=== {MODEL_NAME} ===")
print(f"Time: {datetime.now(timezone.utc).isoformat()}")

def wait_health(port, timeout=180, label=""):
    for i in range(timeout // 2):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                print(f"  {label} ready after {i*2}s")
                return True
        except:
            pass
    print(f"  {label} timeout after {timeout}s")
    return False

def kill_server():
    subprocess.run(["pkill", "-f", "llama-server.*fable-711"], capture_output=True)
    time.sleep(2)

# ── Phase 1: Tok/s probe ──────────────────────────────────────────────
print("\n[Phase 1] Tok/s probe...")

env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = "0"

# Start server
cmd = [BINARY,
    "--model", MODEL_FILE,
    "--gpu-layers", "99",
    "--flash-attn", "on",
    "--ctx-size", "32768",
    "--batch-size", "2048",
    "--ubatch-size", "512",
    "--threads", "8", "--threads-batch", "8",
    "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
    "--host", "0.0.0.0",
    "--port", str(PORT),
    "--parallel", "2",
    "--temp", "0.0",
    "-n", "512",
    "--reasoning", "off",
]
logf = open(os.path.join(LOGS, "fable711_server.log"), "w")
proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
if not wait_health(PORT, 120, "Server"):
    print("Server failed to start!")
    proc.kill()
    exit(1)

# Tok/s probe: same 3 prompt classes as before
TOKS = {}
for pclass, prompt in [("code", "Write a Python function to find the longest palindromic substring in a string."),
                        ("essay", "Write a 200-word essay about the economic impact of artificial intelligence."),
                        ("factual", "What were the main causes of the fall of the Roman Empire? List five key factors.")]:
    t0 = time.time()
    payload = json.dumps({"model": "local/fable-711", "messages": [{"role": "user", "content": prompt}],
                           "max_tokens": 256, "temperature": 0.0})
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                  data=payload.encode(),
                                  headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        resp = json.loads(r.read())
    t1 = time.time()
    elapsed = t1 - t0
    toks = resp["usage"]["completion_tokens"]
    TOKS[pclass] = {"tps": round(toks / elapsed, 1), "tokens": toks, "time_s": round(elapsed, 1)}
    print(f"  {pclass}: {TOKS[pclass]['tps']} tok/s ({toks} tokens in {elapsed:.1f}s)")

# ── Phase 2: LiveCodeBench ────────────────────────────────────────────
print("\n[Phase 2] LiveCodeBench (75 problems)...")
# Clean LCB output dir
output_dir = os.path.join(LCB_DIR, "output", LCB_MODEL.replace("/", "_"))
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)

lcb_cmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
    "--model", LCB_MODEL,
    "--scenario", "codegeneration",
    "--release_version", "release_latest",
    "--n", "1",
    "--temperature", "0.0",
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
lcb_log = os.path.join(LOGS, "fable711_lcb.log")

t0 = time.time()
result = subprocess.run(lcb_cmd, stdout=open(lcb_log, "w"), stderr=subprocess.STDOUT,
                        env=lcb_env, cwd=LCB_DIR, timeout=36000)
elapsed = time.time() - t0
print(f"LCB finished in {elapsed/60:.0f} min, exit code {result.returncode}")

# Parse LCB results
pass1 = None
for root, dirs, files in os.walk(output_dir):
    for f in files:
        if f.endswith("_eval.json"):
            try:
                with open(os.path.join(root, f)) as rf:
                    data = json.load(rf)
                if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                    pass1 = data[0].get("pass@1")
            except:
                pass
    if pass1 is not None:
        break

# Fallback: regex in log
if pass1 is None:
    with open(lcb_log) as f:
        for line in f:
            if "pass@1" in line.lower() or "accuracy" in line.lower():
                m = re.search(r'(\d+\.?\d*)\s*%', line)
                if m:
                    pass1 = float(m.group(1)) / 100
                    break

print(f"  LCB pass@1: {pass1}")

# ── Phase 3: Tau2 ─────────────────────────────────────────────────────
print("\n[Phase 3] Tau2 (15 airline tasks)...")
tau2_result = {"reward": None, "task_pass_rate": None, "wall_time_s": 0}
tau2_log = os.path.join(LOGS, "fable711_tau2.log")

try:
    t0 = time.time()
    tau2_cmd = [
        "uv", "run", "--cwd", TAU2_DIR, "tau2", "run",
        "--domain", "airline",
        "--agent-llm", USER_SIM_NAME,
        "--agent-llm-args", json.dumps({"api_key": "none", "api_base": f"http://127.0.0.1:{USER_PORT}/v1"}),
        "--user-llm", USER_SIM_NAME,
        "--user-llm-args", json.dumps({"api_key": "none", "api_base": f"http://127.0.0.1:{USER_PORT}/v1"}),
        "--user-port", str(USER_PORT),
        "--max-steps", "32",
    ]
    result = subprocess.run(tau2_cmd, stdout=open(tau2_log, "w"), stderr=subprocess.STDOUT,
                            timeout=1800, cwd=TAU2_DIR)
    tau2_result["wall_time_s"] = round(time.time() - t0, 1)
    
    # Parse tau2 results
    with open(tau2_log) as f:
        content = f.read()
    rm = re.search(r'Average Reward\s+([\d.]+)', content)
    if rm:
        tau2_result["reward"] = float(rm.group(1))
    pm = re.search(r'Pass Rate:\s+(\d+)\s*/\s*(\d+)', content)
    if pm:
        tau2_result["task_pass_rate"] = f"{int(pm.group(1))}/{int(pm.group(2))}"
    print(f"  tau2 reward: {tau2_result['reward']}")
except Exception as e:
    tau2_result["error"] = str(e)
    print(f"  tau2 error: {e}")

# ── Save results ──────────────────────────────────────────────────────
kill_server()

final = {
    "model": MODEL_NAME,
    "file": "Qwen3.6-27B-Fable-Fus-711-UnHeretic-NM-DAU-NEO-MAX-NEO-MTP-Q4_K_M.gguf",
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "tok_s": TOKS,
    "lcb": {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode},
    "tau2": tau2_result,
}

rfile = os.path.join(RESULTS, "fablefusion711_results.json")
with open(rfile, "w") as f:
    json.dump(final, f, indent=2)
print(f"\nResults saved to {rfile}")
print(f"\n=== SUMMARY ===")
print(f"tok/s:     code={TOKS.get('code',{}).get('tps','N/A')}  essay={TOKS.get('essay',{}).get('tps','N/A')}  factual={TOKS.get('factual',{}).get('tps','N/A')}")
print(f"LCB:       {pass1}")
print(f"tau2:      {tau2_result.get('reward','N/A')}")
