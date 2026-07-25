#!/usr/bin/env python3
"""
Rerun tau2 on Nanbeige4-3B-Thinking Q4_K_M with --reasoning off on the server.
"""
import subprocess, json, time, os, urllib.request, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
MODEL = "/home/files/llms/Nanbeige4-3B-Thinking-Q4_K_M.gguf"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
TAU2_DIR = "/home/caimlas/git/tau2-bench"
BONSAI_PORT = 8081

os.makedirs(LOGS, exist_ok=True)

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

# Start server with --reasoning off
print("Starting Nanbeige with --reasoning off...")
cmd = [
    BINARY, "--model", MODEL, "--flash-attn", "on",
    "--gpu-layers", "99", "--ctx-size", "131072",
    "--batch-size", "2048", "--ubatch-size", "512",
    "--host", "0.0.0.0", "--port", str(PORT),
    "--parallel", "2", "--temp", "0.0", "-n", "4096",
    "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
    "--threads", "6", "--threads-batch", "6",
    "--reasoning", "off",
]

env = dict(os.environ)
env["CUDA_VISIBLE_DEVICES"] = "1"
logf = open(os.path.join(LOGS, "Nanbeige_Q4_K_M_reasoning_off_server.log"), "w")
proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)

for _ in range(120):
    time.sleep(2)
    try:
        r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
        if json.loads(r.read()).get("status") == "ok":
            print("Server up!")
            break
    except:
        pass
    if proc.poll() is not None:
        print("Server died!")
        logf.close()
        exit(1)

# Quick test
import http.client
conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=30)
conn.request("POST", "/v1/chat/completions", json.dumps({
    "messages": [{"role": "user", "content": "Write a Python function that returns the sum of two numbers."}],
    "max_tokens": 200, "temperature": 0.0, "stream": False,
}), {"Content-Type": "application/json"})
resp = json.loads(conn.getresponse().read())
content = resp['choices'][0]['message']['content']
print(f"Test response: {repr(content[:150])}")
print(f"Tokens: {resp.get('usage', {})}")

if not content.strip():
    print("WARNING: Empty content with --reasoning off!")

# Ensure Bonsai up
try:
    urllib.request.urlopen(f"http://localhost:{BONSAI_PORT}/health", timeout=5)
except:
    subprocess.run(["sudo", "systemctl", "start", "caimlas-bonsai"], capture_output=True)
    time.sleep(10)

# Run tau2
print("\nRunning tau2-bench (airline, 15 tasks, --reasoning off)...")
tau2_cmd = [
    "uv", "run", "tau2", "run",
    "--domain", "airline",
    "--agent-llm", "openai/Nanbeige-Q4KM-reasoningoff",
    "--agent-llm-args", json.dumps({
        "api_key": "none",
        "api_base": f"http://127.0.0.1:{PORT}/v1",
        "temperature": 0.0,
    }),
    "--user-llm", "openai/gpt-4o-mini",
    "--user-llm-args", json.dumps({
        "api_key": "none",
        "api_base": f"http://localhost:{BONSAI_PORT}/v1",
    }),
    "--num-tasks", "15", "--num-trials", "1",
    "--max-concurrency", "2", "--max-steps", "30",
    "--max-errors", "5", "--timeout", "300",
    "--seed", "42", "--save-to", "tau2_3060_Nanbeige-Q4KM-reasoningoff",
]
tau2_env = dict(os.environ)
tau2_env["OPENAI_API_KEY"] = "none"

logpath = os.path.join(LOGS, "Nanbeige_Q4_K_M_reasoning_off_tau2.log")
start = time.time()
result = None
try:
    with open(logpath, "w") as lf:
        result = subprocess.run(tau2_cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=86400, env=tau2_env, cwd=TAU2_DIR)
    elapsed = time.time() - start
except:
    elapsed = time.time() - start

reward = None
with open(logpath) as f:
    content_str = f.read()
m = re.search(r'Average Reward\s+([\d.]+)', content_str)
if m: reward = float(m.group(1))

rc = result.returncode if result else -1
print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")

# Update progress
with open(PROGRESS_FILE) as f:
    p = json.load(f)
for m_entry in p['models']:
    if m_entry['name'] == 'Nanbeige4-3B-Thinking Q4_K_M':
        m_entry['tau2'] = {"reward": reward, "wall_time_s": round(elapsed, 1), "exit_code": rc, "note": "--reasoning off on server"}
        break
save_progress(p)

proc.terminate()
try: proc.wait(timeout=10)
except: proc.kill()
logf.close()

# Restart Gemma
subprocess.run(["sudo", "systemctl", "start", "caimlas-gemma"], capture_output=True)
print("\nDONE!")
