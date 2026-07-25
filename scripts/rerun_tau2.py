#!/usr/bin/env python3
"""
Rerun tau2 on Qwen3.6-35B-A3B IQ3_K_R4 (3060) and Nanbeige4-3B-Thinking Q4_K_M (3060).
Bonsai on V100 as user sim.
"""
import subprocess, json, time, os, urllib.request, http.client, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
BONSAI_PORT = 8081

os.makedirs(LOGS, exist_ok=True)

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def start_server(model_path, ctx, kv_k, kv_v, threads, extra_args, cuda_dev, logname, reasoning_off=False):
    cmd = [
        BINARY, "--model", model_path, "--flash-attn", "on",
        "--batch-size", "2048", "--ubatch-size", "512",
        "--host", "0.0.0.0", "--port", str(PORT),
        "--parallel", "2", "--temp", "0.0", "-n", "4096",
        "--ctx-size", str(ctx),
        "--cache-type-k", kv_k, "--cache-type-v", kv_v,
        "--threads", str(threads), "--threads-batch", str(threads),
        "--gpu-layers", "99",
    ]
    if extra_args:
        cmd.extend(extra_args)
    if reasoning_off:
        cmd.extend(["--reasoning", "off"])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(cuda_dev)

    logf = open(os.path.join(LOGS, logname), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)

    for _ in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            return None, None
    proc.kill()
    return None, None

def run_tau2(agent_name, save_to):
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", f"openai/{agent_name}",
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
        "--seed", "42", "--save-to", save_to,
    ]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    logpath = os.path.join(LOGS, f"{save_to}_tau2.log")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=86400, env=env, cwd=TAU2_DIR)
        elapsed = time.time() - start
    except:
        elapsed = time.time() - start
        result = type('', (), {'returncode': -1})

    reward = None
    with open(logpath) as f:
        content = f.read()
    m = re.search(r'Average Reward\s+([\d.]+)', content)
    if m: reward = float(m.group(1))

    return reward, round(elapsed, 1), result.returncode

# Load progress
with open(PROGRESS_FILE) as f:
    p = json.load(f)

# Stop Gemma, free 3060
print("Stopping Gemma...")
subprocess.run(["sudo", "systemctl", "stop", "caimlas-gemma"], capture_output=True)
time.sleep(3)

# Ensure Bonsai is up as user sim
print("Ensuring Bonsai up...")
try:
    urllib.request.urlopen(f"http://localhost:{BONSAI_PORT}/health", timeout=5)
    print("Bonsai OK")
except:
    subprocess.run(["sudo", "systemctl", "start", "caimlas-bonsai"], capture_output=True)
    time.sleep(10)

# ─── Model 1: Qwen3.6-35B-A3B IQ3_K_R4 (3060, cpu-moe) ──────────────────────
print("\n" + "="*60)
print("  Qwen3.6-35B-A3B IQ3_K_R4 tau2")
print("="*60)

proc, logf = start_server(
    "/home/files/llms/Qwen3.6-35B-A3B-IQ3_K_R4.gguf",
    ctx=262144, kv_k="q8_0", kv_v="q8_0", threads=6,
    extra_args=["--cpu-moe"], cuda_dev="1",
    logname="Qwen3.6-35B-A3B_IQ3_K_R4_tau2_server.log",
    reasoning_off=True,
)

if proc:
    print("Server up!")
    reward, wall, rc = run_tau2("Qwen35-IQ3-R4-tau2-rerun", "tau2_3060_Qwen35-IQ3-R4-rerun")
    print(f"  tau2 reward: {reward}  ({wall/60:.0f} min)")

    for m in p['models']:
        if 'Qwen3.6-35B-A3B IQ3_K_R4' in m['name']:
            m['tau2'] = {"reward": reward, "wall_time_s": wall, "exit_code": rc}
            break
    save_progress(p)

    proc.terminate()
    try: proc.wait(timeout=10)
    except: proc.kill()
    logf.close()
    time.sleep(3)
else:
    print("FAILED to start server!")

# ─── Model 2: Nanbeige4-3B-Thinking Q4_K_M (3060) ───────────────────────────
print("\n" + "="*60)
print("  Nanbeige4-3B-Thinking Q4_K_M tau2")
print("="*60)

proc, logf = start_server(
    "/home/files/llms/Nanbeige4-3B-Thinking-Q4_K_M.gguf",
    ctx=131072, kv_k="q4_0", kv_v="q4_0", threads=6,
    extra_args=[], cuda_dev="1",
    logname="Nanbeige4-3B-Thinking_Q4_K_M_tau2_server.log",
    reasoning_off=False,  # nanbeige uses chat_template_kwargs
)

if proc:
    print("Server up!")
    reward, wall, rc = run_tau2("Nanbeige-Q4KM-tau2", "tau2_3060_Nanbeige-Q4KM")
    print(f"  tau2 reward: {reward}  ({wall/60:.0f} min)")

    for m in p['models']:
        if m['name'] == 'Nanbeige4-3B-Thinking Q4_K_M':
            m['tau2'] = {"reward": reward, "wall_time_s": wall, "exit_code": rc}
            break
    save_progress(p)

    proc.terminate()
    try: proc.wait(timeout=10)
    except: proc.kill()
    logf.close()
else:
    print("FAILED to start server!")

# Restart Gemma
print("\nRestarting Gemma...")
subprocess.run(["sudo", "systemctl", "start", "caimlas-gemma"], capture_output=True)

print("\nDONE!")
