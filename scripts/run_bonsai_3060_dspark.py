#!/usr/bin/env python3
"""
Run LCB + tau2 for Ternary-Bonsai-27B on 3060 WITH DSpark speculative decoding.
Uses PrismML fork binary. Max ctx with dspark = 100K (65K for parallel=2 tau2).
"""
import subprocess, json, time, os, urllib.request, shutil, re, http.client
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
MODEL = "/home/files/llms/Ternary-Bonsai-27B-Q2_0.gguf"
DRAFT = "/home/files/llms/Ternary-Bonsai-27B-dspark-Q4_1.gguf"
PORT = 18099
USER_PORT = 8081
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

LCB_MODEL = "local/ternary-bonsai-27b"
MODEL_NAME = "Ternary-Bonsai-27B Q2_0 (3060+dspark)"

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def start_server(ctx_size, parallel=1):
    cmd = [
        BINARY, "--model", MODEL, "--flash-attn", "on",
        "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
        "--parallel", str(parallel), "--temp", "0.0", "-n", "4096",
        "--gpu-layers", "99", "--ctx-size", str(ctx_size), "--ubatch-size", "512",
        "--threads", "6", "--threads-batch", "6",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        "--spec-draft-model", DRAFT, "--spec-draft-n-max", "4",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    logf = open(os.path.join(LOGS, "bonsai_3060_dspark_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(120):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                print(f"  Server up (dspark) at ctx={ctx_size}, parallel={parallel}")
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, "bonsai_3060_dspark_server.log")) as f:
                return None, None, f"Server died: {f.read()[-500:]}"
    proc.kill()
    return None, None, "Timeout"

def kill_server(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def probe_tps():
    print("  Probing tok/s (with dspark)...")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8, "temperature": 0.0, "stream": False,
        })
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        conn.getresponse().read()
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Write a detailed essay about the history of computing, from Babbage to modern GPUs. Include key milestones, important figures, and technological breakthroughs."}],
            "max_tokens": 256, "temperature": 0.0, "stream": False,
        })
        start = time.time()
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        elapsed = time.time() - start
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 256)
        tps = completion_tokens / elapsed if elapsed > 0 else 0
        print(f"  tok/s: {tps:.1f} ({completion_tokens} tokens in {elapsed:.1f}s)")
        return round(tps, 1)
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None

def run_lcb():
    logpath = os.path.join(LOGS, "bonsai_3060_dspark_lcb.log")
    output_dir = os.path.join(LCB_DIR, "output", LCB_MODEL.replace("/", "_"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", LCB_MODEL,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
        "--num_problems", "75", "--openai_timeout", "300", "--evaluate",
    ]
    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    print("  Running LCB (75 problems, thinking off, dspark)...")
    start = time.time()
    with open(logpath, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=36000, env=env, cwd=LCB_DIR)
    elapsed = time.time() - start

    pass1 = None
    with open(logpath) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in reversed(lines):
        try:
            val = float(line)
            if 0 <= val <= 1:
                pass1 = val
                break
        except ValueError:
            m = re.search(r'(\d+\.?\d*)\s*%', line)
            if m:
                pass1 = float(m.group(1)) / 100
                break

    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def run_tau2():
    logpath = os.path.join(LOGS, "bonsai_3060_dspark_tau2.log")
    save_dir = "tau2_3060_bonsai_dspark"
    agent_model = "openai/Ternary-Bonsai-27B_Q2_0_dspark_3060"
    user_model = "openai/Qwopus3.6-27B-Coder-Compat-MTP"

    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", agent_model,
        "--agent-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://127.0.0.1:{PORT}/v1",
            "temperature": 0.0,
        }),
        "--user-llm", user_model,
        "--user-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://localhost:{USER_PORT}/v1",
        }),
        "--num-tasks", "15", "--num-trials", "1",
        "--max-concurrency", "2", "--max-steps", "30",
        "--max-errors", "5", "--timeout", "300",
        "--seed", "42", "--save-to", save_dir,
    ]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print("  Running tau2 (airline, 15 tasks, dspark)...")
    start = time.time()
    with open(logpath, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=86400, env=env, cwd=TAU2_DIR)
    elapsed = time.time() - start

    reward = None
    task_pass = None
    with open(logpath) as f:
        content = f.read()
    m = re.search(r'Average Reward\s+([\d.]+)', content)
    if m: reward = float(m.group(1))
    m = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    if m: task_pass = float(m.group(1))

    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {
        "reward": reward, "task_pass_rate": task_pass,
        "wall_time_s": round(elapsed, 1), "exit_code": result.returncode,
    }

def main():
    os.makedirs(LOGS, exist_ok=True)
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

    mr = None
    for m in progress["models"]:
        if m["name"] == MODEL_NAME:
            mr = m
            break
    if mr is None:
        mr = {"name": MODEL_NAME, "file": "Ternary-Bonsai-27B-Q2_0.gguf", "category": "27B Ternary"}
        progress["models"].append(mr)

    mr["dspark"] = True

    # LCB at 100K context (max with dspark), parallel=1
    print("\n" + "="*60)
    print("  LCB RUN (ctx=100000, parallel=1, WITH dspark)")
    print("="*60)
    proc, logf, err = start_server(100000, parallel=1)
    if err:
        print(f"  FATAL: {err}")
        mr["lcb_error"] = err
        save_progress(progress)
        return
    try:
        tps = probe_tps()
        mr["decode_tps_3060_dspark"] = tps
        save_progress(progress)
        lcb = run_lcb()
        mr["livecodebench_3060_dspark"] = lcb
        save_progress(progress)
    finally:
        kill_server(proc, logf)

    # tau2 at 65536 context, parallel=2 (32K/slot)
    print("\n" + "="*60)
    print("  TAU2 RUN (ctx=65536, parallel=2, 32K/slot, WITH dspark)")
    print("="*60)
    proc, logf, err = start_server(65536, parallel=2)
    if err:
        print(f"  FATAL: {err}")
        mr["tau2_error"] = err
        save_progress(progress)
        return
    try:
        tau2 = run_tau2()
        mr["tau2_3060_dspark"] = tau2
        save_progress(progress)
    finally:
        kill_server(proc, logf)

    save_progress(progress)
    print("\n" + "="*60)
    print("BONSAI 3060+DSPARK BENCHMARK COMPLETE")
    print("="*60)
    lcb = mr.get("livecodebench_3060_dspark", {})
    tau2 = mr.get("tau2_3060_dspark", {})
    print(f"  LCB={lcb.get('pass_at_1')}  tau2={tau2.get('reward')}  tps={mr.get('decode_tps_3060_dspark')}")

if __name__ == "__main__":
    main()
