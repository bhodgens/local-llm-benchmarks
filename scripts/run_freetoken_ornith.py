#!/usr/bin/env python3
"""
Benchmark Ornith-1.5-35B-A3B-NVFP4 on FreeToken (3060).
MoE offload: experts live in host RAM (35B MoE won't fit 12GB VRAM), LRU expert cache on GPU.
LCB (75) + tau2 (15, user sim on 8081) + tok/s probe. Engine: freetoken.

Usage: /home/caimlas/bench-venv/bin/python scripts/run_freetoken_ornith.py
"""
import subprocess, json, time, os, urllib.request, shutil, re, glob
from datetime import datetime, timezone

FT_BIN = "/home/caimlas/freetoken-venv/bin/ft"
MODEL_DIR = "/home/files/llms/ornith-15-35b-nvfp4"
PORT = 18051
USER_PORT = 8081
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

MODEL_NAME = "Ornith-1.5-35B-A3B NVFP4 [FreeToken]"
LCB_MODEL = "local/ornith-15-35b-freetoken"

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def start_server():
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    # torch is cu130 but system nvcc is 12.0 — allow the mismatch for triton JIT
    env["FREETOKEN_ALLOW_CUDA_MISMATCH"] = "1"
    logf = open(os.path.join(LOGS, "freetoken_ornith_server.log"), "w")
    cmd = [FT_BIN, "serve", "--model-path", MODEL_DIR,
           "--host", "127.0.0.1", "--port", str(PORT),
           "--moe-backend", "offload",
           "--memory-ratio", "0.85",
           "--max-seq-len-override", "131072"]
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(200):  # MoE + offload = slow load (~10 min)
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if r.status == 200:
                print(f"  FreeToken server up after ~{(i+1)*3}s")
                return proc, logf, None
        except Exception:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, "freetoken_ornith_server.log")) as f:
                err = f.read()[-600:]
            return None, None, f"Server died: {err}"
    proc.kill()
    return None, None, "Timeout"

def kill_server(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=15)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(5)

def timed_completion(prompt, max_tokens=256):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
    payload = json.dumps({
        "model": MODEL_DIR,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.3, "stream": False,
    })
    start = time.time()
    conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    body = resp.read()
    elapsed = time.time() - start
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status} from server: {body[:200]!r}")
    data = json.loads(body)
    usage = data.get("usage") or {}
    toks = usage.get("completion_tokens")
    if not isinstance(toks, (int, float)) or toks <= 0:
        raise RuntimeError(f"no completion_tokens in response usage (elapsed {elapsed:.3f}s): {body[:200]!r}")
    return toks / elapsed if elapsed > 0 else 0, toks

def probe_tps():
    try:
        timed_completion("Write a short hello world function.", max_tokens=8)
        tps, toks = timed_completion(
            "Write a detailed essay about the history of computing, from Babbage to modern GPUs.",
            max_tokens=256)
        print(f"  tok/s: {tps:.1f} ({toks} tokens)")
        return round(tps, 1)
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None

def run_livecodebench():
    logpath = os.path.join(LOGS, "freetoken_ornith_lcb.log")
    for d in glob.glob(os.path.join(LCB_DIR, "output", "*Ornith*")):
        shutil.rmtree(d)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", LCB_MODEL,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
        "--num_problems", "75", "--openai_timeout", "600", "--evaluate",
    ]
    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    print("  Running LiveCodeBench (75 problems)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=36000, env=env, cwd=LCB_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 36000, "error": "timeout"}

    pass1 = None
    for d in glob.glob(os.path.join(LCB_DIR, "output", "*Ornith*")):
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith("_eval.json") and not f.endswith("_all.json"):
                    try:
                        data = json.load(open(os.path.join(root, f)))
                        if isinstance(data, list) and data:
                            pass1 = data[0].get("pass@1")
                    except: pass
    if pass1 is None:
        try:
            tail = open(logpath).read().strip().split("\n")[-1]
            v = re.match(r'^([\d.]+)$', tail.strip())
            if v: pass1 = float(v.group(1))
        except: pass
    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed,1), "exit_code": result.returncode}

def run_tau2():
    safe = "Ornith-1.5-35B-A3B_NVFP4_FreeToken"
    logpath = os.path.join(LOGS, f"{safe}_tau2.log")
    save_dir = f"tau2_ft_{safe}"

    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", f"openai/{safe}",
        "--agent-llm-args", json.dumps({
            "api_key": "none", "api_base": f"http://127.0.0.1:{PORT}/v1", "temperature": 0.0}),
        "--user-llm", "openai/Qwythos-27B-v1",
        "--user-llm-args", json.dumps({
            "api_key": "none", "api_base": f"http://localhost:{USER_PORT}/v1"}),
        "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
        "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
        "--seed", "42", "--save-to", save_dir,
    ]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print("  Running tau2-bench (airline, 15 tasks)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=86400, env=env, cwd=TAU2_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"reward": None, "wall_time_s": 86400, "error": "timeout"}

    reward = task_pass = None
    content = open(logpath).read()
    m = re.search(r'Average Reward\s+([\d.]+)', content)
    if m: reward = float(m.group(1))
    m2 = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    if m2: task_pass = float(m2.group(1))
    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {"reward": reward, "task_pass_rate": task_pass,
            "wall_time_s": round(elapsed,1), "exit_code": result.returncode}

def main():
    os.makedirs(LOGS, exist_ok=True)
    progress = load_progress()

    mr = None
    for m in progress["models"]:
        if m["name"] == MODEL_NAME:
            mr = m
            break
    if mr is None:
        mr = {"name": MODEL_NAME, "file": "ornith-15-35b-nvfp4/",
              "category": "MoE 35B", "gpu": "3060", "engine": "freetoken",
              "template": "stock", "thinking": True}
        progress["models"].append(mr)

    if mr.get("livecodebench", {}).get("pass_at_1") is not None and mr.get("tau2", {}).get("reward") is not None:
        print("SKIP: already complete")
        return

    try:
        r = urllib.request.urlopen(f"http://localhost:{USER_PORT}/health", timeout=3)
        if json.loads(r.read()).get("status") != "ok":
            print(f"  WARNING: user sim on :{USER_PORT} not healthy")
    except Exception:
        print(f"  WARNING: user sim on :{USER_PORT} unreachable (tau2 will fail)")

    proc, logf, err = start_server()
    if err:
        print(f"  FATAL: {err[:400]}")
        mr["lcb_error"] = err
        save_progress(progress)
        return

    try:
        if mr.get("livecodebench", {}).get("pass_at_1") is None:
            mr["decode_tps_3060"] = probe_tps()
            save_progress(progress)
            mr["livecodebench"] = run_livecodebench()
            save_progress(progress)
        if mr.get("tau2", {}).get("reward") is None:
            mr["tau2_3060"] = run_tau2()
            save_progress(progress)
    finally:
        kill_server(proc, logf)
        save_progress(progress)

    print("FREETOKEN ORNITH BENCHMARK COMPLETE")

if __name__ == "__main__":
    main()
