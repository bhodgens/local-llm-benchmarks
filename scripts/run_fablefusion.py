#!/usr/bin/env python3
"""
Benchmark Qwen3.6-27B-FableFusion-MTP Q4_K_M on the V100.
Runs: tok/s probe + LiveCodeBench (75 problems) + tau2-bench (15 airline tasks).
Sets up LFM2.5-8B user sim on 3060 (:8082) automatically.
"""
import subprocess, json, time, os, urllib.request, shutil, re, signal
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099          # FableFusion on V100
USER_PORT = 8082      # LFM user sim on 3060
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

MODEL_NAME = "Qwen3.6-27B-FableFusion-MTP Q4_K_M"
MODEL_FILE = "Qwen3.6-27B-FableFusion-MTP-Q4_K_M.gguf"
LCB_MODEL = "local/qwen36-27b-fable-mtp"
USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"

os.makedirs(LOGS, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

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

def start_user_sim():
    """Start LFM2.5-8B as tau2 user sim on 3060 at :8082."""
    print("Starting LFM2.5-8B user sim on 3060 (:8082)...")
    # Stop Gemma to free 3060 VRAM
    subprocess.run(["sudo", "systemctl", "stop", "caimlas-gemma"],
                   capture_output=True, timeout=15)
    time.sleep(3)

    path = os.path.join(LLMS_DIR, USER_SIM_FILE)
    cmd = [BINARY,
        "--model", path,
        "--device", "CUDA0",
        "--gpu-layers", "99",
        "--flash-attn", "on",
        "--ctx-size", "32768",
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--threads", "4", "--threads-batch", "4",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--host", "0.0.0.0",
        "--port", str(USER_PORT),
        "--parallel", "4",
        "--temp", "0.0",
        "-n", "1024",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"  # 3060
    logf = open(os.path.join(LOGS, "user_sim_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    ok = wait_health(USER_PORT, 120, "User sim")
    if not ok:
        logf.close()
        with open(os.path.join(LOGS, "user_sim_server.log")) as f:
            err = f.read()[-500:]
        print(f"  User sim failed: {err}")
        proc.kill()
        return None, None
    return proc, logf

def stop_user_sim(proc, logf):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
    if logf:
        logf.close()
    time.sleep(2)

def start_model():
    """Start FableFusion on V100 at :18099."""
    print("Stopping caimlas-bonsai to free V100...")
    subprocess.run(["sudo", "systemctl", "stop", "caimlas-bonsai"],
                   capture_output=True, timeout=15)
    time.sleep(3)

    path = os.path.join(LLMS_DIR, MODEL_FILE)
    if not os.path.exists(path):
        print(f"FATAL: {path} not found!")
        return None, None

    cmd = [BINARY,
        "--model", path,
        "--device", "CUDA0",
        "--gpu-layers", "99",
        "--flash-attn", "on",
        "--ctx-size", "262144",
        "--batch-size", "2048",
        "--ubatch-size", "512",
        "--threads", "8", "--threads-batch", "8",
        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--parallel", "2",
        "--temp", "0.0",
        "-n", "4096",
    ]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # V100
    safe = MODEL_NAME.replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_v100_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    ok = wait_health(PORT, 360, "FableFusion")
    if not ok:
        logf.close()
        with open(os.path.join(LOGS, f"{safe}_v100_server.log")) as f:
            err = f.read()[-1000:]
        print(f"  FableFusion server failed: {err}")
        proc.kill()
        return None, None
    return proc, logf

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
    if logf:
        logf.close()
    time.sleep(3)

def probe_tps():
    """Quick tok/s probe."""
    import http.client
    print("  Probing decode tok/s...")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
        # Warmup
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Write a short hello world function."}],
            "max_tokens": 8, "temperature": 0.0, "stream": False,
        })
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()

        # Timed decode: 256 tokens
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
    """Run LiveCodeBench (75 problems, thinking off)."""
    safe = MODEL_NAME.replace(" ", "_")
    logpath = os.path.join(LOGS, f"{safe}_v100_lcb.log")
    output_dir = os.path.join(LCB_DIR, "output", LCB_MODEL.replace("/", "_"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
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
    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    print("  Running LiveCodeBench (75 problems, thinking off)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=36000, env=env, cwd=LCB_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 36000, "error": "timeout"}
    except Exception as e:
        return {"pass_at_1": None, "wall_time_s": time.time() - start, "error": str(e)}

    pass1 = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith("_eval.json"):
                with open(os.path.join(root, f)) as rf:
                    try:
                        data = json.load(rf)
                        if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                            pass1 = data[0].get("pass@1")
                    except:
                        pass
    if pass1 is None:
        with open(logpath) as f:
            content = f.read()
        for line in content.split("\n"):
            if "pass@1" in line.lower() or "accuracy" in line.lower():
                m = re.search(r'(\d+\.?\d*)\s*%', line)
                if m:
                    pass1 = float(m.group(1)) / 100
                    break
    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def run_tau2():
    """Run tau2-bench (airline, 15 tasks)."""
    safe = MODEL_NAME.replace(" ", "_").replace("(", "").replace(")", "")
    logpath = os.path.join(LOGS, f"{safe}_v100_tau2.log")
    save_dir = f"tau2_v100_{safe}"
    agent_model = f"openai/{safe}"
    user_model = "openai/LFM2.5-8B-A1B-Clean-RealWorld-v2"

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
        "--num-tasks", "15",
        "--num-trials", "1",
        "--max-concurrency", "2",
        "--max-steps", "30",
        "--max-errors", "5",
        "--timeout", "300",
        "--seed", "42",
        "--save-to", save_dir,
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
    except Exception as e:
        return {"reward": None, "wall_time_s": time.time() - start, "error": str(e)}

    reward = None
    task_pass = None
    with open(logpath) as f:
        content = f.read()
    m = re.search(r'Average Reward\s+([\d.]+)', content)
    if m:
        reward = float(m.group(1))
    pm = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    if pm:
        task_pass = float(pm.group(1))
    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {"reward": reward, "task_pass_rate": task_pass,
            "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def restart_services():
    """Restart Bonsai (V100) and Gemma (3060)."""
    print("Restarting production services...")
    subprocess.run(["sudo", "systemctl", "start", "caimlas-bonsai"], capture_output=True, timeout=15)
    subprocess.run(["sudo", "systemctl", "start", "caimlas-gemma"], capture_output=True, timeout=15)
    time.sleep(5)

def main():
    model_path = os.path.join(LLMS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        print(f"FATAL: {model_path} not found! Download still in progress?")
        return

    progress = load_progress()

    # Remove any stale entry
    progress["models"] = [m for m in progress["models"] if m["name"] != MODEL_NAME]
    mr = {"name": MODEL_NAME, "file": MODEL_FILE, "category": "27B",
          "gpu": "V100", "start_time": datetime.now(timezone.utc).isoformat()}
    progress["models"].append(mr)

    print(f"\n{'='*70}")
    print(f"  BENCHMARKING: {MODEL_NAME}")
    print(f"{'='*70}")

    # 1. Start user sim on 3060
    user_proc, user_logf = start_user_sim()
    if user_proc is None:
        print("FATAL: User sim failed to start. Cannot run tau2.")
        mr["error"] = "user_sim_failed"
        save_progress(progress)
        restart_services()
        return

    try:
        # 2. Start model on V100
        proc, logf = start_model()
        if proc is None:
            mr["error"] = "server_failed"
            save_progress(progress)
            return

        # 3. tok/s probe
        tps = probe_tps()
        mr["decode_tps"] = tps
        save_progress(progress)

        # 4. LiveCodeBench
        lcb = run_lcb()
        mr["livecodebench"] = lcb
        save_progress(progress)

        # 5. tau2-bench
        tau2 = run_tau2()
        mr["tau2"] = tau2
        save_progress(progress)

        mr["status"] = "completed"
        mr["end_time"] = datetime.now(timezone.utc).isoformat()
        save_progress(progress)

    except Exception as e:
        import traceback
        traceback.print_exc()
        mr["error"] = str(e)
        save_progress(progress)
    finally:
        kill_model(proc, logf)
        stop_user_sim(user_proc, user_logf)
        restart_services()

    # Summary
    print(f"\n{'='*70}")
    print(f"  RESULTS: {MODEL_NAME}")
    print(f"{'='*70}")
    print(f"  tok/s:     {mr.get('decode_tps', '?')}")
    print(f"  LCB pass@1: {mr.get('livecodebench', {}).get('pass_at_1', '?')}")
    print(f"  tau2 reward: {mr.get('tau2', {}).get('reward', '?')}")

if __name__ == "__main__":
    main()
