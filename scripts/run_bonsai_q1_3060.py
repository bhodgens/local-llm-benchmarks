#!/usr/bin/env python3
"""
Benchmark Bonsai-27B Q1_0 (1.1 bpw) on the RTX 3060 (12GB).
PrismML fork binary, q4_0 KV, hybrid attention keeps KV tiny.
Compares against Q2_0 (1.71 bpw) which scored LCB=62.7%, tau2=0.80 (V100).

VRAM budget: 3.8 GB weights + ~4.3 GB KV @200K + ~0.3 compute = ~8.4 GB
262K context (OOM'd for Q2_0 at 12.8GB) should fit at ~9.7 GB.
"""
import subprocess, json, time, os, urllib.request, shutil, re, http.client
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
MODEL = "/home/files/llms/Bonsai-27B-Q1_0.gguf"
LLMS_DIR = "/home/files/llms"
PORT = 18099
USER_PORT = 8081  # Bonsai Q2_0 on V100 (caimlas-bonsai)
USER_SIM_MODEL = "Ternary-Bonsai-27B-Q2_0.gguf"
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

LCB_MODEL = "local/bonsai-27b-q1_0"
MODEL_NAME = "Bonsai-27B Q1_0 (3060)"
CATEGORY = "27B Ternary"

os.makedirs(LOGS, exist_ok=True)

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def start_server(ctx_size, parallel=1, extra_args=None):
    cmd = [
        BINARY, "--model", MODEL, "--flash-attn", "on",
        "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
        "--parallel", str(parallel), "--temp", "0.0", "-n", "4096",
        "--gpu-layers", "99", "--ctx-size", str(ctx_size), "--ubatch-size", "512",
        "--threads", "6", "--threads-batch", "6",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
    ]
    if extra_args:
        cmd.extend(extra_args)
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    logf = open(os.path.join(LOGS, "bonsai_q1_3060_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                print(f"  Server up at ctx={ctx_size}, parallel={parallel}")
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, "bonsai_q1_3060_server.log")) as f:
                err = f.read()[-500:]
            return None, None, f"Server died: {err}"
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
    print("  Probing tok/s...")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
        # Warmup
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8, "temperature": 0.0, "stream": False,
        })
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        conn.getresponse().read()
        # Timed decode
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Write a detailed essay about the history of computing, from Babbage to modern GPUs. Include key milestones and technological breakthroughs."}],
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

def probe_vram():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "1"],
            capture_output=True, text=True, timeout=5)
        mb = int(result.stdout.strip())
        print(f"  VRAM used: {mb} MiB ({mb/1024:.1f} GB)")
        return mb
    except:
        return None

def run_lcb():
    logpath = os.path.join(LOGS, "bonsai_q1_3060_lcb.log")
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
    print("  Running LCB (75 problems, thinking off)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=36000, env=env, cwd=LCB_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 36000, "error": "timeout"}
    # Parse from eval JSON (authoritative)
    pass1 = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith("_eval.json"):
                with open(os.path.join(root, f)) as rf:
                    try:
                        data = json.load(rf)
                        if isinstance(data, list) and len(data) >= 1:
                            pass1 = data[0].get("pass@1")
                    except:
                        pass
    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def run_tau2():
    logpath = os.path.join(LOGS, "bonsai_q1_3060_tau2.log")
    save_dir = "tau2_3060_bonsai_q1"
    agent_model = "openai/Bonsai-27B-Q1_0_3060"
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", agent_model,
        "--agent-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://127.0.0.1:{PORT}/v1",
            "temperature": 0.0,
        }),
        "--user-llm", f"openai/{USER_SIM_MODEL}",
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
    print("  Running tau2 (airline, 15 tasks)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=86400, env=env, cwd=TAU2_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"reward": None, "wall_time_s": 86400, "error": "timeout"}
    reward, task_pass = None, None
    with open(logpath) as f:
        content = f.read()
    m = re.search(r'Average Reward\s+([\d.]+)', content)
    if m: reward = float(m.group(1))
    m = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    if m: task_pass = float(m.group(1))
    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {"reward": reward, "task_pass_rate": task_pass,
            "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

GPU_SERVICE_3060 = "caimlas-gemma"

def stop_prod():
    print(f"Stopping {GPU_SERVICE_3060} to free 3060...")
    subprocess.run(["sudo", "systemctl", "stop", GPU_SERVICE_3060], capture_output=True, timeout=15)
    time.sleep(5)

def restart_prod():
    print(f"Restarting {GPU_SERVICE_3060}...")
    subprocess.run(["sudo", "systemctl", "start", GPU_SERVICE_3060], capture_output=True, timeout=15)
    time.sleep(5)

def main():
    if not os.path.exists(MODEL):
        print(f"FATAL: {MODEL} not found!")
        return
    progress = load_progress()
    progress["models"] = [m for m in progress["models"] if m["name"] != MODEL_NAME]
    mr = {"name": MODEL_NAME, "file": "Bonsai-27B-Q1_0.gguf", "category": CATEGORY,
          "gpu": "3060", "quant": "Q1_0", "bpw": 1.13,
          "start_time": datetime.now(timezone.utc).isoformat()}
    progress["models"].append(mr)
    stop_prod()

    try:
        # === Max context probe: try 262K (OOM'd for Q2_0) ===
        print("\n" + "="*60)
        print("  MAX CONTEXT PROBE: 262144 (OOM'd for Q2_0 at 12.8GB)")
        print("="*60)
        proc, logf, err = start_server(262144, parallel=1)
        if err:
            print(f"  262K failed (expected for tight VRAM): {err[:200]}")
            mr["max_ctx_3060"] = None
            mr["ctx_262k_error"] = err[:300]
            save_progress(progress)
        else:
            vram = probe_vram()
            mr["max_ctx_3060"] = 262144
            mr["vram_262k_mb"] = vram
            save_progress(progress)
            kill_server(proc, logf)

        # === LCB at 131K context, parallel=1 ===
        print("\n" + "="*60)
        print("  LCB RUN (ctx=131072, parallel=1)")
        print("="*60)
        proc, logf, err = start_server(131072, parallel=1)
        if err:
            print(f"  FATAL: {err}")
            mr["lcb_error"] = err
            save_progress(progress)
            return
        try:
            vram = probe_vram()
            mr["vram_131k_mb"] = vram
            tps = probe_tps()
            mr["decode_tps_3060"] = tps
            save_progress(progress)
            lcb = run_lcb()
            mr["livecodebench_3060"] = lcb
            save_progress(progress)
        finally:
            kill_server(proc, logf)

        # === tau2 at 65536 context, parallel=2 (32K/slot, same as Q2_0) ===
        print("\n" + "="*60)
        print("  TAU2 RUN (ctx=65536, parallel=2, 32K/slot)")
        print("="*60)
        proc, logf, err = start_server(65536, parallel=2)
        if err:
            print(f"  FATAL: {err}")
            mr["tau2_error"] = err
            save_progress(progress)
            return
        try:
            tau2 = run_tau2()
            mr["tau2_3060"] = tau2
            save_progress(progress)
        finally:
            kill_server(proc, logf)

        mr["status"] = "completed"
        mr["end_time"] = datetime.now(timezone.utc).isoformat()
        save_progress(progress)
    finally:
        restart_prod()

    print("\n" + "="*60)
    print("BONSAI Q1_0 3060 BENCHMARK COMPLETE")
    print("="*60)
    lcb = mr.get("livecodebench_3060", {})
    tau2 = mr.get("tau2_3060", {})
    print(f"  tok/s:      {mr.get('decode_tps_3060')}")
    print(f"  LCB pass@1: {lcb.get('pass_at_1')}")
    print(f"  tau2 reward: {tau2.get('reward')}")
    print(f"  max_ctx:    {mr.get('max_ctx_3060')}")
    print(f"  VRAM 262K:  {mr.get('vram_262k_mb')} MiB")
    print(f"  VRAM 131K:  {mr.get('vram_131k_mb')} MiB")

if __name__ == "__main__":
    main()
