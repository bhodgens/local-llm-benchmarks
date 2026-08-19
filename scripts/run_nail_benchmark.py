#!/usr/bin/env python3
"""
Single-model benchmark runner for Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL on 3060.
Runs: tok/s probe -> LiveCodeBench (75 problems, thinking off) -> tau2-bench (airline, 15 tasks).
User simulator = Qwythos on V100 (port 8081).
"""
import subprocess, json, time, os, urllib.request, shutil, re, sys
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

MODEL_FILE = "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
MODEL_NAME = "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL"
LCB_MODEL = "local/nail-35b"
SAFE_NAME = "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL"

# MoE args: cpu-moe for 12GB GPU, 262K ctx, 6 threads for expert offload
MOE_ARGS = [
    "--gpu-layers", "99",
    "--cpu-moe",
    "--ctx-size", "262144",
    "--ubatch-size", "512",
    "--threads", "6",
    "--threads-batch", "6",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--reasoning", "off",  # thinking off for all benchmarks
]

os.makedirs(LOGS, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}


def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)


def wait_for_server(port, timeout=240):
    for i in range(timeout // 2):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return True
        except:
            pass
    return False


def start_model(args, port=PORT, device="1"):
    path = os.path.join(LLMS_DIR, MODEL_FILE)
    cmd = [BINARY, "--model", path, "--device", "CUDA0",
           "--flash-attn", "on", "--batch-size", "2048",
           "--host", "127.0.0.1", "--port", str(port),
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    cmd.extend(args)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = device
    logf = open(os.path.join(LOGS, f"{SAFE_NAME}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)

    for i in range(120):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                log(f"  Server ready after {i*2}s")
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{SAFE_NAME}_server.log")) as f:
                err = f.read()[-500:]
            return None, None, f"Server died:\n{err}"
    proc.kill()
    logf.close()
    return None, None, "Server timeout (240s)"


def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
            proc.wait(timeout=5)
    if logf:
        logf.close()
    time.sleep(3)


def throughput_probe(port=PORT):
    """256-token decode probe"""
    payload = json.dumps({
        "messages": [{"role": "user", "content": "Write a Python function that checks if a number is prime."}],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=payload,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    u = data.get("usage", {})
    # Try timings from server
    t = data.get("timings", {})
    if t:
        decode_tps = round(t.get("predicted_n", 1) / (t.get("predicted_ms", 1) / 1000), 1)
        prompt_tps = round(t.get("prompt_n", 1) / (t.get("prompt_ms", 1) / 1000), 1)
    else:
        # Fallback: use usage tokens (less accurate)
        decode_tps = u.get("completion_tokens", 1)
        prompt_tps = u.get("prompt_tokens", 1)
    return decode_tps, prompt_tps


def run_livecodebench():
    """LiveCodeBench 75 problems, thinking off"""
    logpath = os.path.join(LOGS, f"{SAFE_NAME}_lcb.log")

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", LCB_MODEL,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1",
        "--temperature", "0.0",
        "--max_tokens", "4096",
        "--num_problems", "75",
        "--openai_timeout", "600",
        "--evaluate",
        "--use_cache",
    ]

    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    log("  Running LiveCodeBench (75 problems)...")
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

    # Parse from eval JSON files
    pass1 = None
    # Output dir uses DISPLAY NAME from lm_styles.py
    output_dir = os.path.join(LCB_DIR, "output", "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL")

    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith("_eval.json"):
                with open(os.path.join(root, f)) as rf:
                    try:
                        data = json.load(rf)
                        if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                            val = data[0].get("pass@1")
                            if val is not None:
                                pass1 = val
                    except:
                        pass

    # Fallback: scan log
    if pass1 is None:
        with open(logpath) as f:
            content = f.read()
        for line in content.split("\n"):
            if "pass@1" in line.lower() or "accuracy" in line.lower():
                m = re.search(r'(\d+\.?\d*)\s*%', line)
                if m:
                    pass1 = float(m.group(1)) / 100
                    break

    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1),
            "exit_code": result.returncode}


def run_tau2():
    """tau2-bench airline domain, 15 tasks. User sim = Qwythos V100:8081"""
    logpath = os.path.join(LOGS, f"{SAFE_NAME}_tau2.log")
    save_dir = f"tau2_{SAFE_NAME}"

    # Clean previous results to avoid EOFError
    sim_dir = os.path.join(TAU2_DIR, "data", "simulations", save_dir)
    if os.path.exists(sim_dir):
        shutil.rmtree(sim_dir)

    agent_model = f"openai/{SAFE_NAME}"
    user_model = "openai/Qwythos-27B-v1"

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
            "api_base": "http://127.0.0.1:8081/v1",
        }),
        "--num-tasks", "15",
        "--num-trials", "1",
        "--max-concurrency", "2",
        "--max-steps", "32",
        "--max-errors", "5",
        "--timeout", "300",
        "--seed", "42",
        "--save-to", save_dir,
    ]

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    log("  Running tau2-bench (airline, 15 tasks)...")
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

    # Parse reward from log
    reward = None
    task_pass = None
    with open(logpath) as f:
        content = f.read()
    reward_match = re.search(r'Average Reward\s+([\d.]+)', content)
    if reward_match:
        reward = float(reward_match.group(1))
    pass_match = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    if pass_match:
        task_pass = float(pass_match.group(1))

    return {
        "reward": reward,
        "task_pass_rate": task_pass,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
    }


def get_vram():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5)
        lines = out.strip().split("\n")
        for line in lines:
            idx, used = line.strip().split(", ")
            if idx == "1":  # 3060
                return int(used)
    except:
        pass
    return None


def main():
    log("=" * 60)
    log("Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL Benchmark on 3060")
    log("=" * 60)

    # Check model exists
    model_path = os.path.join(LLMS_DIR, MODEL_FILE)
    if not os.path.exists(model_path):
        log(f"ERROR: Model not found at {model_path}")
        sys.exit(1)
    sz = os.path.getsize(model_path) / 1e9
    log(f"Model: {sz:.2f} GB")

    progress = load_progress()

    # Find or create model entry
    mr = None
    for m in progress["models"]:
        if m["name"] == MODEL_NAME:
            mr = m
            break
    if mr is None:
        mr = {"name": MODEL_NAME, "file": MODEL_FILE, "gpu": "3060"}
        progress["models"].append(mr)

    # Start model on 3060
    log("Starting model server on 3060...")
    proc, logf, err = start_model(MOE_ARGS)
    if err:
        log(f"FAILED to start server: {err}")
        mr["error"] = err
        mr["status"] = "failed"
        save_progress(progress)
        sys.exit(1)

    try:
        # VRAM check
        vram = get_vram()
        mr["vram_mib"] = vram
        log(f"VRAM usage: {vram} MiB / 12288 MiB")

        # Throughput probe
        log("Throughput probe (256 tokens)...")
        decode_tps, prompt_tps = throughput_probe()
        mr["decode_tps"] = decode_tps
        mr["prompt_tps"] = prompt_tps
        log(f"Decode: {decode_tps} tok/s, Prompt: {prompt_tps} tok/s")
        save_progress(progress)

        # LiveCodeBench
        log("Starting LiveCodeBench...")
        lcb = run_livecodebench()
        mr["livecodebench"] = lcb
        log(f"LCB pass@1: {lcb.get('pass_at_1')} ({lcb.get('wall_time_s',0)/60:.0f} min)")
        save_progress(progress)

        # tau2-bench
        # Check V100 user sim is available
        try:
            r = urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5)
            if json.loads(r.read()).get("status") == "ok":
                log("V100 user sim (Qwythos) is up on :8081")
            else:
                log("WARNING: V100 user sim not healthy, skipping tau2")
                mr["tau2"] = {"reward": None, "error": "user sim unavailable"}
                save_progress(progress)
                mr["status"] = "completed"
                save_progress(progress)
                return
        except:
            log("WARNING: V100 user sim not reachable on :8081, skipping tau2")
            mr["tau2"] = {"reward": None, "error": "user sim unreachable"}
            save_progress(progress)
            mr["status"] = "completed"
            save_progress(progress)
            return

        log("Starting tau2-bench...")
        tau2 = run_tau2()
        mr["tau2"] = tau2
        log(f"tau2 reward: {tau2.get('reward')} ({tau2.get('wall_time_s',0)/60:.0f} min)")
        save_progress(progress)

        mr["status"] = "completed"

    except Exception as e:
        log(f"ERROR: {e}")
        mr["status"] = "error"
        mr["error"] = str(e)
    finally:
        mr["end_time"] = datetime.now(timezone.utc).isoformat()
        kill_model(proc, logf)
        save_progress(progress)

    # Summary
    log("\n" + "=" * 60)
    log("RESULTS SUMMARY")
    log("=" * 60)
    log(f"  Model:      {MODEL_NAME}")
    log(f"  VRAM:       {mr.get('vram_mib', '?')} MiB / 12288 MiB")
    log(f"  Decode:     {mr.get('decode_tps', '?')} tok/s")
    log(f"  LCB pass@1: {mr.get('livecodebench', {}).get('pass_at_1', '?')}")
    log(f"  tau2 reward:{mr.get('tau2', {}).get('reward', '?')}")


if __name__ == "__main__":
    main()
