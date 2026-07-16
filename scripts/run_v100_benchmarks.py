#!/usr/bin/env python3
"""
Run LCB + tau2 + tok/s probe for new models on the V100 (32GB).
Agent model runs on V100 (CUDA0, device 0).
User simulator for tau2: LFM on :8082 (3060, stays up).

Supports both upstream llama.cpp and PrismML fork (for Ternary-Bonsai dspark).
"""
import subprocess, json, time, os, urllib.request, shutil, re
from datetime import datetime, timezone

BINARY_UPSTREAM = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
BINARY_PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099  # Benchmark model on V100
USER_PORT = 8082  # User simulator: LFM on 3060
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

# V100 service to stop during benchmarks
V100_SERVICE = "caimlas-coder"

MODELS = [
    {
        "name": "Qwen3.5-9B-DeepSeek-V4-Flash Q4_K_M",
        "file": "Qwen3.5-9B-DeepSeek-V4-Flash-Q4_K_M.gguf",
        "lcb_model": "local/qwen35-9b-dsv4",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "9B",
        "thinking": True,
    },
    {
        "name": "gemma-4-12B-it-QAT Q4_0",
        "file": "gemma-4-12B-it-QAT-Q4_0.gguf",
        "lcb_model": "local/gemma4-it-qat-q4",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "12-14B",
        "thinking": True,
    },
    {
        "name": "gemma-4-26B-A4B-it-QAT Q4_0",
        "file": "gemma-4-26B-A4B-it-QAT-Q4_0.gguf",
        "lcb_model": "local/gemma4-26b-a4b-qat",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "26B MoE",
        "thinking": True,
    },
    {
        "name": "Ternary-Bonsai-27B Q2_0 (dspark)",
        "file": "Ternary-Bonsai-27B-Q2_0.gguf",
        "draft_file": "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
        "lcb_model": "local/ternary-bonsai-27b",
        "binary": BINARY_PRISMML,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"],
        "category": "27B Ternary",
        "thinking": True,
    },
    {
        "name": "Qwen3.6-27B-MTP Q4_K_M",
        "file": "Qwen3.6-27B-MTP-Q4_K_M.gguf",
        "lcb_model": "local/qwen36-27b-mtp",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "27B",
        "thinking": True,
    },
]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def stop_v100():
    subprocess.run(["sudo", "systemctl", "stop", V100_SERVICE], capture_output=True, timeout=15)
    time.sleep(3)

def start_v100():
    subprocess.run(["sudo", "systemctl", "start", V100_SERVICE], capture_output=True, timeout=15)
    for _ in range(60):
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://localhost:8081/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return
        except:
            pass

def start_model(model, use_draft=False):
    binary = model.get("binary", BINARY_UPSTREAM)
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [binary, "--model", path, "--flash-attn", "on",
           "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])

    # Add draft model for speculative decoding if requested
    if use_draft and "draft_file" in model:
        draft_path = os.path.join(LLMS_DIR, model["draft_file"])
        cmd.extend(["--spec-draft-model", draft_path, "--spec-draft-n-max", "4"])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # V100
    safe = model["name"].replace(" ", "_")
    suffix = "_dspark" if use_draft else ""
    logf = open(os.path.join(LOGS, f"{safe}_v100_server{suffix}.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{safe}_v100_server{suffix}.log")) as f:
                err = f.read()[-500:]
            return None, None, f"Server died: {err}"
    proc.kill()
    return None, None, "Timeout"

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

def probe_tps(model, use_draft=False):
    """Quick tok/s probe: send a 256-token completion request at 8K context."""
    import http.client
    safe = model["name"].replace(" ", "_")
    suffix = "_dspark" if use_draft else ""
    print(f"  Probing decode tok/s...")

    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
        # Warmup
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Write a short hello world function."}],
            "max_tokens": 8,
            "temperature": 0.0,
            "stream": False,
        })
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        resp.read()

        # Timed decode: 256 tokens
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Write a detailed essay about the history of computing, from Babbage to modern GPUs. Include key milestones, important figures, and technological breakthroughs."}],
            "max_tokens": 256,
            "temperature": 0.0,
            "stream": False,
        })
        start = time.time()
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        elapsed = time.time() - start

        # Parse usage stats
        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 256)
        tps = completion_tokens / elapsed if elapsed > 0 else 0

        # Also check if server reports timings
        stats = ""
        try:
            health = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=3)
            props = json.loads(health.read())
        except:
            pass

        print(f"  tok/s: {tps:.1f} ({completion_tokens} tokens in {elapsed:.1f}s)")
        return round(tps, 1)
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None

def run_livecodebench(model_name, lcb_model):
    safe = model_name.replace(" ", "_")
    logpath = os.path.join(LOGS, f"{safe}_v100_lcb.log")

    # Clear previous output for this model
    output_dir = os.path.join(LCB_DIR, "output", lcb_model.replace("/", "_"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", lcb_model,
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

    print(f"  Running LiveCodeBench (75 problems, thinking off)...")
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

    # Parse results
    pass1 = None
    empty_count = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            src = os.path.join(root, f)
            if f.endswith("_eval.json"):
                with open(src) as rf:
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

def run_tau2(model_name, model_file, max_steps=30):
    safe = model_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
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
        "--max-steps", str(max_steps),
        "--max-errors", "5",
        "--timeout", "300",
        "--seed", "42",
        "--save-to", save_dir,
    ]

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print(f"  Running tau2-bench (airline, 15 tasks)...")
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

    # Parse reward from output
    reward = None
    task_pass = None
    with open(logpath) as f:
        content = f.read()
    reward_match = re.search(r'Average Reward\s+([\d.]+)', content)
    if reward_match:
        reward = float(reward_match.group(1))
    pass_match = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    task_pass = float(pass_match.group(1)) if pass_match else None

    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {
        "reward": reward,
        "task_pass_rate": task_pass,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
    }

def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    progress = load_progress()

    done_names = set()
    for m in progress["models"]:
        if "livecodebench" in m and m.get("livecodebench", {}).get("pass_at_1") is not None:
            done_names.add(m["name"])

    for model in MODELS:
        if model["name"] in done_names:
            print(f"SKIP: {model['name']} (already done)")
            continue

        print(f"\n{'='*70}")
        print(f"  {model['name']}")
        print(f"{'='*70}")

        # Find or create model entry
        mr = None
        for m in progress["models"]:
            if m["name"] == model["name"]:
                mr = m
                break
        if mr is None:
            mr = {"name": model["name"], "file": model["file"], "category": model.get("category", "")}
            progress["models"].append(mr)

        # Check user simulator is up (LFM on 3060)
        try:
            r = urllib.request.urlopen(f"http://localhost:{USER_PORT}/health", timeout=3)
            health = json.loads(r.read())
            if health.get("status") != "ok":
                print(f"  WARNING: User simulator on :{USER_PORT} not healthy!")
        except:
            print(f"  WARNING: User simulator on :{USER_PORT} not reachable!")

        stop_v100()

        # Determine if we should try dspark (draft model) for Bonsai
        use_draft = "draft_file" in model
        use_draft_succeeded = False

        proc, logf, err = start_model(model, use_draft=use_draft)
        if err and use_draft:
            # Retry without draft model
            print(f"  Draft model failed ({err[:100]}), retrying without dspark...")
            use_draft = False
            model_clean = dict(model)
            model_clean["name"] = model["name"].replace(" (dspark)", "")
            proc, logf, err = start_model(model, use_draft=False)

        if err:
            print(f"  FATAL: Server failed to start: {err[:200]}")
            if "failures" not in mr:
                mr["failures"] = []
            mr["failures"].append({
                "benchmark": "all",
                "error": str(err)[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            mr["lcb_error"] = err
            save_progress(progress)
            start_v100()
            continue

        if use_draft:
            print(f"  DSpark speculative decoding active")
            mr["dspark"] = True
        else:
            mr["dspark"] = False

        try:
            # 1. tok/s probe
            tps = probe_tps(model, use_draft=use_draft)
            mr["decode_tps"] = tps
            save_progress(progress)

            # 2. LiveCodeBench
            lcb = run_livecodebench(model["name"], model["lcb_model"])
            mr["livecodebench"] = lcb
            save_progress(progress)

            # 3. tau2-bench
            tau2 = run_tau2(model["name"], model["file"], max_steps=30)
            mr["tau2"] = tau2
            save_progress(progress)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            if "failures" not in mr:
                mr["failures"] = []
            mr["failures"].append({
                "benchmark": "unknown",
                "error": str(e)[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            kill_model(proc, logf)
            save_progress(progress)

        start_v100()

    print(f"\n{'='*70}")
    print(f"V100 BENCHMARKS COMPLETE")
    print(f"{'='*70}")
    for m in sorted(progress["models"], key=lambda x: x.get("livecodebench", {}).get("pass_at_1") or 0, reverse=True):
        lcb = m.get("livecodebench", {})
        tau2 = m.get("tau2", {})
        tps = m.get("decode_tps")
        print(f"  {m['name']:<45} LCB={lcb.get('pass_at_1', '?')}  tau2={tau2.get('reward', '?')}  tok/s={tps}")

if __name__ == "__main__":
    main()
