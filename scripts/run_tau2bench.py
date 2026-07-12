#!/usr/bin/env python3
"""
Run tau2-bench on each model sequentially on the 3060.
Uses the V100 (port 8081) as the user simulator for all runs.

tau2-bench tests agent tool-use capability: the agent model must use API tools
to solve customer service tasks (booking flights, processing returns, etc).
This is fundamentally different from HumanEval/LiveCodeBench which test code generation.

Domains: airline (50 tasks), retail, telecom
We run airline (50 tasks) as the primary benchmark.
The user simulator runs on the V100, the agent model runs on the 3060.
"""
import subprocess, json, time, os, urllib.request, shutil
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099  # Agent model on 3060
USER_PORT = 8081  # User simulator on V100 (production service)
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
TAU2_DIR = "/home/caimlas/git/tau2-bench"
DOMAIN = "airline"
NUM_TASKS = 15
NUM_TRIALS = 1

# All models to benchmark (agent model on 3060)
# User simulator is always V100 Qwopus on port 8081
MODELS = [
    {"name": "LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M",
     "file": "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "LFM2.5-8B-A1B base Q4_K_M",
     "file": "LFM2.5-8B-A1B-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "LFM2.5-8B-A1B Q6_K",
     "file": "LFM2.5-8B-A1B-Q6_K.gguf",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "qwen2.5-coder-14b-instruct Q4_K_M",
     "file": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"],
     "max_steps": 15},  # Model max ctx is 32K, need fewer steps to avoid truncation
    {"name": "Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M",
     "file": "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M",
     "file": "Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "RavenX-OpenFable-Holo3 Q4_K_M",
     "file": "RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    # New models
    {"name": "DeepSeek-R1-0528-Qwen3-8B Q8_0",
     "file": "DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"],
     "max_steps": 15},
    {"name": "DeepSeek-R1-0528-Qwen3-8B Q4_K_M",
     "file": "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma4-coding Q4_K_M",
     "file": "gemma4-coding-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma4-v2 Q4_K_M",
     "file": "gemma4-v2-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma-4-12B-it-QAT Q4_0",
     "file": "gemma-4-12B-it-QAT-Q4_0.gguf",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "DeepSeek-Coder-V2-Lite IQ4_XS",
     "file": "DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf",
     "args": ["--gpu-layers","99","--ctx-size","16384","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"],
     "max_steps": 10},
    {"name": "RavenX-OpenFable-Coderagent gemma4 Q4_K_M",
     "file": "RavenX-OpenFable-Coderagent-gemma4-fable5-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwen3.6-35B-A3B IQ3_K_R4",
     "file": "Qwen3.6-35B-A3B-IQ3_K_R4.gguf",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def stop_prod():
    subprocess.run(["sudo","systemctl","stop","caimlas-llama"], capture_output=True, timeout=15)
    time.sleep(3)

def start_prod():
    subprocess.run(["sudo","systemctl","start","caimlas-llama"], capture_output=True, timeout=15)
    for _ in range(60):
        time.sleep(3)
        try:
            r = urllib.request.urlopen("http://localhost:8080/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return
        except:
            pass

def start_model(model):
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [BINARY, "--model", path, "--device", "CUDA0", "--flash-attn", "on",
           "--batch-size", "2048", "--host", "127.0.0.1", "--port", str(PORT),
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_tau2_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(120):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            return None, None, "Server died"
    proc.kill()
    return None, None, "Timeout"

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def build_retry_config(model, attempt):
    """Build progressively reduced settings for retry attempts.
    attempt 0 = original, 1 = moderate reduction, 2 = aggressive, 3 = minimal
    """
    is_moe = "--cpu-moe" in model.get("args", [])
    
    configs = [
        # Attempt 1: reduce ctx and KV quant
        {"ctx": "32768", "kv": "q4_0", "ubatch": "256", "steps": 15, "ngl": "99"},
        # Attempt 2: aggressive reduction
        {"ctx": "16384", "kv": "q4_0", "ubatch": "128", "steps": 10, "ngl": "99"},
        # Attempt 3: minimal context
        {"ctx": "8192", "kv": "q4_0", "ubatch": "64", "steps": 5, "ngl": "99"},
    ]
    
    if attempt > len(configs):
        return None
    
    cfg = configs[attempt - 1]
    threads = "6" if is_moe else "4"
    
    args = ["--gpu-layers", cfg["ngl"], "--ctx-size", cfg["ctx"],
            "--ubatch-size", cfg["ubatch"],
            "--threads", threads, "--threads-batch", threads,
            "--cache-type-k", cfg["kv"], "--cache-type-v", cfg["kv"]]
    
    if is_moe:
        args.insert(2, "--cpu-moe")
    
    return {"args": args, "max_steps": cfg["steps"]}

def record_failure(progress, model_name, benchmark, error, settings, attempt):
    """Record a failure attempt for the report"""
    for m in progress.get("models", []):
        if m["name"] == model_name:
            if "failures" not in m:
                m["failures"] = []
            m["failures"].append({
                "benchmark": benchmark,
                "attempt": attempt,
                "error": str(error)[:500],
                "settings": settings,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            break

def run_tau2(model_name, model_file, max_steps=30):
    """Run tau2-bench airline domain"""
    safe = model_name.replace(" ", "_").replace("/", "_")
    logpath = os.path.join(LOGS, f"{safe}_tau2.log")
    save_dir = f"tau2_{safe}"

    # Use the model name as the agent LLM identifier
    # tau2 uses LiteLLM, so we pass openai/<name> with api_base
    agent_model = f"openai/{safe}"
    
    # User simulator: always V100 Qwopus
    user_model = "openai/Qwopus3.6-27B-Coder-Compat-MTP"

    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", DOMAIN,
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
        "--num-tasks", str(NUM_TASKS),
        "--num-trials", str(NUM_TRIALS),
        "--max-concurrency", "2",
        "--max-steps", str(max_steps),
        "--max-errors", "5",
        "--timeout", "300",
        "--seed", "42",
        "--save-to", save_dir,
    ]

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print(f"  Running tau2-bench ({DOMAIN}, {NUM_TASKS} tasks)...")
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

    # Parse results from simulation output
    reward = None
    task_pass = None
    sim_dir = os.path.join(TAU2_DIR, "data", "simulations", save_dir)
    
    # tau2 saves results in data/simulations/<save_dir>/
    # Look for metrics in the output
    if os.path.exists(logpath):
        with open(logpath) as f:
            content = f.read()
        # Parse reward from the metrics display
        import re
        reward_match = re.search(r'Average Reward\s+([\d.]+)', content)
        if reward_match:
            reward = float(reward_match.group(1))
        
        # Also try task_pass_rate
        pass_match = re.search(r'Task Pass Rate\s+([\d.]+)', content)
        task_pass = float(pass_match.group(1)) if pass_match else None

    return {
        "reward": reward,
        "task_pass_rate": task_pass,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
    }

def main():
    progress = load_progress()

    # Skip models that already have tau2 results
    done = set()
    for m in progress["models"]:
        if "tau2" in m and m.get("tau2", {}).get("reward") is not None:
            done.add(m["name"])

    for model in MODELS:
        if model["name"] in done:
            print(f"SKIP: {model['name']} (tau2 already done)")
            continue

        print(f"\n{'='*70}\n  {model['name']}\n{'='*70}")

        # Find or create model entry
        mr = None
        for m in progress["models"]:
            if m["name"] == model["name"]:
                mr = m
                break
        if mr is None:
            mr = {"name": model["name"], "file": model["file"]}
            progress["models"].append(mr)

        # Check V100 is up (user simulator)
        try:
            r = urllib.request.urlopen(f"http://localhost:{USER_PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") != "ok":
                print("  WARNING: V100 service not available for user simulator!")
        except:
            print("  WARNING: V100 service not available for user simulator!")

        stop_prod()
        
        # Adaptive retry: try progressively reduced settings on failure
        retry_queue = [(model, None)]  # (model_config, retry_config_override)
        attempt = 0
        max_attempts = 3
        
        while retry_queue and attempt < max_attempts:
            orig_model, override = retry_queue.pop(0)
            current_model = dict(orig_model)
            
            if override:
                # Apply reduced settings
                current_model["args"] = override["args"]
                current_model["max_steps"] = override["max_steps"]
                print(f"  RETRY attempt {attempt+1}/{max_attempts} with reduced settings...")
            
            proc, logf, err = start_model(current_model)
            if err:
                attempt += 1
                record_failure(progress, model["name"], "tau2", err, {"ctx": "current", "steps": current_model.get("max_steps",30)}, attempt)
                mr["tau2_error"] = err
                save_progress(progress)
                print(f"  Server failed (attempt {attempt}): {err[:100]}")
                
                # Build reduced config for retry
                if attempt < max_attempts:
                    retry_cfg = build_retry_config(orig_model, attempt)
                    if retry_cfg:
                        retry_queue.append((orig_model, retry_cfg))
                        continue
                start_prod()
                break

            try:
                result = run_tau2(current_model["name"], current_model["file"], current_model.get("max_steps", 30))
                if result.get("reward") is not None:
                    mr["tau2"] = result
                    mr.pop("tau2_error", None)
                    print(f"  tau2 reward: {result.get('reward')}")
                    print(f"  wall_time: {result.get('wall_time_s',0)/60:.0f} min")
                    break
                else:
                    # Run "succeeded" but reward is None - usually all tasks failed
                    attempt += 1
                    record_failure(progress, model["name"], "tau2", err_msg, {"ctx": "current", "steps": current_model.get("max_steps",30)}, attempt)
                    err_msg = result.get("error", "all tasks failed")
                    mr["tau2_error"] = err_msg
                    print(f"  No valid reward (attempt {attempt}): {err_msg[:80]}")
                    if attempt < max_attempts:
                        retry_cfg = build_retry_config(orig_model, attempt)
                        if retry_cfg:
                            retry_queue.append((orig_model, retry_cfg))
            except Exception as e:
                attempt += 1
                record_failure(progress, model["name"], "tau2", e, {"ctx": "current", "steps": current_model.get("max_steps",30)}, attempt)
                mr["tau2_error"] = str(e)
                print(f"  ERROR (attempt {attempt}): {e}")
                if attempt < max_attempts:
                    retry_cfg = build_retry_config(orig_model, attempt)
                    if retry_cfg:
                        retry_queue.append((orig_model, retry_cfg))
            finally:
                kill_model(proc, logf)
                save_progress(progress)

    start_prod()
    print(f"\n{'='*70}\nTAU2-BENCH COMPLETE\n{'='*70}")
    for m in sorted(progress["models"], key=lambda x: x.get("tau2",{}).get("reward") or 0, reverse=True):
        t = m.get("tau2", {})
        he = m.get("humaneval", {})
        lcb = m.get("livecodebench", {})
        print(f"  {m['name']:<45} tau2={t.get('reward','?')}  HE={he.get('pass_at_1','?')}  LCB={lcb.get('pass_at_1','?')}")

if __name__ == "__main__":
    main()
