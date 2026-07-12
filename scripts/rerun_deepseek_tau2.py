#!/usr/bin/env python3
"""
Targeted tau2 rerun for DeepSeek-R1 models with reduced VRAM settings.
Waits for the main tau2 run to free the 3060, then runs these 2 models.
"""
import subprocess, json, time, os, urllib.request, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
USER_PORT = 8081
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
TAU2_DIR = "/home/caimlas/git/tau2-bench"

# Safe settings: reduced ctx + q4_0 KV to avoid OOM
DEEPSEEK_MODELS = [
    {"name": "DeepSeek-R1-0528-Qwen3-8B Q8_0",
     "file": "DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
     "args": ["--gpu-layers","99","--ctx-size","16384","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"],
     "max_steps": 10},
    {"name": "DeepSeek-R1-0528-Qwen3-8B Q4_K_M",
     "file": "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"],
     "max_steps": 15},
]

def wait_for_port_free(port, timeout=86400):
    """Wait until port is no longer responding (previous run finished)"""
    print("Waiting for port %d to be free..." % port)
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen("http://127.0.0.1:%d/health" % port, timeout=2)
            time.sleep(30)
        except:
            print("Port %d is free!" % port)
            return True
    return False

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
           "--parallel", "1", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, "%s_tau2_rerun_server.log" % safe), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(120):
        time.sleep(2)
        try:
            r = urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, "%s_tau2_rerun_server.log" % safe)) as f:
                return None, None, "Server died: %s" % f.read()[-300:]
    proc.kill()
    return None, None, "Timeout"

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def run_tau2(model_name, max_steps):
    safe = model_name.replace(" ", "_").replace("/", "_")
    logpath = os.path.join(LOGS, "%s_tau2_rerun.log" % safe)
    save_dir = "tau2_rerun_%s" % safe

    agent_model = "openai/%s" % safe
    user_model = "openai/Qwopus3.6-27B-Coder-Compat-MTP"

    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", agent_model,
        "--agent-llm-args", json.dumps({
            "api_key": "none",
            "api_base": "http://127.0.0.1:%d/v1" % PORT,
            "temperature": 0.0,
        }),
        "--user-llm", user_model,
        "--user-llm-args", json.dumps({
            "api_key": "none",
            "api_base": "http://localhost:%d/v1" % USER_PORT,
        }),
        "--num-tasks", "15",
        "--num-trials", "1",
        "--max-concurrency", "1",
        "--max-steps", str(max_steps),
        "--max-errors", "5",
        "--timeout", "300",
        "--seed", "42",
        "--save-to", save_dir,
    ]

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print("  Running tau2-bench (airline, 15 tasks, max_steps=%d)..." % max_steps)
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
    reward_match = re.search(r'Average Reward\s+([\d.]+)', content)
    if reward_match:
        reward = float(reward_match.group(1))
    pass_match = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    task_pass = float(pass_match.group(1)) if pass_match else None

    return {
        "reward": reward,
        "task_pass_rate": task_pass,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "rerun": True,
    }

def main():
    # Wait for the main tau2 run to finish (port 18099 to be free)
    # But also check if it already finished
    try:
        urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2)
        # Port is in use - main tau2 still running
        if not wait_for_port_free(PORT, timeout=86400):
            print("Main tau2 run still going after 24h. Aborting DeepSeek rerun.")
            return
    except:
        print("Port already free - main tau2 done")

    # Load progress and clear old DeepSeek errors
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)
    for m in progress["models"]:
        if "DeepSeek-R1" in m["name"]:
            if "tau2_error" in m:
                del m["tau2_error"]
            if "tau2" in m:
                del m["tau2"]
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)

    for model in DEEPSEEK_MODELS:
        print("\n" + "="*70)
        print("  %s" % model["name"])
        print("="*70)

        mr = None
        for m in progress["models"]:
            if m["name"] == model["name"]:
                mr = m
                break
        if mr is None:
            mr = {"name": model["name"], "file": model["file"]}
            progress["models"].append(mr)

        stop_prod()
        attempt = 0
        max_attempts = 3
        retry_args = None
        
        while attempt < max_attempts:
            attempt += 1
            current_args = retry_args if retry_args else model["args"]
            current_steps = model.get("max_steps", 15)
            
            # Start model with current settings
            path = os.path.join(LLMS_DIR, model["file"])
            cmd = [BINARY, "--model", path, "--device", "CUDA0", "--flash-attn", "on",
                   "--batch-size", "2048", "--host", "127.0.0.1", "--port", str(PORT),
                   "--parallel", "1", "--temp", "0.0", "-n", "4096"]
            cmd.extend(current_args)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = "1"
            safe = model["name"].replace(" ", "_")
            logf = open(os.path.join(LOGS, "%s_tau2_rerun_server.log" % safe), "w")
            proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
            
            # Wait for server
            err = None
            for i in range(120):
                time.sleep(2)
                try:
                    r = urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=3)
                    if json.loads(r.read()).get("status") == "ok":
                        break
                except:
                    pass
                if proc.poll() is not None:
                    logf.close()
                    with open(os.path.join(LOGS, "%s_tau2_rerun_server.log" % safe)) as f:
                        err = "Server died: %s" % f.read()[-200:]
                    break
            else:
                proc.kill()
                err = "Timeout waiting for server"
            
            if err:
                print("  Attempt %d/%d FAILED: %s" % (attempt, max_attempts, err[:100]))
                # Build reduced config
                if attempt < max_attempts:
                    if attempt == 1:
                        retry_args = ["--gpu-layers","99","--ctx-size","8192","--ubatch-size","128",
                                      "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"]
                        current_steps = 8
                    elif attempt == 2:
                        retry_args = ["--gpu-layers","80","--ctx-size","4096","--ubatch-size","64",
                                      "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"]
                        current_steps = 5
                    print("  Retrying with reduced settings...")
                    continue
                else:
                    mr["tau2_error"] = err
                    with open(PROGRESS_FILE, "w") as f:
                        json.dump(progress, f, indent=2)
                    break
            
            # Run tau2
            try:
                result = run_tau2(model["name"], current_steps)
                if result.get("reward") is not None:
                    mr["tau2"] = result
                    mr.pop("tau2_error", None)
                    print("  tau2 reward: %s" % result.get("reward"))
                    break
                else:
                    err = result.get("error", "all tasks failed")
                    print("  Attempt %d/%d: no valid reward: %s" % (attempt, max_attempts, str(err)[:80]))
                    if attempt < max_attempts:
                        retry_args = ["--gpu-layers","99","--ctx-size","8192","--ubatch-size","128",
                                      "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"]
                        current_steps = 8
                        continue
                    else:
                        mr["tau2_error"] = str(err)
            except Exception as e:
                print("  Attempt %d/%d ERROR: %s" % (attempt, max_attempts, e))
                mr["tau2_error"] = str(e)
            finally:
                kill_model(proc, logf)
                with open(PROGRESS_FILE, "w") as f:
                    json.dump(progress, f, indent=2)
        
        if proc:
            kill_model(proc, logf)

    start_prod()
    print("\n" + "="*70)
    print("DEEPSEEK TAU2 RERUN COMPLETE")
    print("="*70)

if __name__ == "__main__":
    main()
