#!/usr/bin/env python3
"""
Run Aider Polyglot benchmark for 7 models.
Runs inside Docker container for full language support (Python, Go, Rust, JS, Java, C++).
Each model runs on the 3060 via llama.cpp server, Docker connects via host network.
"""
import subprocess, json, time, os, urllib.request, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
AIDER_DIR = "/home/caimlas/git/aider"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results", "aider")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
NUM_TESTS = 10  # 10 exercises per language = 60 total per model

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

MODELS = [
    {"name": "Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M",
     "file": "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "RavenX-OpenFable-Holo3 Q4_K_M",
     "file": "RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma-4-12B-it-QAT Q4_0",
     "file": "gemma-4-12B-it-QAT-Q4_0.gguf",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "qwen2.5-coder-14b-instruct Q4_K_M",
     "file": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
     "args": ["--gpu-layers","99","--ctx-size","8192","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"]},
    {"name": "gemma4-v2 Q4_K_M",
     "file": "gemma4-v2-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M",
     "file": "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "LFM2.5-8B-A1B Q6_K",
     "file": "LFM2.5-8B-A1B-Q6_K.gguf",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
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
           "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
           "--parallel", "1", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_aider_server.log"), "w")
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
            with open(os.path.join(LOGS, f"{safe}_aider_server.log")) as f:
                return None, None, f"Server died: {f.read()[-300:]}"
    proc.kill()
    return None, None, "Timeout"

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def run_aider_in_docker(model_name):
    """Run Aider benchmark inside Docker container"""
    safe = model_name.replace(" ", "_")
    run_name = f"aider_diff_{safe}"
    logpath = os.path.join(LOGS, f"{safe}_aider_diff.log")

    # Docker command - mount aider dir, set env vars, connect to host server
    cmd = [
        "docker", "run", "--rm",
        "--network", "host",
        "-e", "AIDER_DOCKER=1",
        "-e", "OPENAI_API_KEY=none",
        "-e", f"OPENAI_API_BASE=http://127.0.0.1:{PORT}/v1",
        "-e", "HOME=/tmp",
        "-v", f"{AIDER_DIR}:/aider",
        "-v", f"{AIDER_DIR}/tmp.benchmarks:/aider/tmp.benchmarks",
        "-v", f"/home/caimlas/llm-benchmarks/aider_model_settings.yml:/aider/aider_model_settings.yml",
        "-w", "/aider",
        "aider-benchmark",
        "bash", "-c",
        f"git config --global --add safe.directory /aider && "
        f"git config --global --add safe.directory '*' && "
        f"pip install -e '.[dev]' -q 2>/dev/null && "
        f"python3 benchmark/benchmark.py {run_name} "
        f"--model 'openai/{model_name}' "
        f"--edit-format diff "
        f"--threads 1 "
        f"--num-tests {NUM_TESTS} "
        f"--read-model-settings /aider/aider_model_settings.yml "
        f"--exercises-dir polyglot-benchmark"
    ]

    print(f"  Running Aider Polyglot ({NUM_TESTS} exercises per language)...")
    start = time.time()

    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=7200)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_rate": None, "wall_time_s": 7200, "error": "timeout"}
    except Exception as e:
        return {"pass_rate": None, "wall_time_s": time.time() - start, "error": str(e)}

    # Parse results from log
    pass_rate = None
    with open(logpath) as f:
        content = f.read()

    # Look for pass rates
    for pattern in [r'pass_rate_1:\s+([\d.]+)', r'Percent passed:\s+([\d.]+)',
                    r'(\d+)/(\d+)\s+passed']:
        match = re.search(pattern, content)
        if match:
            if len(match.groups()) == 2:
                pass_rate = int(match.group(1)) / int(match.group(2)) if int(match.group(2)) > 0 else 0
            else:
                val = float(match.group(1))
                pass_rate = val if val <= 1.0 else val / 100
            break

    # Extract completion tokens for throughput tracking
    completion_tokens = 0
    match = re.search(r'completion_tokens:\s+(\d+)', content)
    if match:
        completion_tokens = int(match.group(1))

    return {
        "pass_rate": pass_rate,
        "wall_time_s": round(elapsed, 1),
        "completion_tokens": completion_tokens,
        "exit_code": result.returncode,
    }

def main():
    progress = load_progress()

    # Check for existing aider results
    done = set()
    for m in progress["models"]:
        if "aider_diff" in m and m["aider_diff"].get("pass_rate") is not None:
            done.add(m["name"])
            print(f"SKIP: {m['name']} (Aider diff already done)")

    for model in MODELS:
        if model["name"] in done:
            continue

        print(f"\n{'='*70}\n  {model['name']}\n{'='*70}")

        mr = None
        for m in progress["models"]:
            if m["name"] == model["name"]:
                mr = m
                break
        if mr is None:
            mr = {"name": model["name"], "file": model["file"]}
            progress["models"].append(mr)

        stop_prod()
        proc, logf, err = start_model(model)
        if err:
            mr["aider_error"] = err
            save_progress(progress)
            print(f"  FAILED: {err}")
            start_prod()
            continue

        try:
            result = run_aider_in_docker(model["name"])
            mr["aider_diff"] = result
            print(f"  Aider diff pass_rate: {result.get('pass_rate')}")
            print(f"  Wall time: {result.get('wall_time_s',0)/60:.0f} min")
            print(f"  Completion tokens: {result.get('completion_tokens', 0)}")
        except Exception as e:
            mr["aider_diff_error"] = str(e)
            print(f"  ERROR: {e}")
        finally:
            kill_model(proc, logf)
            save_progress(progress)

    start_prod()
    print(f"\n{'='*70}\nAIDER POLYGLOT COMPLETE\n{'='*70}")
    for m in progress["models"]:
        aider = m.get("aider", {})
        pr = aider.get("pass_rate")
        wt = aider.get("wall_time_s", 0)
        pr_str = "%.1f%%" % (pr*100) if pr is not None else "?"
        print(f"  {m['name']:<45} pass_rate={pr_str}  ({wt/60:.0f} min)")

if __name__ == "__main__":
    main()
