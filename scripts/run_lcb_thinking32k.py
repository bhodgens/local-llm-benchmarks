#!/usr/bin/env python3
"""
Run LiveCodeBench on MoE models with thinking ENABLED and 32K token budget.
Only runs the 3 Qwen3.6-architecture MoE models.
"""
import subprocess, json, time, os, urllib.request, shutil
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(LOGS_DIR := os.path.join(SCRATCH, "logs"))
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"

MOE_MODELS = [
    {"name": "Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M (think-32k)",
     "file": "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
     "lcb_model": "local/qwopus-35b-coder",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M (think-32k)",
     "file": "Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf",
     "lcb_model": "local/qwen35-abliterated",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "RavenX-OpenFable-Holo3 Q4_K_M (think-32k)",
     "file": "RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
     "lcb_model": "local/ravenx-holo3",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
]

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
           "--parallel", "1", "--temp", "0.0", "-n", "32768"]
    cmd.extend(model["args"])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_think32k_server.log"), "w")
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

def run_lcb_thinking32k(model_name, lcb_model):
    safe = model_name.replace(" ", "_")
    logpath = os.path.join(LOGS, f"{safe}_think32k.log")

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", lcb_model,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1",
        "--temperature", "0.0",
        "--max_tokens", "32768",
        "--num_problems", "75",
        "--openai_timeout", "3600",
        "--evaluate",
    ]

    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"

    print(f"  Running LiveCodeBench (thinking ON, 32K tokens)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=72000, env=env, cwd=LCB_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 72000, "error": "timeout"}
    except Exception as e:
        return {"pass_at_1": None, "wall_time_s": time.time() - start, "error": str(e)}

    # Parse results and copy to thinking_on_32k backup
    pass1 = None
    output_dir = os.path.join(LCB_DIR, "output", lcb_model.replace("/", "_"))
    backup_dir = os.path.join(SCRATCH, "results", "lcb_thinking_on_32k")
    os.makedirs(backup_dir, exist_ok=True)

    for root, dirs, files in os.walk(output_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, output_dir)
            dst = os.path.join(backup_dir, lcb_model.replace("/", "_"))
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(src, os.path.join(dst, rel))

            if f.endswith("_eval.json"):
                with open(src) as rf:
                    try:
                        data = json.load(rf)
                        if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                            pass1 = data[0].get("pass@1")
                    except:
                        pass

    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1),
            "exit_code": result.returncode, "thinking": True, "max_tokens": 32768}

def main():
    for model in MOE_MODELS:
        print(f"\n{'='*70}\n  {model['name']}\n{'='*70}")
        stop_prod()
        proc, logf, err = start_model(model)
        if err:
            print(f"  FAILED: {err}")
            start_prod()
            continue
        try:
            result = run_lcb_thinking32k(model["name"], model["lcb_model"])
            print(f"  pass@1: {result.get('pass_at_1')}")
            print(f"  wall_time: {result.get('wall_time_s',0)/60:.0f} min")
        except Exception as e:
            print(f"  ERROR: {e}")
        finally:
            kill_model(proc, logf)

    start_prod()
    print(f"\n{'='*70}\nTHINKING-ON 32K COMPLETE\n{'='*70}")

if __name__ == "__main__":
    main()
