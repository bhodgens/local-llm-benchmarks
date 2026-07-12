#!/usr/bin/env python3
"""
Run LiveCodeBench on each model sequentially.
Uses OpenAI-compatible chat completions endpoint against local llama.cpp server.
"""
import subprocess, json, time, os, urllib.request
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"

MODELS = [
    {"name": "LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M",
     "file": "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf",
     "lcb_model": "local/lfm-coder-v2",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "LFM2.5-8B-A1B base Q4_K_M",
     "file": "LFM2.5-8B-A1B-Q4_K_M.gguf",
     "lcb_model": "local/lfm-base",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "LFM2.5-8B-A1B Q6_K",
     "file": "LFM2.5-8B-A1B-Q6_K.gguf",
     "lcb_model": "local/lfm-q6",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "qwen2.5-coder-14b-instruct Q4_K_M",
     "file": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
     "lcb_model": "local/qwen-coder-14b",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M",
     "file": "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
     "lcb_model": "local/qwopus-35b-coder",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M",
     "file": "Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf",
     "lcb_model": "local/qwen35-abliterated",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "RavenX-OpenFable-Holo3 Q4_K_M",
     "file": "RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
     "lcb_model": "local/ravenx-holo3",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    # New models - batch 2
    {"name": "DeepSeek-R1-0528-Qwen3-8B Q8_0",
     "file": "DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
     "lcb_model": "local/deepseek-r1-qwen3-8b-q8",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"]},
    {"name": "DeepSeek-R1-0528-Qwen3-8B Q4_K_M",
     "file": "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
     "lcb_model": "local/deepseek-r1-qwen3-8b-q4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma4-coding Q4_K_M",
     "file": "gemma4-coding-Q4_K_M.gguf",
     "lcb_model": "local/gemma4-coding-q4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma4-v2 Q4_K_M",
     "file": "gemma4-v2-Q4_K_M.gguf",
     "lcb_model": "local/gemma4-v2-q4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma-4-12B-it-QAT Q4_0",
     "file": "gemma-4-12B-it-QAT-Q4_0.gguf",
     "lcb_model": "local/gemma4-it-qat-q4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "DeepSeek-Coder-V2-Lite IQ4_XS",
     "file": "DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf",
     "lcb_model": "local/deepseek-coder-v2-lite",
     "args": ["--gpu-layers","99","--ctx-size","16384","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"]},
    {"name": "RavenX-OpenFable-Coderagent gemma4 Q4_K_M",
     "file": "RavenX-OpenFable-Coderagent-gemma4-fable5-Q4_K_M.gguf",
     "lcb_model": "local/ravenx-coderagent-gemma4",
     "args": ["--gpu-layers","99","--ctx-size","131072","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwen3.6-35B-A3B IQ3_K_R4",
     "file": "Qwen3.6-35B-A3B-IQ3_K_R4.gguf",
     "lcb_model": "local/qwen35-iq3",
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
           "--parallel", "1", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_lcb_server.log"), "w")
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
            with open(os.path.join(LOGS, f"{safe}_lcb_server.log")) as f:
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

def record_failure_lcb(progress, model_name, error, settings):
    """Record a failure for the LCB report"""
    for m in progress.get("models", []):
        if m["name"] == model_name:
            if "failures" not in m:
                m["failures"] = []
            m["failures"].append({
                "benchmark": "livecodebench",
                "error": str(error)[:500],
                "settings": settings,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            break

def run_livecodebench(model_name, lcb_model):
    """Run LiveCodeBench code generation scenario"""
    safe = model_name.replace(" ", "_")
    logpath = os.path.join(LOGS, f"{safe}_lcb.log")

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
        "--use_cache",
    ]

    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    print(f"  Running LiveCodeBench...")
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

    # Parse results from output directory and copy to nothink backup
    pass1 = None
    output_dir = os.path.join(LCB_DIR, "output", lcb_model.replace("/", "_"))
    nothink_dir = os.path.join(SCRATCH, "results", "lcb_thinking_off")
    os.makedirs(nothink_dir, exist_ok=True)
    
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            src = os.path.join(root, f)
            rel = os.path.relpath(src, output_dir)
            dst_dir = os.path.join(nothink_dir, lcb_model.replace("/", "_"))
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, rel)
            import shutil
            shutil.copy2(src, dst)
            
            if f.endswith("_eval.json"):
                with open(src) as rf:
                    try:
                        data = json.load(rf)
                        if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                            pass1 = data[0].get("pass@1")
                    except:
                        pass

    # Fallback: scan log
    if pass1 is None:
        with open(logpath) as f:
            content = f.read()
        for line in content.split("\n"):
            if "pass@1" in line.lower() or "accuracy" in line.lower():
                import re
                m = re.search(r'(\d+\.?\d*)\s*%', line)
                if m:
                    pass1 = float(m.group(1)) / 100
                    break

    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1),
            "exit_code": result.returncode}

def main():
    progress = load_progress()

    # Check for existing results - skip if already has LCB
    done = set()
    for m in progress["models"]:
        if "livecodebench" in m and m.get("status") not in ("failed", "error"):
            done.add(m["name"])

    for model in MODELS:
        if model["name"] in done:
            print(f"SKIP: {model['name']} (LCB already done)")
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

        stop_prod()
        proc, logf, err = start_model(model)
        if err:
            record_failure_lcb(progress, model["name"], err, model.get("args", []))
            mr["lcb_error"] = err
            save_progress(progress)
            start_prod()
            continue

        try:
            lcb = run_livecodebench(model["name"], model["lcb_model"])
            mr["livecodebench"] = lcb
            print(f"  LCB pass@1: {lcb.get('pass_at_1')}")
        except Exception as e:
            record_failure_lcb(progress, model["name"], e, model.get("args", []))
            mr["lcb_error"] = str(e)
            print(f"  ERROR: {e}")
        finally:
            kill_model(proc, logf)
            save_progress(progress)

    start_prod()
    print(f"\n{'='*70}\nLIVECODEBENCH COMPLETE\n{'='*70}")
    for m in progress["models"]:
        lcb = m.get("livecodebench", {})
        he = m.get("humaneval", {})
        print(f"  {m['name']:<45} HE={he.get('pass_at_1','?')}  LCB={lcb.get('pass_at_1','?')}  tok/s={m.get('decode_tps','?')}")

if __name__ == "__main__":
    main()
