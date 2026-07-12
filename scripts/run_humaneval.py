#!/usr/bin/env python3
"""
Phase 1: Run HumanEval on all models. LiveCodeBench added after.
Uses lm-evaluation-harness local-completions model against llama.cpp server.
"""
import subprocess, json, time, os, sys, urllib.request
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"

# tokenizer: a valid HF model name for lm-eval's tokenizer backend
# (used only for token counting, actual tokenization happens server-side)
MODELS = [
    {"name": "LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M",
     "file": "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "LFM2.5-8B-A1B base Q4_K_M",
     "file": "LFM2.5-8B-A1B-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "LFM2.5-8B-A1B Q6_K",
     "file": "LFM2.5-8B-A1B-Q6_K.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "qwen2.5-coder-14b-instruct Q4_K_M",
     "file": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
     "tokenizer": "Qwen/Qwen2.5-Coder-14B-Instruct",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M",
     "file": "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwen3.6-35B-A3B IQ3_K_R4",
     "file": "Qwen3.6-35B-A3B-IQ3_K_R4.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M",
     "file": "Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--cpu-moe","--ctx-size","262144","--ubatch-size","512",
              "--threads","6","--threads-batch","6","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "RavenX-OpenFable-Holo3 Q4_K_M",
     "file": "RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
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

def start_proxy():
    """Start the completions proxy on port 18098"""
    proxy_cmd = ["/home/caimlas/bench-venv/bin/python3", 
                 "/home/caimlas/llm-benchmarks/scripts/completions_proxy.py", "18098"]
    proxy_proc = subprocess.Popen(proxy_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    return proxy_proc

def stop_proxy(proxy_proc):
    if proxy_proc:
        proxy_proc.terminate()
        try: proxy_proc.wait(timeout=5)
        except: proxy_proc.kill()

def start_model(model):
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [BINARY, "--model", path, "--device", "CUDA0", "--flash-attn", "on",
           "--batch-size", "2048", "--host", "127.0.0.1", "--port", str(PORT),
           "--parallel", "1", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_server.log"), "w")
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
            with open(os.path.join(LOGS, f"{safe}_server.log")) as f:
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

def probe():
    payload = json.dumps({"prompt":"Write a Python function that checks if a number is prime.",
                          "n_predict":64, "temperature":0.0, "stream":False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    t = json.loads(resp.read()).get("timings", {})
    return (round(t.get("predicted_n",1)/(t.get("predicted_ms",1)/1000), 1),
            round(t.get("prompt_n",1)/(t.get("prompt_ms",1)/1000), 1))

def run_humaneval(name, tokenizer):
    outdir = os.path.join(RESULTS, name.replace(" ", "_"), "humaneval")
    os.makedirs(outdir, exist_ok=True)
    safe = name.replace(" ", "_")
    logpath = os.path.join(LOGS, f"{safe}_humaneval.log")

    # Use our proxy that converts OpenAI format to llama.cpp native /completion
    # This bypasses the /v1/completions content validation that rejects some model outputs
    model_args = f"model={name},base_url=http://127.0.0.1:18098/v1/completions,num_concurrent=1,tokenizer={tokenizer},max_length=16384"

    cmd = [BENCH_PY, "-m", "lm_eval",
           "--model", "local-completions",
           "--model_args", model_args,
           "--tasks", "humaneval",
           "--gen_kwargs", "temperature=0.0,max_gen_toks=1024",
           "--batch_size", "1",
           "--output_path", outdir,
           "--confirm_run_unsafe_code",
           "--log_samples"]

    print(f"  Running HumanEval...")
    start = time.time()
    env = dict(os.environ)
    env["HF_ALLOW_CODE_EVAL"] = "1"  # Required for HumanEval code execution
    with open(logpath, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200, env=env)
    elapsed = time.time() - start

    # Parse results
    pass1 = None
    for root, dirs, files in os.walk(outdir):
        for f in files:
            if f.startswith("results") and f.endswith(".json"):
                with open(os.path.join(root, f)) as rf:
                    data = json.load(rf)
                try:
                    # Try multiple possible key names
                    he_results = data["results"]["humaneval"]
                    for key in ["pass@1,none", "acc,none", "pass@1,create_test", "pass@1"]:
                        if key in he_results:
                            pass1 = he_results[key]
                            break
                except (KeyError, TypeError):
                    pass

    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1),
            "exit_code": result.returncode}

def main():
    progress = load_progress()
    done = {m["name"] for m in progress["models"] if m.get("status") in ("completed", "gated")}

    for model in MODELS:
        if model["name"] in done:
            print(f"SKIP: {model['name']} (already done)")
            continue

        print(f"\n{'='*70}\n  {model['name']}\n{'='*70}")

        mr = {"name": model["name"], "file": model["file"],
              "status": "running",
              "start_time": datetime.now(timezone.utc).isoformat()}

        stop_prod()
        proxy_proc = start_proxy()
        proc, logf, err = start_model(model)
        if err:
            mr["status"] = "failed"
            mr["error"] = err
            mr["end_time"] = datetime.now(timezone.utc).isoformat()
            progress["models"] = [m for m in progress["models"] if m["name"] != model["name"]] + [mr]
            save_progress(progress)
            kill_model(proc, logf)
            stop_proxy(proxy_proc)
            start_prod()
            continue

        try:
            print("  Throughput probe...")
            dec_tps, prompt_tps = probe()
            mr["decode_tps"] = dec_tps
            mr["prompt_tps"] = prompt_tps
            print(f"  Decode: {dec_tps} tok/s")

            # HumanEval
            he = run_humaneval(model["name"], model["tokenizer"])
            mr["humaneval"] = he
            print(f"  HumanEval pass@1: {he['pass_at_1']}")

            if dec_tps < 24:
                mr["status"] = "gated"
                mr["reason"] = f"Throughput {dec_tps} < 24 tok/s"
            else:
                mr["status"] = "humaneval_done"

            mr["end_time"] = datetime.now(timezone.utc).isoformat()
        except Exception as e:
            mr["status"] = "error"
            mr["error"] = str(e)
            print(f"  ERROR: {e}")
        finally:
            kill_model(proc, logf)
            stop_proxy(proxy_proc)
            progress["models"] = [m for m in progress["models"] if m["name"] != model["name"]] + [mr]
            save_progress(progress)

    # Restart production
    print("\nRestarting production...")
    start_prod()

    # Summary
    print(f"\n{'='*70}\nHUMANEVAL COMPLETE\n{'='*70}")
    for m in sorted(progress["models"], key=lambda x: x.get("humaneval",{}).get("pass_at_1") or 0, reverse=True):
        he = m.get("humaneval", {})
        print(f"  {m['name']:<45} {m.get('status','?'):<12} "
              f"pass@1={he.get('pass_at_1','?')}  tok/s={m.get('decode_tps','?')}")

if __name__ == "__main__":
    main()
