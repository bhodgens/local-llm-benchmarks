#!/usr/bin/env python3
"""
Run HumanEval on all models using local-chat-completions.
This properly formats prompts as chat messages so instruction-tuned models work.
"""
import subprocess, json, time, os, urllib.request, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"

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
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M",
     "file": "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
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
    {"name": "DeepSeek-R1-0528-Qwen3-8B Q8_0",
     "file": "DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","32768","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"]},
    {"name": "DeepSeek-R1-0528-Qwen3-8B Q4_K_M",
     "file": "DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma4-coding Q4_K_M",
     "file": "gemma4-coding-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma4-v2 Q4_K_M",
     "file": "gemma4-v2-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "gemma-4-12B-it-QAT Q4_0",
     "file": "gemma-4-12B-it-QAT-Q4_0.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
              "--threads","4","--threads-batch","4","--cache-type-k","q8_0","--cache-type-v","q8_0"]},
    {"name": "DeepSeek-Coder-V2-Lite IQ4_XS",
     "file": "DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","16384","--ubatch-size","256",
              "--threads","4","--threads-batch","4","--cache-type-k","q4_0","--cache-type-v","q4_0"]},
    {"name": "RavenX-OpenFable-Coderagent gemma4 Q4_K_M",
     "file": "RavenX-OpenFable-Coderagent-gemma4-fable5-Q4_K_M.gguf",
     "tokenizer": "Xenova/gpt-4",
     "args": ["--gpu-layers","99","--ctx-size","65536","--ubatch-size","512",
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

def start_proxy():
    """Start the HumanEval proxy on port 18098"""
    proxy_cmd = ["/home/caimlas/bench-venv/bin/python3",
                 "/home/caimlas/llm-benchmarks/scripts/humaneval_proxy.py"]
    import subprocess as sp
    proc = sp.Popen(proxy_cmd, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    time.sleep(2)
    return proc

def stop_proxy(proc):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=5)
        except: proc.kill()

def start_model(model):
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [BINARY, "--model", path, "--device", "CUDA0", "--flash-attn", "on",
           "--batch-size", "2048", "--host", "127.0.0.1", "--port", str(PORT),
           "--parallel", "1", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_humaneval_chat_server.log"), "w")
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

def run_humaneval_chat(name, tokenizer):
    """Run HumanEval using local-chat-completions (proper chat formatting)"""
    safe = name.replace(" ", "_")
    outdir = os.path.join(RESULTS, safe, "humaneval_chat")
    os.makedirs(outdir, exist_ok=True)
    logpath = os.path.join(LOGS, f"{safe}_humaneval_chat.log")

    model_args = (
        f"model={name},"
        f"base_url=http://127.0.0.1:18098/v1/chat/completions,"
        f"num_concurrent=1,"
        f"tokenizer={tokenizer},"
        f"max_length=16384"
    )

    cmd = [
        BENCH_PY, "-m", "lm_eval",
        "--model", "local-chat-completions",
        "--model_args", model_args,
        "--tasks", "humaneval",
        "--gen_kwargs", "temperature=0.0,max_gen_toks=2048",
        "--apply_chat_template",
        "--confirm_run_unsafe_code",
        "--batch_size", "1",
        "--output_path", outdir,
        "--log_samples",
    ]

    env = dict(os.environ)
    env["HF_ALLOW_CODE_EVAL"] = "1"

    print(f"  Running HumanEval (chat mode)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=7200, env=env)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 7200, "error": "timeout"}
    except Exception as e:
        return {"pass_at_1": None, "wall_time_s": time.time() - start, "error": str(e)}

    # Parse pass@1 from results
    pass1 = None
    for root, dirs, files in os.walk(outdir):
        for f in files:
            if f.startswith("results") and f.endswith(".json"):
                with open(os.path.join(root, f)) as rf:
                    try:
                        data = json.load(rf)
                        # Try multiple possible key formats
                        results_data = data.get("results", {}).get("humaneval", {})
                        for key in ["pass@1,none", "pass@1", "acc,none"]:
                            if key in results_data:
                                pass1 = results_data[key]
                                break
                        if pass1 is None:
                            # Try nested format
                            for k, v in results_data.items():
                                if "pass@1" in k and isinstance(v, (int, float)):
                                    pass1 = v
                                    break
                    except:
                        pass

    # Fallback: scan log
    if pass1 is None:
        with open(logpath) as f:
            content = f.read()
        for line in content.split("\n"):
            if "pass@1" in line and "|" in line:
                m = re.search(r'(\d+\.?\d*)', line.split("|")[-1])
                if m:
                    pass1 = float(m.group(1))
                    break

    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1),
            "exit_code": result.returncode}

def main():
    progress = load_progress()

    # Check for existing valid HumanEval results (skip models with real scores, not 0.0)
    done = set()
    for m in progress["models"]:
        he = m.get("humaneval", {})
        if he.get("pass_at_1") is not None and he.get("pass_at_1", 0) > 0:
            done.add(m["name"])
            print(f"SKIP: {m['name']} (HE already {he['pass_at_1']:.1%})")

    for model in MODELS:
        if model["name"] in done:
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
        proxy_proc = start_proxy()
        proc, logf, err = start_model(model)
        if err:
            mr["he_chat_error"] = err
            save_progress(progress)
            print(f"  FAILED: {err}")
            stop_proxy(proxy_proc)
            start_prod()
            continue

        try:
            he = run_humaneval_chat(model["name"], model["tokenizer"])
            mr["humaneval"] = {"pass_at_1": he["pass_at_1"]}
            mr["humaneval_chat"] = he
            print(f"  HumanEval pass@1: {he.get('pass_at_1')}")
            print(f"  Wall time: {he.get('wall_time_s',0)/60:.0f} min")
        except Exception as e:
            mr["he_chat_error"] = str(e)
            print(f"  ERROR: {e}")
        finally:
            kill_model(proc, logf)
            stop_proxy(proxy_proc)
            save_progress(progress)

    start_prod()
    print(f"\n{'='*70}\nHUMANEVAL CHAT COMPLETE\n{'='*70}")
    for m in progress["models"]:
        he = m.get("humaneval", {})
        he_chat = m.get("humaneval_chat", {})
        print(f"  {m['name']:<45} pass@1={he.get('pass_at_1','?')}  ({he_chat.get('wall_time_s',0)/60:.0f} min)")

if __name__ == "__main__":
    main()
