#!/usr/bin/env python3
"""
Benchmark Qwen3.8-Whittle-MoE-27B-A17.8B (v2.1):
  1. V100: BF16 native (partial offload, experts on CPU via --override-tensor)
  2. V100: Q4_K_M (full GPU)
  3. 3060: Q4_K_M (--cpu-moe)
LCB (75) + tau2 (15) + tok/s. Whittle is qwen35moe, no MTP head (mtp_num_hidden_layers=0).

V100 runs chain after the Carnice run completes (same GPU). 3060 run needs user sim on 8081 (bonsai).
"""
import subprocess, json, time, os, urllib.request, shutil, re, glob
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
V100_PORT = 18053
M3060_PORT = 18054

MODELS = [
    {
        "name": "Whittle-MoE-27B-A18B v2.1 BF16",
        "file": "Whittle-MoE-27B-A18B-v2.1-BF16.gguf",  # converted from safetensors
        "lcb_model": "local/whittle-27b-bf16",
        "gpu": 0, "port": V100_PORT,
        # 53GB BF16: attention + shared experts on GPU, routed experts to CPU
        "args": ["--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                 "--reasoning-format", "none",
                 "--chat-template-file", "/home/caimlas/llm-benchmarks/templates/sharp_chat_template.jinja",
                 "--override-tensor", "exps=CPU"],
        "category": "MoE 27B",
        "thinking": True,
    },
    {
        "name": "Whittle-MoE-27B-A18B v2.1 Q4_K_M",
        "file": "Whittle-MoE-27B-A18B-v2.1-Q4_K_M.gguf",
        "lcb_model": "local/whittle-27b-q4km",
        "gpu": 0, "port": V100_PORT,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                 "--reasoning-format", "none",
                 "--chat-template-file", "/home/caimlas/llm-benchmarks/templates/sharp_chat_template.jinja"],
        "category": "MoE 27B",
        "thinking": True,
    },
    {
        "name": "Whittle-MoE-27B-A18B v2.1 Q4_K_M (3060)",
        "file": "Whittle-MoE-27B-A18B-v2.1-Q4_K_M.gguf",
        "lcb_model": "local/whittle-27b-q4km",
        "gpu": 1, "port": M3060_PORT,
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "32768", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
                 "--reasoning-format", "none",
                 "--chat-template-file", "/home/caimlas/llm-benchmarks/templates/sharp_chat_template.jinja"],
        "category": "MoE 27B",
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

def start_model(model):
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [BINARY, "--model", path, "--flash-attn", "on",
           "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(model["port"]),
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(model["gpu"])
    safe = model["name"].replace(" ", "_").replace("(", "").replace(")", "")
    logf = open(os.path.join(LOGS, f"{safe}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(300):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{model['port']}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{safe}_server.log")) as f:
                err = f.read()[-500:]
            return None, None, f"Server died: {err}"
    proc.kill()
    return None, None, "Timeout"

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=15)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(5)

def timed_completion(port, prompt, max_tokens=256):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.3, "stream": False,
    })
    start = time.time()
    conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    elapsed = time.time() - start
    toks = data.get("usage", {}).get("completion_tokens", max_tokens)
    return toks / elapsed if elapsed > 0 else 0, toks

def probe_tps(model):
    try:
        timed_completion(model["port"], "Write a short hello world function.", max_tokens=8)
        tps, toks = timed_completion(model["port"],
            "Write a detailed essay about the history of computing, from Babbage to modern GPUs.", 256)
        print(f"  tok/s: {tps:.1f} ({toks} tokens)")
        return round(tps, 1)
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None

def run_livecodebench(model):
    name = model["name"]
    lcb_model = model["lcb_model"]
    port = model["port"]
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    logpath = os.path.join(LOGS, f"{safe}_lcb.log")

    for d in glob.glob(os.path.join(LCB_DIR, "output", "*Whittle*")):
        shutil.rmtree(d)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", lcb_model,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
        "--num_problems", "75", "--openai_timeout", "600", "--evaluate",
    ]
    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    print("  Running LiveCodeBench (75 problems)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=36000, env=env, cwd=LCB_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 36000, "error": "timeout"}

    pass1 = None
    for d in glob.glob(os.path.join(LCB_DIR, "output", "*Whittle*")):
        for root, dirs, files in os.walk(d):
            for f in files:
                if f.endswith("_eval.json") and not f.endswith("_all.json"):
                    try:
                        data = json.load(open(os.path.join(root, f)))
                        if isinstance(data, list) and data:
                            pass1 = data[0].get("pass@1")
                    except: pass
    if pass1 is None:
        try:
            tail = open(logpath).read().strip().split("\n")[-1]
            v = re.match(r'^([\d.]+)$', tail.strip())
            if v: pass1 = float(v.group(1))
        except: pass
    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed,1), "exit_code": result.returncode}

def run_tau2(model):
    name = model["name"]
    port = model["port"]
    safe = name.replace(" ", "_").replace("(", "").replace(")", "")
    gpu_suffix = "_3060" if model["gpu"] == 1 else ""
    logpath = os.path.join(LOGS, f"{safe}_tau2.log")
    save_dir = f"tau2_whittle_{safe}"

    user_port = 8081  # bonsai on V100
    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", f"openai/{safe}",
        "--agent-llm-args", json.dumps({
            "api_key": "none", "api_base": f"http://127.0.0.1:{port}/v1", "temperature": 0.0}),
        "--user-llm", "openai/Qwythos-27B-v1",
        "--user-llm-args", json.dumps({
            "api_key": "none", "api_base": f"http://localhost:{user_port}/v1"}),
        "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
        "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
        "--seed", "42", "--auto-resume", "--save-to", save_dir,
    ]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print("  Running tau2-bench (airline, 15 tasks)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=86400, env=env, cwd=TAU2_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"reward": None, "wall_time_s": 86400, "error": "timeout"}

    reward = task_pass = None
    content = open(logpath).read()
    m = re.search(r'Average Reward\s+([\d.]+)', content)
    if m: reward = float(m.group(1))
    m2 = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    if m2: task_pass = float(m2.group(1))
    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {"reward": reward, "task_pass_rate": task_pass,
            "wall_time_s": round(elapsed,1), "exit_code": result.returncode}

def main():
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None  # optional: bf16 | q4km-v100 | q4km-3060
    os.makedirs(LOGS, exist_ok=True)
    progress = load_progress()

    done_names = set()
    for m in progress["models"]:
        if m.get("livecodebench", {}).get("pass_at_1") is not None and m.get("tau2", {}).get("reward") is not None:
            done_names.add(m["name"])

    for model in MODELS:
        if only:
            ml = model["name"].lower()
            match = only.lower() in ml or only.lower() in model["file"].lower()
            # "q4_km" alone should mean the V100 entry only; use "3060" for the 3060 entry
            if only == "Q4_K_M" and model["gpu"] != 0:
                match = False
            if not match:
                continue
        if not os.path.exists(os.path.join(LLMS_DIR, model["file"])):
            print(f"SKIP: {model['file']} not on disk")
            continue
        if model["name"] in done_names:
            print(f"SKIP: {model['name']} (already done)")
            continue

        print(f"\n{'='*70}\n  {model['name']}\n{'='*70}")
        mr = None
        for m in progress["models"]:
            if m["name"] == model["name"]:
                mr = m
                break
        if mr is None:
            mr = {"name": model["name"], "file": model["file"],
                  "category": model["category"], "gpu": f"GPU{model['gpu']}", "template": "sharp"}
            progress["models"].append(mr)

        try:
            r = urllib.request.urlopen("http://localhost:8081/health", timeout=3)
            if json.loads(r.read()).get("status") != "ok":
                print("  WARNING: user sim (bonsai :8081) not healthy")
        except Exception:
            print("  WARNING: user sim (bonsai :8081) unreachable — tau2 needs it")

        proc, logf, err = start_model(model)
        if err:
            print(f"  FATAL: {err[:300]}")
            mr["lcb_error"] = err
            mr["failures"] = [{"benchmark": "all", "error": str(err)[:500],
                               "timestamp": datetime.now(timezone.utc).isoformat()}]
            save_progress(progress)
            continue

        try:
            key = "decode_tps" if model["gpu"] == 0 else "decode_tps_3060"
            mr[key] = probe_tps(model)
            save_progress(progress)
            lcb_key = "livecodebench" if model["gpu"] == 0 else "livecodebench_3060"
            mr[lcb_key] = run_livecodebench(model)
            save_progress(progress)
            tau2_key = "tau2" if model["gpu"] == 0 else "tau2_3060"
            mr[tau2_key] = run_tau2(model)
            save_progress(progress)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            mr.setdefault("failures", []).append({"benchmark": "unknown", "error": str(e)[:500],
                                                  "timestamp": datetime.now(timezone.utc).isoformat()})
        finally:
            kill_model(proc, logf)
            save_progress(progress)

    print("\nWHITTLE BENCHMARKS COMPLETE")

if __name__ == "__main__":
    main()
