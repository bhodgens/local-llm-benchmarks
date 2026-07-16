#!/usr/bin/env python3
"""
Run LCB + tau2 for gemma-4 models on 3060.
1. gemma-4-12B-it-QAT Q4_0 at 128K context (upstream binary, no MTP)
2. gemma4-v2-agentic Q3_K_M with MTP draft
3. gemma4-v2-agentic Q4_K_M with MTP draft
Falls back to no-MTP if the draft model fails to load.
"""
import subprocess, json, time, os, urllib.request, shutil, re, http.client
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
USER_PORT = 8081  # Qwopus on V100 for tau2 user sim
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
MTP_DRAFT = os.path.join(LLMS_DIR, "gemma-4-12B-it-MTP-Q8_0.gguf")

MODELS = [
    {
        "name": "gemma-4-12B-it-QAT Q4_0 (3060 128K)",
        "file": "gemma-4-12B-it-QAT-Q4_0.gguf",
        "lcb_model": "local/gemma4-12b-qat-3060",
        "ctx_lcb": 131072,
        "ctx_tau2": 65536,  # parallel=2 -> 32K/slot
        "kv": "q4_0",
        "use_mtp": False,
        "category": "12-14B",
    },
    {
        "name": "gemma4-v2-agentic Q3_K_M (3060+MTP)",
        "file": "gemma4-v2-agentic-Q3_K_M.gguf",
        "lcb_model": "local/gemma4-v2-agentic-q3",
        "ctx_lcb": 131072,
        "ctx_tau2": 65536,
        "kv": "q4_0",
        "use_mtp": True,
        "category": "12-14B",
    },
    {
        "name": "gemma4-v2-agentic Q4_K_M (3060+MTP)",
        "file": "gemma4-v2-agentic-Q4_K_M.gguf",
        "lcb_model": "local/gemma4-v2-agentic-q4",
        "ctx_lcb": 65536,
        "ctx_tau2": 65536,
        "kv": "q4_0",
        "use_mtp": True,
        "category": "12-14B",
    },
]

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def start_server(model, ctx_size, parallel=1):
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [
        BINARY, "--model", path, "--flash-attn", "on",
        "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
        "--parallel", str(parallel), "--temp", "0.0", "-n", "4096",
        "--gpu-layers", "99", "--ctx-size", str(ctx_size), "--ubatch-size", "512",
        "--threads", "6", "--threads-batch", "6",
        "--cache-type-k", model["kv"], "--cache-type-v", model["kv"],
        "--jinja",
    ]
    if model.get("use_mtp") and os.path.exists(MTP_DRAFT):
        cmd.extend(["--spec-draft-model", MTP_DRAFT,
                     "--spec-type", "draft-mtp", "--spec-draft-n-max", "4"])
        mtp_note = " (MTP draft enabled)"
    else:
        mtp_note = ""

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    safe = model["name"].replace(" ", "_").replace("(", "").replace(")", "")
    logf = open(os.path.join(LOGS, f"{safe}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(120):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                print(f"  Server up{mtp_note} at ctx={ctx_size}, parallel={parallel}")
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{safe}_server.log")) as f:
                err = f.read()[-500:]
            if model.get("use_mtp") and "mtp" in err.lower() or "draft" in err.lower() or "assistant" in err.lower():
                print(f"  MTP draft failed, retrying without...")
                model["use_mtp"] = False
                return start_server(model, ctx_size, parallel)
            return None, None, f"Server died: {err}"
    proc.kill()
    return None, None, "Timeout"

def kill_server(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def probe_tps(model_name):
    print("  Probing tok/s...")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=60)
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8, "temperature": 0.0, "stream": False,
        })
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        conn.getresponse().read()
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Write a detailed essay about the history of computing, from Babbage to modern GPUs. Include key milestones, important figures, and technological breakthroughs."}],
            "max_tokens": 256, "temperature": 0.0, "stream": False,
        })
        start = time.time()
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read())
        elapsed = time.time() - start
        usage = data.get("usage", {})
        ct = usage.get("completion_tokens", 256)
        tps = ct / elapsed if elapsed > 0 else 0
        print(f"  tok/s: {tps:.1f} ({ct} tokens in {elapsed:.1f}s)")
        return round(tps, 1)
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None

def run_lcb(model_name, lcb_model):
    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    logpath = os.path.join(LOGS, f"{safe}_lcb.log")
    output_dir = os.path.join(LCB_DIR, "output", lcb_model.replace("/", "_"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", lcb_model,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
        "--num_problems", "75", "--openai_timeout", "300", "--evaluate",
    ]
    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    print("  Running LCB (75 problems, thinking off)...")
    start = time.time()
    with open(logpath, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=36000, env=env, cwd=LCB_DIR)
    elapsed = time.time() - start

    pass1 = None
    with open(logpath) as f:
        lines = [l.strip() for l in f if l.strip()]
    for line in reversed(lines):
        try:
            val = float(line)
            if 0 <= val <= 1:
                pass1 = val
                break
        except ValueError:
            m = re.search(r'(\d+\.?\d*)\s*%', line)
            if m:
                pass1 = float(m.group(1)) / 100
                break

    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def run_tau2(model_name):
    safe = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    logpath = os.path.join(LOGS, f"{safe}_tau2.log")
    save_dir = f"tau2_{safe}"
    agent_model = f"openai/{safe}"
    user_model = "openai/Qwopus3.6-27B-Coder-Compat-MTP"

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
        "--num-tasks", "15", "--num-trials", "1",
        "--max-concurrency", "2", "--max-steps", "30",
        "--max-errors", "5", "--timeout", "300",
        "--seed", "42", "--save-to", save_dir,
    ]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print("  Running tau2 (airline, 15 tasks)...")
    start = time.time()
    with open(logpath, "w") as lf:
        result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                timeout=86400, env=env, cwd=TAU2_DIR)
    elapsed = time.time() - start

    reward = None
    task_pass = None
    with open(logpath) as f:
        content = f.read()
    m = re.search(r'Average Reward\s+([\d.]+)', content)
    if m: reward = float(m.group(1))
    m = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    if m: task_pass = float(m.group(1))

    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {
        "reward": reward, "task_pass_rate": task_pass,
        "wall_time_s": round(elapsed, 1), "exit_code": result.returncode,
    }

def main():
    os.makedirs(LOGS, exist_ok=True)

    # Stop 3060 services
    subprocess.run(["sudo", "systemctl", "stop", "caimlas-ravenx", "caimlas-lfm"],
                   capture_output=True, timeout=15)
    time.sleep(3)

    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"  {model['name']}")
        print(f"{'='*60}")

        mr = None
        for m in progress["models"]:
            if m["name"] == model["name"]:
                mr = m
                break
        if mr is None:
            mr = {"name": model["name"], "file": model["file"], "category": model["category"], "gpu": "3060"}
            progress["models"].append(mr)

        # LCB run
        proc, logf, err = start_server(model, model["ctx_lcb"], parallel=1)
        if err:
            print(f"  FATAL: {err}")
            mr["lcb_error"] = err
            save_progress(progress)
            continue
        try:
            tps = probe_tps(model["name"])
            mr["decode_tps"] = tps
            mr["mtp_enabled"] = model.get("use_mtp", False)
            save_progress(progress)
            lcb = run_lcb(model["name"], model["lcb_model"])
            mr["livecodebench"] = lcb
            save_progress(progress)
        finally:
            kill_server(proc, logf)

        # tau2 run
        proc, logf, err = start_server(model, model["ctx_tau2"], parallel=2)
        if err:
            print(f"  FATAL (tau2): {err}")
            mr["tau2_error"] = err
            save_progress(progress)
            continue
        try:
            tau2 = run_tau2(model["name"])
            mr["tau2"] = tau2
            save_progress(progress)
        finally:
            kill_server(proc, logf)

    # Restart 3060 services
    subprocess.run(["sudo", "systemctl", "start", "caimlas-ravenx", "caimlas-lfm"],
                   capture_output=True, timeout=15)

    print(f"\n{'='*60}")
    print("GEMMA 3060 BENCHMARKS COMPLETE")
    print(f"{'='*60}")
    for m in sorted(progress["models"], key=lambda x: -(x.get("livecodebench", {}).get("pass_at_1") or 0)):
        lcb = m.get("livecodebench", {})
        tau2 = m.get("tau2", {})
        if "3060" in m["name"] or "agentic" in m["name"].lower():
            print(f"  {m['name']:<50} LCB={lcb.get('pass_at_1','?')}  tau2={tau2.get('reward','?')}  tps={m.get('decode_tps','?')}")

if __name__ == "__main__":
    main()
