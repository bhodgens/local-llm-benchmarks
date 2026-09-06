#!/usr/bin/env python3
"""
Benchmark Fuse-2-MoE (Fuse4 arch: Qwen3.5-4B host + pruned Qwen3.8-27B experts).
  Leg 1: BF16 (17.75GB) on V100 (CUDA0), user sim LFM2.5-8B on 3060 :8082
  Leg 2: Q4_K_M (5.7GB)  on 3060 (CUDA1), user sim LFM2.5-8B on V100 :18097
Same user-sim model both legs for cross-leg comparability.
Requires patched llama.cpp: /home/caimlas/git/llama.cpp-fuse4/build/bin/
Usage: /home/caimlas/bench-venv/bin/python scripts/run_fuse2.py
"""
import subprocess, json, time, os, urllib.request, shutil, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp-fuse4/build/bin/llama-server"
SIM_BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
FUSE_DIR = "/home/caimlas/models/fuse2"
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress_fuse2.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
SIM_GGUF = "/home/files/llms/LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"

LEGS = [
    {
        "name": "Fuse-2-MoE-BF16",
        "file": os.path.join(FUSE_DIR, "Fuse-2-MoE-BF16.gguf"),
        "lcb_model": "local/fuse2-bf16",
        "gpu": "0",            # V100
        "port": 18099,
        "sim_gpu": "1",        # 3060
        "sim_port": 8082,
        "ctx": "32768",
        "args": ["--gpu-layers", "99", "--ctx-size", "32768", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    },
    {
        "name": "Fuse-2-MoE-Q4_K_M",
        "file": os.path.join(FUSE_DIR, "Fuse-2-MoE-Q4_K_M.gguf"),
        "lcb_model": "local/fuse2-q4",
        "gpu": "1",            # 3060
        "port": 18099,
        "sim_gpu": "0",        # V100
        "sim_port": 18097,
        "ctx": "16384",
        "args": ["--gpu-layers", "99", "--ctx-size", "16384", "--ubatch-size", "256",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
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

def start_sim(leg):
    cmd = [SIM_BINARY, "--model", SIM_GGUF, "--flash-attn", "on",
           "--host", "0.0.0.0", "--port", str(leg["sim_port"]),
           "--gpu-layers", "99", "--ctx-size", "32768", "--ubatch-size", "512",
           "--threads", "6", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = leg["sim_gpu"]
    safe = leg["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_usersim.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(150):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{leg['sim_port']}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                print(f"  user sim up on :{leg['sim_port']} (gpu {leg['sim_gpu']})")
                return proc, logf
        except Exception:
            pass
        if proc.poll() is not None:
            logf.close()
            return None, None
    proc.kill(); logf.close()
    return None, None

def kill_proc(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except Exception: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def start_model(leg):
    cmd = [BINARY, "--model", leg["file"], "--flash-attn", "on",
           "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(leg["port"]),
           "--parallel", "2", "--temp", "0.0", "-n", "4096", "--jinja"]
    cmd.extend(leg["args"])
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = leg["gpu"]
    safe = leg["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(240):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{leg['port']}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf, None
        except Exception:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{safe}_server.log")) as f:
                err = f.read()[-800:]
            return None, None, f"Server died: {err}"
    proc.kill()
    return None, None, "Timeout waiting for server health"

def probe_tps(leg):
    import http.client
    print("  Probing decode tok/s...")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", leg["port"], timeout=90)
        warm = json.dumps({"messages": [{"role": "user", "content": "Write a short hello world function."}],
                           "max_tokens": 8, "temperature": 0.0, "stream": False})
        conn.request("POST", "/v1/chat/completions", warm, {"Content-Type": "application/json"})
        conn.getresponse().read()
        payload = json.dumps({"messages": [{"role": "user", "content": "Write a detailed essay about the history of computing, from Babbage to modern GPUs. Include key milestones, important figures, and technological breakthroughs."}],
                              "max_tokens": 256, "temperature": 0.0, "stream": False})
        start = time.time()
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read())
        elapsed = time.time() - start
        usage = data.get("usage", {})
        ct = usage.get("completion_tokens", 256)
        content = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        tps = ct / elapsed if elapsed > 0 else 0
        print(f"  tok/s: {tps:.1f} ({ct} tokens in {elapsed:.1f}s); reply_chars={len(content)}")
        return round(tps, 1), content
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None, ""

def run_livecodebench(leg, num_problems=75, timeout_s=36000):
    model_name, lcb_model = leg["name"], leg["lcb_model"]
    safe = model_name.replace(" ", "_")
    logpath = os.path.join(LOGS, f"{safe}_lcb.log")
    for dirname in [lcb_model.replace("/", "_"), model_name]:
        output_dir = os.path.join(LCB_DIR, "output", dirname)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
    cmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
           "--model", lcb_model, "--scenario", "codegeneration",
           "--release_version", "release_latest", "--n", "1",
           "--temperature", "0.0", "--max_tokens", "4096",
           "--num_problems", str(num_problems), "--openai_timeout", "300",
           "--evaluate"]
    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{leg['port']}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"
    print(f"  Running LiveCodeBench ({num_problems} problems, thinking off)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=timeout_s, env=env, cwd=LCB_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": timeout_s, "error": "timeout"}
    except Exception as e:
        return {"pass_at_1": None, "wall_time_s": time.time() - start, "error": str(e)}
    pass1 = None
    cands = [lcb_model.replace("/", "_"), model_name]
    for dirname in cands:
        output_dir = os.path.join(LCB_DIR, "output", dirname)
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith("_eval.json") and not f.endswith("_eval_all.json"):
                    try:
                        data = json.load(open(os.path.join(root, f)))
                        if isinstance(data, list) and data and isinstance(data[0], dict):
                            pass1 = data[0].get("pass@1")
                    except Exception:
                        pass
        if pass1 is not None:
            break
    if pass1 is None:
        # robust: glob any output dir containing the base name
        for d in os.listdir(os.path.join(LCB_DIR, "output")):
            if "Fuse-2" in d or "fuse2" in d:
                for root, dirs, files in os.walk(os.path.join(LCB_DIR, "output", d)):
                    for f in files:
                        if f.endswith("_eval.json") and not f.endswith("_eval_all.json"):
                            try:
                                data = json.load(open(os.path.join(root, f)))
                                if isinstance(data, list) and data and isinstance(data[0], dict):
                                    pass1 = data[0].get("pass@1")
                            except Exception:
                                pass
    if pass1 is None:
        with open(logpath) as f:
            content = f.read()
        m = re.findall(r"^([01]?\.\d+)", content, re.M)
        if m:
            pass1 = float(m[-1])
    n_peg = 0
    try:
        with open(os.path.join(LOGS, f"{safe}_server.log")) as f:
            n_peg = f.read().count("peg-native")
    except Exception:
        pass
    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min, peg_errors={n_peg})")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1),
            "exit_code": result.returncode, "peg_errors": n_peg}

def run_tau2(leg, max_steps=30):
    model_name = leg["name"]
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", model_name)
    logpath = os.path.join(LOGS, f"{safe}_tau2.log")
    save_dir = f"tau2_fuse2_{safe}"
    # clear stale save dir (EOFError pitfall)
    stale = os.path.join(TAU2_DIR, "data", "simulations", save_dir)
    if os.path.exists(stale):
        shutil.rmtree(stale)
    agent_model = f"openai/{safe}"
    user_model = "openai/LFM2.5-8B-A1B-Clean-RealWorld-v2"
    cmd = ["uv", "run", "tau2", "run",
           "--domain", "airline",
           "--agent-llm", agent_model,
           "--agent-llm-args", json.dumps({
               "api_key": "none",
               "api_base": f"http://127.0.0.1:{leg['port']}/v1",
               "temperature": 0.0}),
           "--user-llm", user_model,
           "--user-llm-args", json.dumps({
               "api_key": "none",
               "api_base": f"http://localhost:{leg['sim_port']}/v1"}),
           "--num-tasks", "15", "--num-trials", "1",
           "--max-concurrency", "2", "--max-steps", str(max_steps),
           "--max-errors", "5", "--timeout", "300",
           "--seed", "42", "--save-to", save_dir]
    env = dict(os.environ); env["OPENAI_API_KEY"] = "none"
    print("  Running tau2-bench (airline, 15 tasks)...")
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
    reward, task_pass = None, None
    content = open(logpath).read()
    m = re.search(r"Average Reward\s+([\d.]+)", content)
    if m: reward = float(m.group(1))
    m = re.search(r"Task Pass Rate\s+([\d.]+)", content)
    if m: task_pass = float(m.group(1))
    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {"reward": reward, "task_pass_rate": task_pass,
            "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    progress = load_progress()
    done = {}
    for m in progress["models"]:
        if all([m.get("probe_ok"), m.get("livecodebench", {}).get("pass_at_1") is not None,
                m.get("tau2", {}).get("reward") is not None]):
            done[m["name"]] = True
    for leg in LEGS:
        if done.get(leg["name"]):
            print(f"SKIP: {leg['name']} (complete)")
            continue
        print(f"\n{'='*70}\n  {leg['name']} on GPU {leg['gpu']}\n{'='*70}")
        mr = next((m for m in progress["models"] if m["name"] == leg["name"]), None)
        if mr is None:
            mr = {"name": leg["name"], "file": leg["file"], "gpu": f"CUDA{leg['gpu']}"}
            progress["models"].append(mr)
        try:
            sim_proc, sim_log = start_sim(leg)
            if sim_proc is None:
                mr["sim_error"] = "user sim failed to start"
                save_progress(progress)
                continue
            proc, logf, err = start_model(leg)
            if err:
                mr["server_error"] = str(err)[:500]
                save_progress(progress)
                kill_proc(sim_proc, sim_log)
                continue
            mr["thinking"] = False
            try:
                tps, content = probe_tps(leg)
                mr["decode_tps"] = tps
                mr["probe_reply_chars"] = len(content)
                mr["probe_ok"] = (tps is not None and len(content) > 100)
                save_progress(progress)
                if not mr["probe_ok"]:
                    print("  WARNING: probe failed/coherent-content empty; skipping LCB/tau2 for this leg")
                    continue
                mr["livecodebench"] = run_livecodebench(leg)
                save_progress(progress)
                mr["tau2"] = run_tau2(leg)
                save_progress(progress)
            finally:
                kill_proc(proc, logf)
        finally:
            if 'sim_proc' in dir() and sim_proc:
                kill_proc(sim_proc, sim_log)
    print(f"\n{'='*70}\nFUSE-2 BENCHMARKS COMPLETE\n{'='*70}")
    progress = load_progress()
    for m in progress["models"]:
        lcb = m.get("livecodebench", {})
        tau2 = m.get("tau2", {})
        print(f"  {m['name']:<25} LCB={lcb.get('pass_at_1','?')}  tau2={tau2.get('reward','?')}  tok/s={m.get('decode_tps','?')}")

if __name__ == "__main__":
    main()
