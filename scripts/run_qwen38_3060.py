#!/usr/bin/env python3
"""
Benchmark Qwen3.8-27B-UD-IQ2_XXS on the 3060 (12GB).
Runs LCB + tau2 + tok/s. Agent model on 3060 (CUDA1), user sim on V100:8081
(caimlas-qwythos) OR LFM on :8082 — but note: this script assumes the V100 is
busy with the main benchmark run, so user sim must be LFM... which also runs on
3060. Since agent + user sim both need the 3060 is not possible, the user sim
for these runs is V100 caimlas-qwythos on :8081 (started by the caller).

Usage:
    /home/caimlas/bench-venv/bin/python scripts/run_qwen38_3060.py
"""
import subprocess, json, time, os, urllib.request, shutil, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18098      # Benchmark model on 3060
USER_PORT = 8081  # User simulator: Qwythos production service on V100
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

MODELS = [
    {
        "name": "Qwen3.8-27B UD-IQ2_XXS",
        "file": "Qwen3.8-27B-UD-IQ2_XXS.gguf",
        "lcb_model": "local/qwen38-27b-iq2xxs",
        "binary": BINARY,
        # 12GB VRAM: 9GB model + KV cache. 131K q8_0 KV (4.35GB) OOMs.
        # 65536 ctx with q4_0 KV ≈ 1.1GB — fits with headroom.
        "args": ["--gpu-layers", "99", "--ctx-size", "65536", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"],
        "category": "27B Dense",
        "thinking": True,
        "gpu": "3060",
        "has_mtp": False,
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

def start_model(model, extra_args=None):
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [BINARY, "--model", path, "--flash-attn", "on",
           "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    if extra_args:
        cmd.extend(extra_args)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"  # 3060
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_3060_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{safe}_3060_server.log")) as f:
                err = f.read()[-500:]
            return None, None, f"Server died: {err}"
    proc.kill()
    return None, None, "Timeout"

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def timed_completion(prompt, max_tokens=256):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.3, "stream": False,
    })
    start = time.time()
    conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    elapsed = time.time() - start
    usage = data.get("usage", {})
    toks = usage.get("completion_tokens", max_tokens)
    return toks / elapsed if elapsed > 0 else 0, toks

def probe_tps(model):
    try:
        timed_completion("Write a short hello world function.", max_tokens=8)
        tps, toks = timed_completion(
            "Write a detailed essay about the history of computing, from Babbage to modern GPUs.",
            max_tokens=256)
        print(f"  tok/s: {tps:.1f} ({toks} tokens)")
        return round(tps, 1)
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None

def run_livecodebench(model_name, lcb_model):
    safe = model_name.replace(" ", "_")
    logpath = os.path.join(LOGS, f"{safe}_3060_lcb.log")

    for dirname in [lcb_model.replace("/", "_"), model_name]:
        d = os.path.join(LCB_DIR, "output", dirname)
        if os.path.exists(d):
            shutil.rmtree(d)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", lcb_model,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1",
        "--temperature", "0.0",
        "--max_tokens", "4096",
        "--num_problems", "75",
        "--openai_timeout", "600",
        "--evaluate",
    ]

    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    print(f"  Running LiveCodeBench (75 problems, thinking off)...")
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

    pass1 = None
    for dirname in [lcb_model.replace("/", "_"), model_name]:
        output_dir = os.path.join(LCB_DIR, "output", dirname)
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith("_eval.json"):
                    with open(os.path.join(root, f)) as rf:
                        try:
                            data = json.load(rf)
                            if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                                pass1 = data[0].get("pass@1")
                        except:
                            pass
        if pass1 is not None:
            break

    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def run_tau2(model_name, model_file, max_steps=30):
    safe = model_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    logpath = os.path.join(LOGS, f"{safe}_3060_tau2.log")
    save_dir = f"tau2_3060_{safe}"

    agent_model = f"openai/{safe}"
    user_model = "openai/Qwythos-27B-v1"

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
        "--num-tasks", "15",
        "--num-trials", "1",
        "--max-concurrency", "2",
        "--max-steps", str(max_steps),
        "--max-errors", "5",
        "--timeout", "300",
        "--seed", "42",
        "--save-to", save_dir,
    ]

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print(f"  Running tau2-bench (airline, 15 tasks)...")
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

    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {
        "reward": reward,
        "task_pass_rate": task_pass,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
    }

def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    progress = load_progress()

    done_names = set()
    for m in progress["models"]:
        if "livecodebench" in m and m.get("livecodebench", {}).get("pass_at_1") is not None:
            done_names.add(m["name"])

    for model in MODELS:
        path = os.path.join(LLMS_DIR, model["file"])
        if not os.path.exists(path):
            print(f"SKIP: {model['file']} not on disk yet")
            continue
        if model["name"] in done_names:
            print(f"SKIP: {model['name']} (already done)")
            continue

        print(f"\n{'='*70}")
        print(f"  {model['name']}")
        print(f"{'='*70}")

        mr = None
        for m in progress["models"]:
            if m["name"] == model["name"]:
                mr = m
                break
        if mr is None:
            mr = {"name": model["name"], "file": model["file"],
                  "category": model.get("category", ""), "gpu": model.get("gpu", "3060")}
            progress["models"].append(mr)

        # Check user simulator (V100 production Qwythos)
        try:
            r = urllib.request.urlopen(f"http://localhost:{USER_PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") != "ok":
                print(f"  WARNING: User simulator on :{USER_PORT} not healthy!")
        except:
            print(f"  WARNING: User simulator on :{USER_PORT} not reachable! (tau2 will fail)")

        proc, logf, err = start_model(model)
        if err:
            print(f"  FATAL: Server failed to start: {err[:200]}")
            if "failures" not in mr:
                mr["failures"] = []
            mr["failures"].append({
                "benchmark": "all",
                "error": str(err)[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            mr["lcb_error"] = err
            save_progress(progress)
            continue

        try:
            tps = probe_tps(model)
            mr["decode_tps_3060"] = tps
            save_progress(progress)

            lcb = run_livecodebench(model["name"], model["lcb_model"])
            mr["livecodebench_3060"] = lcb
            save_progress(progress)

            tau2 = run_tau2(model["name"], model["file"], max_steps=30)
            mr["tau2_3060"] = tau2
            save_progress(progress)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            if "failures" not in mr:
                mr["failures"] = []
            mr["failures"].append({
                "benchmark": "unknown",
                "error": str(e)[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            kill_model(proc, logf)
            save_progress(progress)

    print(f"\n3060 BENCHMARKS COMPLETE")

if __name__ == "__main__":
    main()
