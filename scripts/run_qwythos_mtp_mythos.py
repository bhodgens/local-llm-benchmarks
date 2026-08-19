#!/usr/bin/env python3
"""
Benchmark Qwythos-27B-MTP and Qwythos-9B-Claude-Mythos-5-1M on V100.
For the 27B-MTP model: MTP acceptance tests (n=3, n=5) + no-spec quality benchmarks.
For the Mythos 9B: standard LCB + tau2 + tok/s.

Agent models run on V100 (CUDA0). User simulator for tau2: LFM on :8082 (3060).

Usage:
    /home/caimlas/bench-venv/bin/python scripts/run_qwythos_mtp_mythos.py

    # Start user sim on 3060 first:
    # CUDA_VISIBLE_DEVICES=1 llama-server --model LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf \
    #   --flash-attn on --host 0.0.0.0 --port 8082 --gpu-layers 99 --ctx-size 131072 \
    #   --ubatch-size 512 --threads 4 --threads-batch 4 --cache-type-k q8_0 --cache-type-v q8_0 \
    #   --parallel 2 --temp 0.0 -n 4096 &
"""
import subprocess, json, time, os, urllib.request, shutil, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099      # Benchmark model on V100
USER_PORT = 8082  # User simulator: LFM on 3060
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
BENCH_RESULTS = "/home/caimlas/llm-benchmarks/bench_results.json"

# V100 service to stop during benchmarks
V100_SERVICE = "caimlas-qwythos"

MODELS = [
    {
        "name": "Qwen3.8-27B Heretic Q4_K_M",
        "file": "Qwen3.8-27B-Heretic-Q4_K_M.gguf",
        "lcb_model": "local/qwen38-27b-heretic-mtp",
        "binary": BINARY,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "27B Dense",
        "thinking": True,
        "gpu": "V100",
        "has_mtp": False,  # repo ships NO MTP head (block_count=64, no nextn tensors)
    },
    {
        "name": "Qwen3.8-27B AEON Ultimate Uncensored Q4_K_M MTP",
        "file": "Qwen3.8-27B-AEON-Ultimate-Uncensored-Q4_K_M.gguf",
        "lcb_model": "local/qwen38-27b-aeon-mtp",
        "binary": BINARY,
        # Hybrid SSM arch: recurrent layers carry state, no KV quant for those.
        # llama.cpp handles mixed cache automatically; keep flags as-is for attention layers.
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "27B Dense",
        "thinking": True,
        "gpu": "V100",
        "has_mtp": True,
    },
    {
        "name": "Qwen3.8-27B Uncensored Q4_K_M MTP",
        "file": "Qwen3.8-27B-Uncensored-Q4_K_M.gguf",
        "lcb_model": "local/qwen38-27b-uncens-mtp",
        "binary": BINARY,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "27B Dense",
        "thinking": True,
        "gpu": "V100",
        "has_mtp": True,
    },
]


# Prompts for the MTP acceptance test (same as bench_orchestrator.py)
PROMPT_EVAL_TEXT = open(__file__).read()[:2000]  # fallback, not used
DECODE_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list. Include docstring, type hints, and handle edge cases.",
    "Explain how merge sort works step by step. Include pseudocode and analyze the time complexity.",
]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def stop_v100():
    subprocess.run(["sudo", "systemctl", "stop", V100_SERVICE], capture_output=True, timeout=15)
    time.sleep(3)

def start_v100():
    subprocess.run(["sudo", "systemctl", "start", V100_SERVICE], capture_output=True, timeout=15)
    for _ in range(60):
        time.sleep(3)
        try:
            r = urllib.request.urlopen("http://localhost:8081/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return
        except:
            pass

def wait_health(port, timeout_s=360):
    for _ in range(timeout_s // 2):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return True
        except:
            pass
    return False

def start_model(model, extra_args=None):
    binary = model.get("binary", BINARY)
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [binary, "--model", path, "--flash-attn", "on",
           "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    if extra_args:
        cmd.extend(extra_args)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # V100
    safe = model["name"].replace(" ", "_")
    suffix = "_" + "_".join(str(a) for a in extra_args).replace("/", "_").replace("-", "_") if extra_args else ""
    logname = f"{safe}_v100_server{suffix}.log"
    logf = open(os.path.join(LOGS, logname), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf, logname, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, logname)) as f:
                err = f.read()[-500:]
            return None, None, logname, f"Server died: {err}"
    proc.kill()
    return None, None, logname, "Timeout"

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf:
        logf.close()
    time.sleep(3)

def timed_completion(prompt, max_tokens=256):
    """Send a completion, return (tps, tokens)."""
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

def mtp_acceptance_test(model, mr, progress):
    """Run MTP acceptance test: no-spec baseline, then MTP n=3 and n=5.
    Parse acceptance rate from server log. Results go to bench_results.json."""
    results = {}
    if os.path.exists(BENCH_RESULTS):
        with open(BENCH_RESULTS) as f:
            results = json.load(f)

    label = model["file"]
    configs = [("no-spec", []), ("MTP-n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]),
               ("MTP-n5", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"])]

    mtp_stats = {}
    for config_name, extra in configs:
        key = f"{label} [{config_name}]"
        if key in results and "decode_tps_avg" in results.get(key, {}):
            print(f"  SKIP: {key} already benchmarked")
            mtp_stats[config_name] = results[key]
            continue

        print(f"  MTP test config: {config_name}")
        proc, logf, logname, err = start_model(model, extra_args=extra)
        if err:
            print(f"  FAILED to start ({config_name}): {err[:150]}")
            results[key] = {"error": str(err)[:300]}
            save_bench_results(results)
            continue

        # warmup
        try:
            timed_completion("Write a short hello world function.", max_tokens=16)
            tps_runs = []
            for prompt in DECODE_PROMPTS:
                tps, toks = timed_completion(prompt, max_tokens=256)
                tps_runs.append(round(tps, 1))
                print(f"    decode: {tps:.1f} tok/s ({toks} tokens)")
            avg_tps = round(sum(tps_runs) / len(tps_runs), 1) if tps_runs else None
        except Exception as e:
            print(f"    probe failed: {e}")
            avg_tps, tps_runs = None, []

        # parse acceptance from server log
        acc_lines = []
        try:
            with open(os.path.join(LOGS, logname)) as f:
                for line in f:
                    ll = line.lower()
                    if "draft acceptance" in ll or ("accepted" in ll and "generated" in ll):
                        acc_lines.append(line.strip())
        except:
            pass

        entry = {"decode_tps_runs": tps_runs, "decode_tps_avg": avg_tps,
                 "mtp_log": {"lines": acc_lines[-6:]}}
        results[key] = entry
        mtp_stats[config_name] = entry
        save_bench_results(results)
        kill_model(proc, logf)

    # Summarize acceptance
    summary = {}
    for cfg, entry in mtp_stats.items():
        accs = []
        for line in entry.get("mtp_log", {}).get("lines", []):
            m = re.search(r'acceptance = ([\d.]+)', line)
            if m:
                accs.append(float(m.group(1)))
        summary[cfg] = {
            "decode_tps": entry.get("decode_tps_avg"),
            "acceptance_rates": accs,
            "acceptance_avg": round(sum(accs) / len(accs), 3) if accs else None,
        }
    print(f"  MTP acceptance summary: {json.dumps(summary, indent=2)}")
    return summary

def save_bench_results(results):
    with open(BENCH_RESULTS, "w") as f:
        json.dump(results, f, indent=2)

def probe_tps(model):
    """Quick tok/s probe: 256-token completion."""
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
    logpath = os.path.join(LOGS, f"{safe}_v100_lcb.log")

    # Clear previous output (LCB writes dirs by display name AND model key)
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
        "--openai_timeout", "300",
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

    # Parse results — check both output dir naming conventions
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
    logpath = os.path.join(LOGS, f"{safe}_v100_tau2.log")
    save_dir = f"tau2_v100_{safe}"

    agent_model = f"openai/{safe}"
    user_model = "openai/LFM2.5-8B-A1B-Clean-RealWorld-v2"

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
            print(f"SKIP: {model['file']} not on disk yet (still downloading?)")
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
                  "category": model.get("category", ""), "gpu": model.get("gpu", "V100")}
            progress["models"].append(mr)

        # Check user simulator
        try:
            r = urllib.request.urlopen(f"http://localhost:{USER_PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") != "ok":
                print(f"  WARNING: User simulator on :{USER_PORT} not healthy!")
        except:
            print(f"  WARNING: User simulator on :{USER_PORT} not reachable! (tau2 will fail)")

        stop_v100()

        # MTP acceptance tests first (separate server starts, no-spec/n3/n5)
        if model.get("has_mtp"):
            print("  --- MTP acceptance tests ---")
            summary = mtp_acceptance_test(model, mr, progress)
            mr["mtp_acceptance"] = summary
            save_progress(progress)

        # Quality benchmarks (MTP enabled for this model per user request)
        spec_args = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"] if model.get("has_mtp") else None
        proc, logf, logname, err = start_model(model, extra_args=spec_args)
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
            start_v100()
            continue

        mr["thinking"] = model.get("thinking", True)

        try:
            # 1. tok/s probe (no-spec)
            tps = probe_tps(model)
            mr["decode_tps"] = tps
            save_progress(progress)

            # 2. LiveCodeBench
            lcb = run_livecodebench(model["name"], model["lcb_model"])
            mr["livecodebench"] = lcb
            save_progress(progress)

            # 3. tau2-bench
            tau2 = run_tau2(model["name"], model["file"], max_steps=30)
            mr["tau2"] = tau2
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

        start_v100()

    print(f"\n{'='*70}")
    print(f"V100 BENCHMARKS COMPLETE")
    print(f"{'='*70}")
    for model in MODELS:
        for m in progress["models"]:
            if m["name"] == model["name"]:
                lcb = m.get("livecodebench", {})
                tau2 = m.get("tau2", {})
                tps = m.get("decode_tps")
                mtp = m.get("mtp_acceptance", {})
                mtp_str = ""
                if mtp:
                    n3 = mtp.get("MTP-n3", {})
                    mtp_str = f"  MTP-n3: {n3.get('decode_tps')} tok/s @ {n3.get('acceptance_avg')} acc"
                print(f"  {m['name']:<45} LCB={lcb.get('pass_at_1', '?')}  tau2={tau2.get('reward', '?')}  tok/s={tps}{mtp_str}")

if __name__ == "__main__":
    main()
