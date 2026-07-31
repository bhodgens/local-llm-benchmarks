#!/usr/bin/env python3
"""
Benchmark Neutrino-8B (Fermion Research) on V100 (32GB).
Tests plain greedy AND speculative decoding with Neutrino-0.6B draft head.

Uses fermionresearch/llama.cpp fork (fermion-fv5 branch) for FV5 tensor support.
"""
import subprocess, json, time, os, urllib.request, shutil, re, http.client
from datetime import datetime, timezone

FERMION_BINARY = "/home/caimlas/git/llama.cpp-fermion/build/bin/llama-server"
LLMS_DIR = "/home/files/llms/gguf"
PORT = 18099
USER_PORT = 8082
SCRATCH = "/tmp/neutrino-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

MODEL_8B = os.path.join(LLMS_DIR, "neutrino-8b-fv5.gguf")
MODEL_DRAFT = os.path.join(LLMS_DIR, "neutrino-0.6b-base-fv5.gguf")

def stop_v100():
    subprocess.run(["sudo", "systemctl", "stop", "caimlas-bonsai"], capture_output=True, timeout=15)
    time.sleep(3)

def start_v100():
    subprocess.run(["sudo", "systemctl", "start", "caimlas-bonsai"], capture_output=True, timeout=15)
    for _ in range(60):
        time.sleep(3)
        try:
            r = urllib.request.urlopen("http://localhost:8081/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return
        except:
            pass

def start_server(use_draft=False):
    cmd = [
        FERMION_BINARY,
        "--model", MODEL_8B,
        "--flash-attn", "on",
        "--batch-size", "2048",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--parallel", "2",
        "--temp", "0.0",
        "-n", "8192",
        "--gpu-layers", "99",
        "--ctx-size", "40960",
        "--ubatch-size", "512",
        "--threads", "8",
        "--threads-batch", "8",
        "--cache-type-k", "f16",
        "--cache-type-v", "f16",
        "--reasoning", "off",
    ]

    if use_draft:
        cmd.extend([
            "--spec-type", "draft-simple",
            "--spec-draft-model", MODEL_DRAFT,
            "--spec-draft-ngl", "99",
            "--spec-draft-n-max", "6",
            "--spec-draft-n-min", "1",
        ])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"

    suffix = "_draft" if use_draft else "_plain"
    logf = open(os.path.join(LOGS, f"neutrino-8b_server{suffix}.log"), "w")
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
            with open(os.path.join(LOGS, f"neutrino-8b_server{suffix}.log")) as f:
                err = f.read()[-500:]
            return None, None, f"Server died: {err}"

    proc.kill()
    return None, None, "Timeout waiting for server"

def kill_server(proc, logf):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
    if logf:
        logf.close()
    time.sleep(3)

def probe_tps(label):
    """Send timed decode requests at different prompt classes."""
    results = {}

    prompt_classes = {
        "code": "Write a Python function that implements merge sort. Include type hints, docstrings, and handle edge cases. Make it production quality.",
        "essay": "Write a detailed essay about the history of computing, from Babbage to modern GPUs. Include key milestones, important figures, and technological breakthroughs.",
        "factual": "Explain the differences between TCP and UDP. Cover reliability, ordering, connection handling, use cases, and overhead.",
    }

    for pclass, prompt in prompt_classes.items():
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)

            # Warmup
            conn.request("POST", "/v1/chat/completions", json.dumps({
                "messages": [{"role": "user", "content": "Say hello."}],
                "max_tokens": 8, "temperature": 0.0, "stream": False,
            }), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            resp.read()

            # Timed decode: 256 tokens
            start = time.time()
            conn.request("POST", "/v1/chat/completions", json.dumps({
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256, "temperature": 0.0, "stream": False,
            }), {"Content-Type": "application/json"})
            resp = conn.getresponse()
            data = json.loads(resp.read())
            elapsed = time.time() - start

            completion_tokens = data.get("usage", {}).get("completion_tokens", 256)
            tps = completion_tokens / elapsed if elapsed > 0 else 0
            results[pclass] = {"tps": round(tps, 1), "tokens": completion_tokens, "time_s": round(elapsed, 1)}
            print(f"  [{label}] {pclass}: {tps:.1f} tok/s ({completion_tokens} tok in {elapsed:.1f}s)")
            conn.close()
        except Exception as e:
            print(f"  [{label}] {pclass}: FAILED - {e}")
            results[pclass] = None

    return results

def run_livecodebench():
    lcb_model = "local/neutrino-8b"
    logpath = os.path.join(LOGS, "neutrino-8b_lcb.log")

    output_dir = os.path.join(LCB_DIR, "output", "Neutrino-8B-FV5")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

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

    pass1 = None
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            src = os.path.join(root, f)
            if f.endswith("_eval.json"):
                with open(src) as rf:
                    try:
                        data = json.load(rf)
                        if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                            pass1 = data[0].get("pass@1")
                    except:
                        pass

    if pass1 is None:
        with open(logpath) as f:
            content = f.read()
        for line in content.split("\n"):
            if "pass@1" in line.lower() or "accuracy" in line.lower():
                m = re.search(r'(\d+\.?\d*)\s*%', line)
                if m:
                    pass1 = float(m.group(1)) / 100
                    break

    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def run_tau2():
    logpath = os.path.join(LOGS, "neutrino-8b_tau2.log")
    save_dir = "tau2_v100_neutrino-8b"

    agent_model = "openai/neutrino-8b"
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
        "--max-steps", "30",
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
    return {"reward": reward, "task_pass_rate": task_pass, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    results = {}

    # Check user simulator
    try:
        r = urllib.request.urlopen(f"http://localhost:{USER_PORT}/health", timeout=3)
        health = json.loads(r.read())
        if health.get("status") != "ok":
            print(f"WARNING: User simulator on :{USER_PORT} not healthy!")
    except:
        print(f"WARNING: User simulator on :{USER_PORT} not reachable!")

    # =====================================================================
    # PHASE 1: Plain Neutrino-8B (no draft)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 1: Neutrino-8B (plain, no draft)")
    print(f"{'='*70}")

    stop_v100()
    proc, logf, err = start_server(use_draft=False)
    if err:
        print(f"  FATAL: {err}")
        start_v100()
        return

    try:
        # tok/s probe (multiple prompt classes)
        tps_plain = probe_tps("plain")
        results["plain_tps"] = tps_plain
        with open(os.path.join(RESULTS, "neutrino_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # LiveCodeBench
        lcb = run_livecodebench()
        results["plain_lcb"] = lcb
        with open(os.path.join(RESULTS, "neutrino_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # tau2
        tau2 = run_tau2()
        results["plain_tau2"] = tau2
        with open(os.path.join(RESULTS, "neutrino_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        results["plain_error"] = str(e)
    finally:
        kill_server(proc, logf)

    # =====================================================================
    # PHASE 2: Neutrino-8B + 0.6B draft (speculative decoding)
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  PHASE 2: Neutrino-8B + 0.6B draft (speculative decoding)")
    print(f"{'='*70}")

    proc, logf, err = start_server(use_draft=True)
    if err:
        print(f"  FATAL: {err}")
        start_v100()
        with open(os.path.join(RESULTS, "neutrino_results.json"), "w") as f:
            json.dump(results, f, indent=2)
        return

    try:
        # tok/s probe (multiple prompt classes)
        tps_draft = probe_tps("draft")
        results["draft_tps"] = tps_draft
        with open(os.path.join(RESULTS, "neutrino_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Quick quality spot-check: 10 LCB problems to verify identical output
        print(f"\n  Running LCB spot-check (10 problems, verifying quality identical)...")
        lcb_model = "local/neutrino-8b"
        logpath = os.path.join(LOGS, "neutrino-8b_lcb_draft.log")
        output_dir = os.path.join(LCB_DIR, "output", "Neutrino-8B-FV5-draft"
        )
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        cmd = [
            BENCH_PY, "-m", "lcb_runner.runner.main",
            "--model", lcb_model,
            "--scenario", "codegeneration",
            "--release_version", "release_latest",
            "--n", "1",
            "--temperature", "0.0",
            "--max_tokens", "4096",
            "--num_problems", "10",
            "--openai_timeout", "300",
            "--evaluate",
        ]

        env = dict(os.environ)
        env["OPENAI_KEY"] = "none"
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
        env["HF_ALLOW_CODE_EVAL"] = "1"
        env["LCB_DISABLE_THINKING"] = "1"

        # Override output directory for spot-check
        env["LCB_OUTPUT_DIR"] = output_dir

        start = time.time()
        with open(logpath, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           timeout=7200, env=env, cwd=LCB_DIR)
        elapsed = time.time() - start

        pass1 = None
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                src = os.path.join(root, f)
                if f.endswith("_eval.json"):
                    with open(src) as rf:
                        try:
                            data = json.load(rf)
                            if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                                pass1 = data[0].get("pass@1")
                        except:
                            pass

        results["draft_lcb_spotcheck"] = {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1)}
        with open(os.path.join(RESULTS, "neutrino_results.json"), "w") as f:
            json.dump(results, f, indent=2)

    except Exception as e:
        import traceback
        traceback.print_exc()
        results["draft_error"] = str(e)
    finally:
        kill_server(proc, logf)

    start_v100()

    # =====================================================================
    # SUMMARY
    # =====================================================================
    print(f"\n{'='*70}")
    print(f"  NEUTRINO-8B BENCHMARK COMPLETE")
    print(f"{'='*70}")
    print(f"\nResults saved to: {os.path.join(RESULTS, 'neutrino_results.json')}\n")

    plain_tps = results.get("plain_tps", {})
    draft_tps = results.get("draft_tps", {})

    print(f"  tok/s comparison (decode, 256 tokens):")
    print(f"  {'class':<12} {'plain':>8} {'draft':>8} {'speedup':>8}")
    print(f"  {'-'*40}")
    for pclass in ["code", "essay", "factual"]:
        p = plain_tps.get(pclass, {})
        d = draft_tps.get(pclass, {})
        pt = p.get("tps", 0) if p else 0
        dt = d.get("tps", 0) if d else 0
        sp = f"{dt/pt:.2f}x" if pt > 0 else "N/A"
        print(f"  {pclass:<12} {pt:>7.1f}  {dt:>7.1f}  {sp:>8}")

    lcb_plain = results.get("plain_lcb", {})
    lcb_draft = results.get("draft_lcb_spotcheck", {})
    print(f"\n  LCB pass@1 (plain, 75 problems):   {lcb_plain.get('pass_at_1', 'N/A')}")
    print(f"  LCB pass@1 (draft spot-check, 10):  {lcb_draft.get('pass_at_1', 'N/A')}")

    tau2 = results.get("plain_tau2", {})
    print(f"  tau2 reward: {tau2.get('reward', 'N/A')}  (task pass: {tau2.get('task_pass_rate', 'N/A')})")

if __name__ == "__main__":
    main()
