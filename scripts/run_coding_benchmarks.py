#!/usr/bin/env python3
"""
Master coding benchmark orchestrator.
Runs HumanEval + LiveCodeBench on each model sequentially on the 3060.
Tracks progress in JSON. Generates HTML report at the end.
"""
import subprocess
import json
import time
import os
import sys
import signal
import shutil
import urllib.request
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_VENV = "/home/caimlas/bench-venv/bin/python"

# Models in execution order (fast first)
MODELS = [
    {
        "name": "LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M",
        "file": "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf",
        "type": "small",
        "args": ["--gpu-layers", "99", "--ctx-size", "65536", "--ubatch-size", "512",
                 "--threads", "4", "--threads-batch", "4", "--cache-type-k", "q8_0",
                 "--cache-type-v", "q8_0"],
    },
    {
        "name": "LFM2.5-8B-A1B base Q4_K_M",
        "file": "LFM2.5-8B-A1B-Q4_K_M.gguf",
        "type": "small",
        "args": ["--gpu-layers", "99", "--ctx-size", "65536", "--ubatch-size", "512",
                 "--threads", "4", "--threads-batch", "4", "--cache-type-k", "q8_0",
                 "--cache-type-v", "q8_0"],
    },
    {
        "name": "LFM2.5-8B-A1B Q6_K",
        "file": "LFM2.5-8B-A1B-Q6_K.gguf",
        "type": "small",
        "args": ["--gpu-layers", "99", "--ctx-size", "65536", "--ubatch-size", "512",
                 "--threads", "4", "--threads-batch", "4", "--cache-type-k", "q8_0",
                 "--cache-type-v", "q8_0"],
    },
    {
        "name": "qwen2.5-coder-14b-instruct Q4_K_M",
        "file": "qwen2.5-coder-14b-instruct-q4_k_m.gguf",
        "type": "medium",
        "args": ["--gpu-layers", "99", "--ctx-size", "32768", "--ubatch-size", "512",
                 "--threads", "4", "--threads-batch", "4", "--cache-type-k", "q8_0",
                 "--cache-type-v", "q8_0"],
    },
    {
        "name": "Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M",
        "file": "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",
        "type": "moe",
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144",
                 "--ubatch-size", "512", "--threads", "6", "--threads-batch", "6",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    },
    {
        "name": "Qwen3.6-35B-A3B IQ3_K_R4",
        "file": "Qwen3.6-35B-A3B-IQ3_K_R4.gguf",
        "type": "moe",
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144",
                 "--ubatch-size", "512", "--threads", "6", "--threads-batch", "6",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    },
    {
        "name": "Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M",
        "file": "Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf",
        "type": "moe",
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144",
                 "--ubatch-size", "512", "--threads", "6", "--threads-batch", "6",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    },
    {
        "name": "RavenX-OpenFable-Holo3 Q4_K_M",
        "file": "RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
        "type": "moe",
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144",
                 "--ubatch-size", "512", "--threads", "6", "--threads-batch", "6",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
    },
]


def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}


def save_progress(progress):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def stop_production():
    """Stop caimlas-llama to free the 3060"""
    subprocess.run(["sudo", "systemctl", "stop", "caimlas-llama"],
                   capture_output=True, timeout=15)
    time.sleep(3)


def start_production():
    """Restart caimlas-llama production service"""
    subprocess.run(["sudo", "systemctl", "start", "caimlas-llama"],
                   capture_output=True, timeout=15)
    for i in range(60):
        time.sleep(3)
        try:
            req = urllib.request.urlopen(f"http://localhost:8080/health", timeout=3)
            if json.loads(req.read()).get("status") == "ok":
                return True
        except:
            pass
    return False


def start_model(model):
    """Start a model server on port 18099"""
    model_path = os.path.join(LLMS_DIR, model["file"])
    if not os.path.exists(model_path):
        return None, f"Model file not found: {model_path}"

    cmd = [BINARY,
        "--model", model_path,
        "--device", "CUDA0",
        "--flash-attn", "on",
        "--batch-size", "2048",
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--parallel", "1",
        "--temp", "0.0",  # greedy for benchmarks
        "-n", "4096",
    ]
    cmd.extend(model["args"])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"

    log_path = os.path.join(LOGS, f"{model['name'].replace(' ', '_')}_server.log")
    logfile = open(log_path, "w")
    print(f"  Starting model server...")
    proc = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT, text=True)

    for i in range(120):
        time.sleep(2)
        try:
            req = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(req.read()).get("status") == "ok":
                print(f"  Server ready after {i*2}s")
                return proc, None
        except:
            pass
        if proc.poll() is not None:
            logfile.close()
            with open(log_path) as f:
                err = f.read()[-500:]
            return None, f"Server died:\n{err}"

    proc.kill()
    logfile.close()
    return None, "Server timeout (240s)"


def kill_model(proc):
    """Kill the model server"""
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
            proc.wait(timeout=5)
    time.sleep(3)


def throughput_probe(port=PORT):
    """Quick decode speed test"""
    payload = json.dumps({
        "prompt": "Write a Python function that checks if a number is prime.",
        "n_predict": 64,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion",
                                 data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=60)
    data = json.loads(resp.read())
    t = data.get("timings", {})
    decode_tps = round(t.get("predicted_n", 1) / (t.get("predicted_ms", 1) / 1000), 1)
    prompt_tps = round(t.get("prompt_n", 1) / (t.get("prompt_ms", 1) / 1000), 1)
    return decode_tps, prompt_tps


def run_humaneval(model_name, port=PORT):
    """Run HumanEval via lm-evaluation-harness"""
    output_dir = os.path.join(RESULTS, model_name.replace(" ", "_"), "humaneval")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(LOGS, f"{model_name.replace(' ', '_')}_humaneval.log")

    cmd = [
        BENCH_VENV, "-m", "lm_eval",
        "--model", "local-completions",
        "--model_args", f"model={model_name},base_url=http://127.0.0.1:{port}/completion,tokenized_requests=False,required_parameters=[{{'name':'max_tokens','param_type':'int','default':1024}}]",
        "--tasks", "humaneval",
        "--gen_kwargs", "temperature=0.0,max_gen_toks=1024",
        "--batch_size", "1",
        "--output_path", output_dir,
        "--log_samples",
    ]

    print(f"  Running HumanEval (164 problems)...")
    start = time.time()
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                timeout=7200)  # 2h max
    elapsed = time.time() - start

    # Parse results
    # lm-eval writes a results.json in output_path
    results_file = None
    for f in os.listdir(output_dir):
        if f.startswith("results") and f.endswith(".json"):
            results_file = os.path.join(output_dir, f)
            break

    if results_file is None:
        # Try subdirectories
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.startswith("results") and f.endswith(".json"):
                    results_file = os.path.join(root, f)
                    break

    pass_at_1 = None
    if results_file and os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        try:
            pass_at_1 = data["results"]["humaneval"]["acc,none"]
        except (KeyError, TypeError):
            try:
                pass_at_1 = data["results"]["humaneval"]["pass@1,none"]
            except (KeyError, TypeError):
                pass_at_1 = None

    return {
        "pass_at_1": pass_at_1,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
        "results_file": results_file,
    }


def run_livecodebench(model_name, port=PORT):
    """Run LiveCodeBench"""
    output_dir = os.path.join(RESULTS, model_name.replace(" ", "_"), "livecodebench")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(LOGS, f"{model_name.replace(' ', '_')}_livecodebench.log")

    # LiveCodeBench uses its own runner script
    # Check available release data
    lcb_path = "/home/caimlas/git/LiveCodeBench"

    cmd = [
        BENCH_VENV, "-m", "livecodebench",
        "evaluate",
        "--model", model_name,
        "--base-url", f"http://127.0.0.1:{port}/v1",
        "--api-key", "none",
        "--release", "latest",
        "--num-problems", "75",
        "--temperature", "0.0",
        "--max-tokens", "4096",
        "--output-dir", output_dir,
    ]

    print(f"  Running LiveCodeBench (75 problems, latest release)...")
    start = time.time()
    try:
        with open(log_path, "w") as logf:
            result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                    timeout=36000, cwd=lcb_path)  # 10h max
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 36000, "error": "timeout"}
    except Exception as e:
        return {"pass_at_1": None, "wall_time_s": time.time() - start, "error": str(e)}

    # Parse results
    pass_at_1 = None
    results_file = os.path.join(output_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            data = json.load(f)
        try:
            pass_at_1 = data.get("pass_at_1") or data.get("acc") or data.get("score")
        except:
            pass

    # Fallback: scan log for score
    if pass_at_1 is None:
        with open(log_path) as f:
            content = f.read()
        for line in content.split("\n"):
            if "pass@1" in line.lower() or "accuracy" in line.lower():
                import re
                m = re.search(r'(\d+\.?\d*)\s*%', line)
                if m:
                    pass_at_1 = float(m.group(1)) / 100
                    break

    return {
        "pass_at_1": pass_at_1,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
    }


def run_model(model, progress):
    """Run full benchmark suite on one model"""
    name = model["name"]
    print(f"\n{'='*70}")
    print(f"  MODEL: {name}")
    print(f"{'='*70}")

    # Check if already done
    for existing in progress["models"]:
        if existing["name"] == name and existing.get("status") == "completed":
            print(f"  SKIP: already completed")
            return existing

    model_result = {
        "name": name,
        "file": model["file"],
        "type": model["type"],
        "status": "running",
        "start_time": datetime.now(timezone.utc).isoformat(),
    }

    # Stop production service
    print("  Stopping production service...")
    stop_production()

    # Start model
    proc, err = start_model(model)
    if err:
        model_result["status"] = "failed"
        model_result["error"] = err
        model_result["end_time"] = datetime.now(timezone.utc).isoformat()
        progress["models"].append(model_result)
        save_progress(progress)
        # Restart production
        start_production()
        return model_result

    try:
        # Throughput probe
        print("  Throughput probe...")
        decode_tps, prompt_tps = throughput_probe()
        model_result["decode_tps"] = decode_tps
        model_result["prompt_tps"] = prompt_tps
        print(f"  Decode: {decode_tps} tok/s, Prompt: {prompt_tps} tok/s")

        if decode_tps < 24:
            model_result["status"] = "gated"
            model_result["reason"] = f"Throughput {decode_tps} < 24 tok/s threshold"
            print(f"  GATED: {decode_tps} < 24 tok/s, skipping benchmarks")

            # Still run HumanEval as the gate benchmark
            he = run_humaneval(name)
            model_result["humaneval"] = he
            model_result["status"] = "gated"
            model_result["end_time"] = datetime.now(timezone.utc).isoformat()
            progress["models"].append(model_result)
            save_progress(progress)
            return model_result

        # HumanEval
        he = run_humaneval(name)
        model_result["humaneval"] = he
        print(f"  HumanEval pass@1: {he['pass_at_1']}")
        save_progress(progress)

        # LiveCodeBench
        lcb = run_livecodebench(name)
        model_result["livecodebench"] = lcb
        print(f"  LiveCodeBench pass@1: {lcb.get('pass_at_1')}")
        save_progress(progress)

        model_result["status"] = "completed"

    except Exception as e:
        model_result["status"] = "error"
        model_result["error"] = str(e)
        print(f"  ERROR: {e}")
    finally:
        model_result["end_time"] = datetime.now(timezone.utc).isoformat()
        kill_model(proc)
        progress["models"] = [m for m in progress["models"] if m["name"] != name] + [model_result]
        save_progress(progress)

    return model_result


def main():
    progress = load_progress()

    # Check for completed models
    completed_names = {m["name"] for m in progress["models"] if m.get("status") == "completed"}
    gated_names = {m["name"] for m in progress["models"] if m.get("status") == "gated"}

    print(f"Progress: {len(completed_names)} completed, {len(gated_names)} gated, "
          f"{len(MODELS) - len(completed_names) - len(gated_names)} remaining")

    for model in MODELS:
        result = run_model(model, progress)
        print(f"\n  -> {result['status']}: {result['name']}")

    # Restart production
    print("\nRestarting production service...")
    start_production()

    print("\n" + "=" * 70)
    print("ALL BENCHMARKS COMPLETE")
    print("=" * 70)
    for m in progress["models"]:
        he = m.get("humaneval", {})
        lcb = m.get("livecodebench", {})
        print(f"  {m['name']:<45} {m.get('status','?'):<10} "
              f"HE={he.get('pass_at_1','?')}  LCB={lcb.get('pass_at_1','?')}  "
              f"tok/s={m.get('decode_tps','?')}")


if __name__ == "__main__":
    main()
