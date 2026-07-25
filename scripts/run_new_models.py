#!/usr/bin/env python3
"""
Benchmark new models on both GPUs.

Phase 1: 3060 benchmarks (CUDA1, stop gemma if running, user sim = Bonsai :8081)
  - Nanbeige4-3B-Thinking Q8_0
  - Nanbeige4-3B-Thinking Q4_K_M
  - BTL-3-Compact AVQ2
  - Laguna-S-2.1 UD-IQ3_S

Phase 2: V100 benchmarks (CUDA0, stop bonsai if running, user sim = Gemma :8080)
  - BTL-3-Compact AVQ2
  - Laguna-S-2.1 UD-IQ3_S

Each model: tok/s probe + LiveCodeBench (75 problems, thinking off) + tau2-bench (airline, 15 tasks).
Adaptive retry on OOM: reduce ctx -> reduce KV type.
"""
import subprocess, json, time, os, urllib.request, shutil, re, http.client, sys
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099  # Benchmark model server
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

# Services
BONSAI_SERVICE = "caimlas-bonsai"
GEMMA_SERVICE = "caimlas-gemma"
BONSAI_PORT = 8081   # V100
GEMMA_PORT = 8080    # 3060

# ─── Model configs ────────────────────────────────────────────────────────────
# Each entry defines: name, file, lcb_model, gpu ("3060" or "V100"),
#   binary override, extra llama-server args, thinking flag, category.
# ctx and KV type are initial targets; adaptive retry reduces on OOM.

PHASE1_3060 = [
    {
        "name": "Nanbeige4-3B-Thinking Q8_0",
        "file": "Nanbeige4-3B-Thinking-Q8_0.gguf",
        "lcb_model": "local/nanbeige4-3b-q8",
        "gpu": "3060",
        "cuda_dev": "1",
        "ctx_size": 131072,
        "kv_k": "q4_0", "kv_v": "q4_0",
        "threads": "6", "threads_batch": "6",
        "thinking": True,
        "category": "3B",
    },
    {
        "name": "Nanbeige4-3B-Thinking Q4_K_M",
        "file": "Nanbeige4-3B-Thinking-Q4_K_M.gguf",
        "lcb_model": "local/nanbeige4-3b-q4km",
        "gpu": "3060",
        "cuda_dev": "1",
        "ctx_size": 131072,
        "kv_k": "q4_0", "kv_v": "q4_0",
        "threads": "6", "threads_batch": "6",
        "thinking": True,
        "category": "3B",
    },
    {
        "name": "BTL-3-Compact AVQ2",
        "file": "BTL-3-Compact-AVQ2.gguf",
        "lcb_model": "local/btl3-compact",
        "gpu": "3060",
        "cuda_dev": "1",
        "ctx_size": 32768,
        "kv_k": "q4_0", "kv_v": "q4_0",
        "threads": "6", "threads_batch": "6",
        "thinking": True,
        "category": "27B",
    },
    {
        "name": "Laguna-S-2.1 UD-IQ3_S",
        "file": "Laguna-S-2.1-UD-IQ3_S.gguf",
        "lcb_model": "local/laguna-s-2.1",
        "gpu": "3060",
        "cuda_dev": "1",
        "ctx_size": 32768,
        "kv_k": "q8_0", "kv_v": "q8_0",
        "threads": "8", "threads_batch": "8",
        "extra_args": ["--n-cpu-moe", "44", "--gpu-layers", "999"],
        "thinking": True,
        "category": "118B MoE",
    },
    {
        "name": "Hermes3.6-35B-A3B Genesis V5 APEX-Compact",
        "file": "Hermes3.6-35B-A3B-Uncensored-Genesis-V5-APEX-Compact.gguf",
        "lcb_model": "local/hermes-v5-apex",
        "gpu": "3060",
        "cuda_dev": "1",
        "ctx_size": 131072,
        "kv_k": "q8_0", "kv_v": "q8_0",
        "threads": "6", "threads_batch": "6",
        "extra_args": ["--cpu-moe", "--gpu-layers", "99"],
        "thinking": True,
        "category": "35B MoE",
    },
]

PHASE2_V100 = [
    {
        "name": "BTL-3-Compact AVQ2 (V100)",
        "file": "BTL-3-Compact-AVQ2.gguf",
        "lcb_model": "local/btl3-compact",
        "gpu": "V100",
        "cuda_dev": "0",
        "ctx_size": 262144,
        "kv_k": "q8_0", "kv_v": "q8_0",
        "threads": "8", "threads_batch": "8",
        "thinking": True,
        "category": "27B",
    },
    {
        "name": "Laguna-S-2.1 UD-IQ3_S (V100)",
        "file": "Laguna-S-2.1-UD-IQ3_S.gguf",
        "lcb_model": "local/laguna-s-2.1",
        "gpu": "V100",
        "cuda_dev": "0",
        "ctx_size": 65536,
        "kv_k": "q8_0", "kv_v": "q8_0",
        "threads": "8", "threads_batch": "8",
        "extra_args": ["--n-cpu-moe", "44", "--gpu-layers", "999"],
        "thinking": True,
        "category": "118B MoE",
    },
    {
        "name": "ThinkingCap-Qwen3.6-27B Q4_K_M (V100)",
        "file": "ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf",
        "lcb_model": "local/thinkingcap-qwen36-27b",
        "gpu": "V100",
        "cuda_dev": "0",
        "ctx_size": 131072,
        "kv_k": "q8_0", "kv_v": "q8_0",
        "threads": "8", "threads_batch": "8",
        "thinking": True,
        "category": "27B",
    },
]

# ─── Progress helpers ─────────────────────────────────────────────────────────

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def get_or_create_entry(progress, model):
    for m in progress["models"]:
        if m["name"] == model["name"]:
            return m
    mr = {
        "name": model["name"],
        "file": model["file"],
        "category": model.get("category", ""),
        "gpu": model["gpu"],
        "thinking": model.get("thinking", False),
    }
    progress["models"].append(mr)
    save_progress(progress)
    return mr

# ─── Service management ───────────────────────────────────────────────────────

def systemctl(action, service):
    subprocess.run(["sudo", "systemctl", action, service], capture_output=True, timeout=30)
    time.sleep(3)

def wait_health(port, timeout=120):
    for _ in range(timeout // 3):
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return True
        except:
            pass
    return False

def wait_health_tabby(port, timeout=120):
    """TabbyAPI /health returns 200 but body differs."""
    for _ in range(timeout // 3):
        time.sleep(3)
        try:
            r = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            if r.status == 200:
                return True
        except:
            pass
    return False

# ─── Model server lifecycle ──────────────────────────────────────────────────

def start_model(model, ctx_override=None, kv_override=None):
    """Start benchmark model server. Returns (proc, logf, error_str)."""
    path = os.path.join(LLMS_DIR, model["file"])
    if not os.path.exists(path):
        return None, None, f"Model file not found: {path}"

    ctx = ctx_override or model["ctx_size"]
    kv_k = (kv_override or {}).get("k", model["kv_k"])
    kv_v = (kv_override or {}).get("v", model["kv_v"])

    cmd = [
        BINARY, "--model", path, "--flash-attn", "on",
        "--batch-size", "2048", "--ubatch-size", "512",
        "--host", "0.0.0.0", "--port", str(PORT),
        "--parallel", "2", "--temp", "0.0", "-n", "4096",
        "--ctx-size", str(ctx),
        "--cache-type-k", kv_k, "--cache-type-v", kv_v,
        "--threads", model["threads"], "--threads-batch", model["threads_batch"],
        "--gpu-layers", "99",  # default; overridden by extra_args if present
    ]
    if "extra_args" in model:
        # Replace default gpu-layers if extra_args specifies it
        cmd = [x for x in cmd if x not in ("--gpu-layers", "99")]
        cmd.extend(model["extra_args"])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = model["cuda_dev"]

    safe = model["name"].replace(" ", "_").replace("(", "").replace(")", "")
    logf = open(os.path.join(LOGS, f"{safe}_{model['gpu']}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)

    for _ in range(180):  # 6 min max
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                print(f"  Server up (ctx={ctx}, kv={kv_k})")
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{safe}_{model['gpu']}_server.log")) as f:
                err = f.read()[-800:]
            return None, None, f"Server died: {err}"

    proc.kill()
    return None, None, "Timeout waiting for server"

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

# ─── Benchmarks ───────────────────────────────────────────────────────────────

def probe_tps(model):
    """Decode tok/s: 256-token completion at modest context."""
    safe = model["name"].replace(" ", "_")
    print(f"  Probing decode tok/s...")
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
        # Warmup
        payload = json.dumps({
            "messages": [{"role": "user", "content": "Say hello."}],
            "max_tokens": 8, "temperature": 0.0, "stream": False,
        })
        conn.request("POST", "/v1/chat/completions", payload,
                      {"Content-Type": "application/json"})
        conn.getresponse().read()

        # Timed decode
        payload = json.dumps({
            "messages": [{"role": "user", "content":
                "Write a detailed essay about the history of computing, "
                "from Babbage to modern GPUs. Include key milestones, "
                "important figures, and technological breakthroughs."}],
            "max_tokens": 256, "temperature": 0.0, "stream": False,
        })
        start = time.time()
        conn.request("POST", "/v1/chat/completions", payload,
                      {"Content-Type": "application/json"})
        data = json.loads(conn.getresponse().read())
        elapsed = time.time() - start
        usage = data.get("usage", {})
        n_tok = usage.get("completion_tokens", 256)
        tps = n_tok / elapsed if elapsed > 0 else 0
        print(f"  tok/s: {tps:.1f} ({n_tok} tok in {elapsed:.1f}s)")
        return round(tps, 1)
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None

def run_livecodebench(model):
    safe = model["name"].replace(" ", "_").replace("(", "").replace(")", "")
    logpath = os.path.join(LOGS, f"{safe}_{model['gpu']}_lcb.log")
    lcb_model = model["lcb_model"]
    output_dir = os.path.join(LCB_DIR, "output", lcb_model.replace("/", "_"))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", lcb_model,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1", "--temperature", "0.0",
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

    # Parse pass@1 from eval JSONs
    pass1 = None
    empty_count = 0
    for root, dirs, files in os.walk(output_dir):
        for f in files:
            if f.endswith("_eval.json"):
                with open(os.path.join(root, f)) as rf:
                    try:
                        data = json.load(rf)
                        if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                            pass1 = data[0].get("pass@1")
                    except: pass
            if f.endswith("_sample_0.json"):
                with open(os.path.join(root, f)) as rf:
                    try:
                        data = json.load(rf)
                        if isinstance(data, list):
                            for item in data:
                                code = item.get("code_answer", "")
                                if not code or not code.strip():
                                    empty_count += 1
                    except: pass

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
    return {"pass_at_1": pass1, "empty_outputs": empty_count,
            "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def run_tau2(model, user_port):
    safe = model["name"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
    logpath = os.path.join(LOGS, f"{safe}_{model['gpu']}_tau2.log")
    save_dir = f"tau2_{model['gpu']}_{safe}"
    agent_model_id = f"openai/{safe}"

    # User simulator model name (doesn't need to match actual model for local)
    user_model_id = "openai/gpt-4o-mini"

    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", agent_model_id,
        "--agent-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://127.0.0.1:{PORT}/v1",
            "temperature": 0.0,
        }),
        "--user-llm", user_model_id,
        "--user-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://localhost:{user_port}/v1",
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

    print(f"  Running tau2-bench (airline, 15 tasks, user sim on :{user_port})...")
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
    m = re.search(r'Average Reward\s+([\d.]+)', content)
    if m: reward = float(m.group(1))
    m = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    if m: task_pass = float(m.group(1))

    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {"reward": reward, "task_pass_rate": task_pass,
            "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

# ─── Adaptive retry settings ──────────────────────────────────────────────────

def get_retry_settings(model, attempt):
    """Return reduced ctx/KV for retry attempt, or None if exhausted."""
    ctx = model["ctx_size"]
    kv_k = model["kv_k"]
    kv_v = model["kv_v"]

    if attempt == 1:
        # Reduce context by half
        return ctx // 2, {"k": "q4_0", "v": "q4_0"}
    elif attempt == 2:
        # Quarter context
        return ctx // 4, {"k": "q4_0", "v": "q4_0"}
    elif attempt == 3:
        # Minimal
        return max(ctx // 8, 4096), {"k": "q4_0", "v": "q4_0"}
    return None, None

# ─── Main benchmark runner ───────────────────────────────────────────────────

def benchmark_model(model, progress, user_port):
    """Run full benchmark suite for one model. Returns result entry."""
    print(f"\n{'='*70}")
    print(f"  {model['name']}  [{model['gpu']}]")
    print(f"{'='*70}")

    mr = get_or_create_entry(progress, model)

    # Check user sim is up (for tau2)
    user_sim_ok = False
    try:
        urllib.request.urlopen(f"http://localhost:{user_port}/health", timeout=5)
        user_sim_ok = True
    except:
        print(f"  WARNING: User simulator on :{user_port} not reachable!")

    # Start model with adaptive retry
    proc, logf, err = start_model(model)
    attempt = 0
    while err and attempt < 3:
        attempt += 1
        new_ctx, new_kv = get_retry_settings(model, attempt)
        if new_ctx is None:
            break
        print(f"  Retry #{attempt}: ctx={new_ctx}, kv={new_kv}")
        if "failures" not in mr: mr["failures"] = []
        mr["failures"].append({
            "benchmark": "server_start",
            "error": str(err)[:300],
            "attempt": attempt,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        save_progress(progress)
        proc, logf, err = start_model(model, ctx_override=new_ctx, kv_override=new_kv)

    if err:
        print(f"  FATAL: {err[:300]}")
        mr["lcb_error"] = err
        if "failures" not in mr: mr["failures"] = []
        mr["failures"].append({
            "benchmark": "all", "error": str(err)[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        save_progress(progress)
        return

    try:
        # 1. tok/s probe
        tps = probe_tps(model)
        mr["decode_tps"] = tps
        save_progress(progress)

        # 2. LiveCodeBench
        lcb = run_livecodebench(model)
        mr["livecodebench"] = lcb
        save_progress(progress)

        # 3. tau2-bench
        if user_sim_ok:
            tau2 = run_tau2(model, user_port)
            mr["tau2"] = tau2
        else:
            print("  SKIP tau2 (no user simulator)")
            mr["tau2"] = {"reward": None, "error": "no user simulator"}
        save_progress(progress)

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        if "failures" not in mr: mr["failures"] = []
        mr["failures"].append({
            "benchmark": "unknown", "error": str(e)[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        save_progress(progress)
    finally:
        kill_model(proc, logf)
        save_progress(progress)

def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    progress = load_progress()

    done_names = set()
    for m in progress["models"]:
        lb = m.get("livecodebench", {})
        if lb.get("pass_at_1") is not None:
            done_names.add(m["name"])

    # ─── Phase 1: 3060 benchmarks ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 1: 3060 BENCHMARKS (CUDA1)")
    print("=" * 70)

    # Ensure gemma is stopped (free 3060 VRAM)
    systemctl("stop", GEMMA_SERVICE)
    # Start Bonsai on V100 as user sim
    systemctl("start", BONSAI_SERVICE)
    if not wait_health(BONSAI_PORT, timeout=180):
        print("WARNING: Bonsai (user sim) not healthy on :8081!")

    for model in PHASE1_3060:
        if model["name"] in done_names:
            print(f"SKIP: {model['name']} (already done)")
            continue
        benchmark_model(model, progress, BONSAI_PORT)

    # ─── Phase 2: V100 benchmarks ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PHASE 2: V100 BENCHMARKS (CUDA0)")
    print("=" * 70)

    # Stop Bonsai (free V100 VRAM), start Gemma as user sim
    systemctl("stop", BONSAI_SERVICE)
    systemctl("start", GEMMA_SERVICE)
    if not wait_health_tabby(GEMMA_PORT, timeout=120):
        print("WARNING: Gemma (user sim) not healthy on :8080!")

    for model in PHASE2_V100:
        if model["name"] in done_names:
            print(f"SKIP: {model['name']} (already done)")
            continue
        benchmark_model(model, progress, GEMMA_PORT)

    # ─── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  ALL BENCHMARKS COMPLETE")
    print("=" * 70)
    for m in sorted(progress["models"], key=lambda x: x.get("livecodebench", {}).get("pass_at_1") or 0, reverse=True):
        lb = m.get("livecodebench", {})
        t2 = m.get("tau2", {})
        tps = m.get("decode_tps")
        print(f"  {m['name']:<45} [{m.get('gpu','?'):>4}]  "
              f"LCB={lb.get('pass_at_1','?')}  tau2={t2.get('reward','?')}  tok/s={tps}")

if __name__ == "__main__":
    main()
