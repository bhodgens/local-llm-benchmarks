#!/usr/bin/env python3
"""
Backfill tau2-bench (airline, 15 tasks) for the three models missing scores:
  1. BTL-4 Q4_K_M              - agent V100 (original script had no tau2 phase)
  2. Qwen3.5-4B-MTP (ThumbLLM) - agent V100 (original script had no tau2 phase)
  3. Nail-Qwen3.6-35B-A3B UD-Q4_K_XL (stock) - agent 3060 (original run: server
     health-check timeout at 240s during cold 22GB mmap load)

User sim: LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M (whittle_final.sh precedent).
  Phase A: user sim on 3060 (GPU1):8081, agents on V100 (GPU0):18099
  Phase B: user sim on V100  (GPU0):8081, agent  on 3060 (GPU1):18099
"""
import subprocess, json, time, os, urllib.request, shutil, re, sys

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
USER_PORT = 8081
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"

os.makedirs(LOGS, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_progress():
    with open(PROGRESS_FILE) as f:
        return json.load(f)


def save_progress(p):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def get_entry(progress, name):
    for m in progress["models"]:
        if m["name"] == name:
            return m
    return None


def start_user_sim(gpu_id):
    """LFM user sim per whittle_final.sh precedent."""
    cmd = [BINARY,
           "--model", os.path.join(LLMS_DIR, USER_SIM_FILE),
           "--flash-attn", "on", "--host", "0.0.0.0", "--port", str(USER_PORT),
           "--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
           "--threads", "4", "--threads-batch", "4",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logf = open(os.path.join(LOGS, "tau2_backfill_user_sim.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(150):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{USER_PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                log(f"  user sim up on GPU{gpu_id}:{USER_PORT}")
                return proc, logf
        except Exception:
            pass
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError("user sim died: " + open(os.path.join(LOGS, "tau2_backfill_user_sim.log")).read()[-400:])
    proc.kill()
    logf.close()
    raise RuntimeError("user sim timeout")


def stop_user_sim(proc, logf):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    if logf:
        logf.close()
    time.sleep(3)


def start_agent(model_file, args, gpu_id, server_log, health_timeout=240):
    """Start agent server, wait health with generous timeout."""
    path = os.path.join(LLMS_DIR, model_file)
    cmd = [BINARY, "--model", path, "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(PORT),
           "--temp", "0.0"] + args
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    logf = open(server_log, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    t0 = time.time()
    for _ in range(health_timeout // 2):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                log(f"  agent server up after {time.time()-t0:.0f}s")
                return proc, logf, None
        except Exception:
            pass
        if proc.poll() is not None:
            logf.close()
            err = open(server_log).read()[-400:]
            return None, None, f"Server died: {err}"
    proc.kill()
    logf.close()
    return None, None, f"Server timeout ({health_timeout}s)"


def stop_agent(proc, logf):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
    if logf:
        logf.close()
    time.sleep(3)


def run_tau2(model_name):
    """tau2 airline, 15 tasks, seed 42, concurrency 2, max-steps 30 (canonical)."""
    safe = re.sub(r"[/()\[\]]", "", model_name.replace(" ", "_"))
    logpath = os.path.join(LOGS, f"tau2_backfill_{safe}.log")
    save_dir = f"tau2_{safe}"
    sim_dir = os.path.join(TAU2_DIR, "data", "simulations", save_dir)
    if os.path.exists(sim_dir):
        shutil.rmtree(sim_dir)

    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", f"openai/{safe}",
        "--agent-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://127.0.0.1:{PORT}/v1",
            "temperature": 0.0,
        }),
        "--user-llm", f"openai/{USER_SIM_NAME}",
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

    log(f"  running tau2 (airline, 15 tasks)...")
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
    m = re.search(r"Average Reward\s+([\d.]+)", content)
    if m:
        reward = float(m.group(1))
    m2 = re.search(r"Task Pass Rate\s+([\d.]+)", content)
    if m2:
        task_pass = float(m2.group(1))
    log(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {"reward": reward, "task_pass_rate": task_pass,
            "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}


def record_failure(entry, benchmark, error, settings):
    entry.setdefault("failures", []).append({
        "benchmark": benchmark,
        "attempt": "backfill",
        "error": str(error)[:500],
        "settings": settings,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
    })


# --- agent configs (flags lifted from each model's original orchestrator) ---

BTL4 = {
    "name": "BTL-4 Q4_K_M",
    "file": "badtheorylabs_BTL-4-Q4_K_M.gguf",
    "gpu": 0,
    "args": ["--gpu-layers", "99", "--ctx-size", "32768", "--batch-size", "2048",
             "--parallel", "1", "-n", "4096", "--ubatch-size", "512",
             "--threads", "8", "--threads-batch", "8",
             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja",
             "--reasoning", "off", "--reasoning-format", "deepseek"],
}

THUMB = {
    "name": "Qwen3.5-4B-MTP Q4_K_M (ThumbLLM)",
    "file": "Qwen3.5-4B-Q4_K_M.gguf",
    "gpu": 0,
    "args": ["--gpu-layers", "99", "--ctx-size", "32768", "--parallel", "1",
             "-n", "4096", "--ubatch-size", "512",
             "--threads", "8", "--threads-batch", "8",
             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja"],
}

# Nail stock: original 3060 cpu-moe config + --no-mmap (loader warning suggested it)
# with ctx fallback ladder 262K -> 128K -> 32K if KV alloc fails.
NAIL_CTX_LADDER = [["262144"], ["131072"], ["32768"]]

NAIL = {
    "name": "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL",
    "file": "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
    "gpu": 1,
    "args": ["--gpu-layers", "99", "--cpu-moe", "--batch-size", "2048",
             "--parallel", "2", "-n", "4096", "--ubatch-size", "512",
             "--threads", "6", "--threads-batch", "6",
             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
             "--reasoning", "off", "--no-mmap"],
}


def run_agent_tau2(progress, cfg, warm_cache=False):
    entry = get_entry(progress, cfg["name"])
    if entry is None:
        log(f"  FATAL: no progress entry for {cfg['name']}")
        return
    if entry.get("tau2", {}).get("reward") is not None:
        log(f"SKIP {cfg['name']} (tau2 already recorded)")
        return

    if warm_cache:
        log("  warming page cache for model file...")
        subprocess.run(["cat", os.path.join(LLMS_DIR, cfg["file"])],
                       stdout=subprocess.DEVNULL, timeout=1200)

    ctx_variants = NAIL_CTX_LADDER if cfg is NAIL else [None]
    for ci, ctxv in enumerate(ctx_variants):
        args = list(cfg["args"])
        if ctxv:
            for i, a in enumerate(args):
                if a == "--ctx-size":
                    args[i + 1] = ctxv[0]
        settings = f"ctx={ctxv[0] if ctxv else args[args.index('--ctx-size')+1] if '--ctx-size' in args else '?'} gpu=GPU{cfg['gpu']}"
        proc, logf, err = start_agent(
            cfg["file"], args, cfg["gpu"],
            os.path.join(LOGS, "tau2_backfill_agent_server.log"),
            health_timeout=900 if warm_cache else 300)
        if err:
            log(f"  agent start failed (ctx {ctxv}): {err[:200]}")
            record_failure(entry, "tau2", err, settings)
            save_progress(progress)
            continue
        try:
            result = run_tau2(cfg["name"])
            if result.get("reward") is not None:
                entry["tau2"] = result
                # Nail stock: clear stale top-level error/status now that tau2 landed
                if cfg is NAIL:
                    if entry.get("status") == "failed":
                        entry.pop("status", None)
                    old = entry.pop("error", None)
                    if old:
                        record_failure(entry, "tau2 (original run)", old, "cold mmap load, 240s health timeout")
                log(f"  OK {cfg['name']}: reward={result['reward']}")
                return
            else:
                log(f"  no reward (ctx {ctxv}), recording failure")
                record_failure(entry, "tau2", result.get("error", "no reward parsed"), settings)
                save_progress(progress)
        except Exception as e:
            record_failure(entry, "tau2", str(e), settings)
            save_progress(progress)
        finally:
            stop_agent(proc, logf)
    save_progress(progress)


def main():
    progress = load_progress()

    log("=== PHASE A: V100 agents, user sim on 3060 ===")
    usim, ulogf = start_user_sim(1)
    try:
        run_agent_tau2(progress, BTL4)
        save_progress(progress)
        run_agent_tau2(progress, THUMB)
        save_progress(progress)
    finally:
        stop_user_sim(usim, ulogf)

    log("=== PHASE B: 3060 agent (Nail), user sim on V100 ===")
    usim, ulogf = start_user_sim(0)
    try:
        run_agent_tau2(progress, NAIL, warm_cache=True)
        save_progress(progress)
    finally:
        stop_user_sim(usim, ulogf)

    log("=== TAU2 BACKFILL COMPLETE ===")
    for cfg in (BTL4, THUMB, NAIL):
        e = get_entry(progress, cfg["name"])
        t = e.get("tau2", {}) if e else {}
        print(f"  {cfg['name']:<45} tau2={t.get('reward')}", flush=True)


if __name__ == "__main__":
    main()
