#!/usr/bin/env python3
"""
Run LiveCodeBench + tau2-bench against each borg model lane, writing the repo's
progress.json format so scripts/generate_report.py can consume it.

Adapted from scripts/run_livecodebench.py and run_tau2bench.py (CUDA host) with the
CUDA-specific parts replaced: models are served by the existing borg lane launchers
(luce_server / Strata / Kyojin), orchestrated exactly like bench_borg_lanes.py.

Run on borg:  sudo /root/lmeval/bin/python3 scripts/borg/run_borg_coding_eval.py
Output:       /root/bench/coding-eval-progress.json  (progress.json format)
"""
import json, os, re, subprocess, time, urllib.request
from datetime import datetime, timezone

LCB_DIR = "/root/LiveCodeBench"
LCB_PY = f"{LCB_DIR}/.venv/bin/python"
TAU2_DIR = "/root/tau2-bench"
TAU2 = f"{TAU2_DIR}/.venv/bin/tau2"
RESULTS = "/root/bench/coding-eval"
LOGS = "/root/bench/coding-eval-logs"
PROGRESS = "/root/bench/coding-eval-progress.json"
PORT = 8902  # fallback; readiness probe scans PORTS
REGISTRY = {
    "local/borg-qwen38-27b": "qwen38-27b",
    "local/borg-qwen38-27b-vision": "qwen38-27b-vision",
    "local/borg-laguna-xs21": "laguna-xs21",
    "local/borg-laguna-s21": "laguna-s21",
    "local/borg-flashnext": "flashnext",
    "local/borg-swift": "swift",
    "local/borg-coder": "coder",
    "local/borg-glm53-flash": "glm53-flash",
    "local/borg-kolibri-1": "kolibri-1",
}
ACTIVE_PORT = PORT
PORTS = (8901, 8902, 8080, 8000)  # all lane ports; server readiness = any responds

os.makedirs(RESULTS, exist_ok=True)
os.makedirs(LOGS, exist_ok=True)

# key, launcher, api model id, tau2 max_steps (context-limited models need fewer)
MODELS = [
    ("qwen38-27b",        "/root/bench/serve_qwen3827b.sh",   "luce",                            30),
    ("qwen38-27b-vision", "/root/bench/serve_qwen38vision.sh","luce",                            30),
    ("laguna-xs21",       "/root/bench/serve_laguna.sh",      "luce",                            30),
    ("laguna-s21",        "/root/bench/serve_laguna_s_split.sh", "luce",                         15),
    ("flashnext",         "/root/strata/run-iq3_s.sh",        "qwen3.8-flash-next-iq3_s",        30),
    ("swift",             "/root/strata/run-swift-iq2_xs.sh", "swift-1.5-iq2_xs",                30),
    ("coder",             "/root/strata/run-coder-iq1_m.sh",  "qwen3.8-flash-next-coder-iq1_m",  30),
    ("glm53-flash",       "/root/bench/serve_kyojin_glm.sh 8000 32768 2", "glm-5.3-exl3",        30),
    ("kolibri-1",         "/root/bench/serve_kolibri.sh",     "Kolibri-1-Q4_K_M.gguf",           15),
]

NUM_LCB_PROBLEMS = 50     # release_latest codegeneration; keeps wall time ~1h/model at borg speeds
NUM_TAU2_TASKS = 15


def load_progress():
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}


def save_progress(p):
    tmp = PROGRESS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, PROGRESS)


def get_entry(progress, name):
    for m in progress["models"]:
        if m["name"] == name:
            return m
    m = {"name": name}
    progress["models"].append(m)
    return m


def start_server(launcher, logname):
    subprocess.run(["/root/bench/cleanup_lanes.sh"], capture_output=True)
    logf = open(os.path.join(LOGS, logname), "w")
    if isinstance(launcher, str):
        cmd = launcher.split()
    else:
        cmd = launcher
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)
    for i in range(180):
        time.sleep(5)
        for port in PORTS:
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=3)
                if b'"data"' in r.read():
                    global ACTIVE_PORT
                    ACTIVE_PORT = port
                    return proc, logf, None
            except Exception:
                pass
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
                if b'"ok"' in r.read():
                    globals()['ACTIVE_PORT'] = port
                    return proc, logf, None
            except Exception:
                pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, logname)) as f:
                return None, None, f"server died: {f.read()[-300:]}"
    return None, None, "timeout waiting for server"


def stop_server(proc, logf):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    if logf:
        logf.close()
    time.sleep(5)
    subprocess.run(["/root/bench/cleanup_lanes.sh"], capture_output=True)


def run_lcb(entry, key, api_model):
    safe = key.replace("/", "_").replace(".", "_")
    outdir = os.path.join(RESULTS, f"lcb-{safe}")
    logpath = os.path.join(LOGS, f"{safe}_lcb.log")
    cmd = [
        LCB_PY, "-m", "lcb_runner.runner.main",
        "--model", f"local/borg-{safe}",  # registry key (see register_lcb_models.py)
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1",
        "--temperature", "0.0",
        "--max_tokens", "4096",
        "--start_date", "2025-04-01",
        "--end_date", "2025-05-01",
        "--openai_timeout", "600",
        "--evaluate",
        "--use_cache",
    ]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{ACTIVE_PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"
    print("  LCB running...", flush=True)
    start = time.time()
    with open(logpath, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=28800,
                       env=env, cwd=LCB_DIR)
    elapsed = time.time() - start
    # find pass@1 in the newest eval json
    pass1 = None
    # LCB writes output under the LanguageModel SHORT name, not the store key
    short = REGISTRY.get(f"local/borg-{safe}", safe)
    for cand in (os.path.join(LCB_DIR, "output", short), os.path.join(LCB_DIR, "output", f"local_borg-{safe}")):
        if not os.path.isdir(cand):
            continue
        for root, dirs, files in os.walk(cand):
            for f in files:
                if f.endswith("_eval.json"):
                    with open(os.path.join(root, f)) as rf:
                        try:
                            data = json.load(rf)
                            if isinstance(data, list) and data and isinstance(data[0], dict):
                                pass1 = data[0].get("pass@1")
                        except Exception:
                            pass
        if pass1 is not None:
            break
    if pass1 is None:  # fallback: grep the log
        content = open(logpath).read()
        m = re.search(r"pass@1[\"']?[:\s]+([0-9.]+)", content)
        if m:
            pass1 = float(m.group(1))
    entry["livecodebench"] = {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1),
                              "num_problems": NUM_LCB_PROBLEMS, "host": "borg"}
    print(f"  LCB pass@1: {pass1}", flush=True)


def run_tau2(entry, key, api_model, max_steps):
    safe = key.replace("/", "_").replace(".", "_")
    logpath = os.path.join(LOGS, f"{safe}_tau2.log")
    save_dir = f"borg_{safe}"
    cmd = [
        TAU2, "run",
        "--domain", "airline",
        "--agent-llm", f"openai/{safe}",
        "--agent-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://127.0.0.1:{ACTIVE_PORT}/v1",
            "temperature": 0.0,
        }),
        # user simulator: same lane server (self-play), keeps the setup single-GPU
        "--user-llm", f"openai/{safe}",
        "--user-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://127.0.0.1:{ACTIVE_PORT}/v1",
            "temperature": 0.0,
        }),
        "--num-tasks", str(NUM_TAU2_TASKS),
        "--num-trials", "1",
        "--max-concurrency", "1",
        "--max-steps", str(max_steps),
        "--max-errors", "5",
        "--timeout", "300",
        "--seed", "42",
        "--save-to", save_dir,
    ]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"
    print("  tau2 running...", flush=True)
    start = time.time()
    with open(logpath, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=43200,
                       env=env, cwd=TAU2_DIR)
    elapsed = time.time() - start
    # parse reward from simulation records
    reward = None
    sim_base = os.path.join(TAU2_DIR, "data", "simulations", save_dir)
    for root, dirs, files in os.walk(sim_base):
        for f in files:
            if f.endswith(".json"):
                try:
                    d = json.load(open(os.path.join(root, f)))
                    recs = d if isinstance(d, list) else [d]
                    rs = [r.get("reward", {}).get("reward") for r in recs
                          if isinstance(r, dict) and isinstance(r.get("reward"), dict)]
                    rs = [x for x in rs if isinstance(x, (int, float))]
                    if rs:
                        reward = sum(rs) / len(rs)
                except Exception:
                    pass
    if reward is None:
        content = open(logpath).read()
        m = re.search(r"[Rr]eward[:\s]+([0-9.]+)", content)
        if m:
            reward = float(m.group(1))
    entry["tau2"] = {"reward": reward, "wall_time_s": round(elapsed, 1),
                     "num_tasks": NUM_TAU2_TASKS, "domain": "airline", "host": "borg"}
    print(f"  tau2 reward: {reward}", flush=True)


def main():
    progress = load_progress()
    for key, launcher, api_model, max_steps in MODELS:
        progress = load_progress()  # re-read each iteration: backfills may land mid-run
        entry_now = next((m for m in progress["models"] if m["name"] == key), None)
        if entry_now is not None and entry_now.get("livecodebench", {}).get("pass_at_1") is not None \
                and entry_now.get("status") == "done":
            print(f"SKIP {key} (LCB done)")
            continue
        print(f"\n{'='*70}\n  {key}\n{'='*70}", flush=True)
        proc, logf, err = start_server(launcher, f"{key}_eval_server.log")
        entry = get_entry(progress, key)
        if err:
            entry["status"] = "failed"
            entry["eval_error"] = err
            save_progress(progress)
            print(f"  SERVER FAILED: {err}", flush=True)
            continue
        try:
            run_lcb(entry, key, api_model)
            save_progress(progress)
            run_tau2(entry, key, api_model, max_steps)
            entry["status"] = "done"
        except Exception as e:
            entry["status"] = "failed"
            entry["eval_error"] = str(e)[:400]
            print(f"  ERROR: {e}", flush=True)
        finally:
            save_progress(progress)
            stop_server(proc, logf)
    print("\nBORG CODING EVAL COMPLETE", flush=True)


if __name__ == "__main__":
    main()
