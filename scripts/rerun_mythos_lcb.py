#!/usr/bin/env python3
"""
Rerun LCB for Qwythos-9B-Claude-Mythos-5-1M MTP Q4_K_M with thinking properly
suppressed. The original run scored 0.40 as an artifact: oai_runner.py's
LCB_DISABLE_THINKING allowlist lacked 'qwythos'/'mythos', so the model thought
by default and burned the 4096-token budget on 39/75 problems -> empty content
-> scored 0. On the 36 problems it answered it went 30/36 (83%).

Run on 3060 (V100 busy with Qwopus3.8 tau2 backfill). Agent server 18099.
Old LCB output dir preserved as <dir>.broken.
"""
import subprocess, json, time, os, urllib.request, shutil, re, sys, glob

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"

NAME = "Qwythos-9B-Claude-Mythos-5-1M MTP Q4_K_M"
FILE = "Qwythos-9B-Claude-Mythos-5-1M-MTP-Q4_K_M.gguf"
LCB_MODEL = "local/mythos-9b-1m"
# lm_styles display name -> LCB output dir
OUT_DIR = os.path.join(LCB_DIR, "output", "Qwythos-9B-Claude-Mythos-5-1M Q4_K_M")


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


def main():
    # 1. preserve broken artifacts
    if os.path.exists(OUT_DIR):
        broken = OUT_DIR + ".thinking-artifact-broken"
        if os.path.exists(broken):
            shutil.rmtree(broken)
        os.rename(OUT_DIR, broken)
        log(f"preserved old output as {os.path.basename(broken)}")

    # 2. start agent server on 3060
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    srv_log = open(os.path.join(LOGS, "mythos_lcb_rerun_server.log"), "w")
    cmd = [BINARY, "-m", os.path.join(LLMS_DIR, FILE), "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "2",
           "--temp", "0.0", "-n", "4096", "--gpu-layers", "99",
           "--ctx-size", "32768", "--ubatch-size", "512",
           "--threads", "6", "--threads-batch", "6",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja"]
    proc = subprocess.Popen(cmd, env=env, stdout=srv_log, stderr=subprocess.STDOUT, text=True)
    for _ in range(240):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            log("agent server up (3060)")
            break
        except Exception:
            pass
        if proc.poll() is not None:
            log("FATAL: server died: " + open(os.path.join(LOGS, "mythos_lcb_rerun_server.log")).read()[-300:])
            sys.exit(1)
    else:
        proc.kill()
        log("FATAL: server timeout")
        sys.exit(1)

    try:
        # 3. LCB with thinking off (allowlist now includes qwythos/mythos)
        lcmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
                "--model", LCB_MODEL,
                "--scenario", "codegeneration", "--release_version", "release_latest",
                "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
                "--num_problems", "75", "--openai_timeout", "300", "--evaluate"]
        lenv = dict(os.environ)
        lenv.update({"OPENAI_KEY": "none",
                     "OPENAI_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
                     "HF_ALLOW_CODE_EVAL": "1",
                     "LCB_DISABLE_THINKING": "1"})
        log("running LCB 75 (thinking OFF)...")
        start = time.time()
        with open(os.path.join(LOGS, "mythos_lcb_rerun.log"), "w") as lf:
            result = subprocess.run(lcmd, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=LCB_DIR, env=lenv, timeout=5 * 3600)
        elapsed = time.time() - start

        # 4. parse pass@1 from eval json (authoritative)
        pass1 = None
        for root, dirs, files in os.walk(OUT_DIR):
            for f in files:
                if f.endswith("_eval.json") and not f.endswith("_all.json"):
                    data = json.load(open(os.path.join(root, f)))
                    if isinstance(data, list) and data:
                        pass1 = data[0].get("pass@1")
        # empty-content audit
        n_empty = n_tot = 0
        for gf in glob.glob(os.path.join(OUT_DIR, "Scenario.codegeneration_1_*.json")):
            if gf.endswith("_all.json") or gf.endswith("_eval.json"):
                continue
            gens = json.load(open(gf))
            for g in gens:
                n_tot += 1
                if not (g.get("output_list") or [""])[0].strip():
                    n_empty += 1
        log(f"LCB pass@1: {pass1} ({elapsed/60:.0f} min) | empty outputs: {n_empty}/{n_tot}")

        # 5. record in progress.json (atomic, minimal race window)
        progress = load_progress()
        for m in progress["models"]:
            if m["name"] == NAME:
                old = m.get("livecodebench", {})
                m["livecodebench"] = {"pass_at_1": pass1,
                                      "wall_time_s": round(elapsed, 1),
                                      "exit_code": result.returncode}
                m.setdefault("failures", []).append({
                    "benchmark": "livecodebench (invalidated run)",
                    "attempt": "1",
                    "error": f"First run scored 0.40 as artifact: LCB_DISABLE_THINKING allowlist "
                             f"lacked 'qwythos'/'mythos', model thought by default, 39/75 outputs "
                             f"empty (4096-token budget consumed by reasoning). Answered problems "
                             f"went 30/36. Rerun with thinking off: {pass1}.",
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                })
                _ = old
                break
        save_progress(progress)
        log("progress.json updated")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        srv_log.close()
    log("DONE")


if __name__ == "__main__":
    main()
