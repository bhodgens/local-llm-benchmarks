#!/usr/bin/env python3
"""
Two new models, full lanes, both GPUs.

Lane A (V100, GPU0, port 18096): Ternary-Bonsai dspark retry (with -fit off,
clean GPU - this time caimlas-qwythos gets stopped properly) -> then
DogukanUrker-BTL-4 Q4_K_M full lane (speed, sanity:25, LCB 75 thinking-off,
tau2 15-task w/ LFM user sim on 3060:8081).

Lane B (3060, GPU1, port 18095): MiniCPM5-2B Q8_0 full lane with USER-SUPPLIED
config (temp 1.0, top-p 0.95, 131072 ctx, f16 KV) -> speed, sanity:25, LCB 75
thinking-off, tau2 w/ user sim on V100:8081 (after Lane A's tau2 releases it).

Writes everything into /tmp/coding-bench/progress.json; identical-model
MiniCPM rows separated by GPU tag (MiniCPM5-2B Q8_0 (V100) / (3060)).
"""
import subprocess, json, time, os, re, urllib.request, shutil, glob, sys

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
GATE = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_progress():
    return json.load(open(PROGRESS_FILE))


def save_progress(p):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def get_entry(progress, name, create=None):
    for m in progress["models"]:
        if m["name"] == name:
            return m
    if create:
        e = dict(create)
        e["name"] = name
        progress["models"].append(e)
        return e
    return None


def wait_health(proc, port, logf, tag, tries=240):
    for _ in range(tries):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"{tag} server died: " + open(logf.name).read()[-350:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return
        except Exception:
            pass
    proc.kill()
    logf.close()
    raise RuntimeError(f"{tag} server timeout")


def identity_ok(port, fname):
    try:
        props = urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5).read().decode()
        return fname.replace(".gguf", "") in props
    except Exception:
        return False


def serve(binary, model, gpu, port, extra, tag):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(os.path.join(LOGS, f"newmodels_{tag}_server.log"), "w")
    proc = subprocess.Popen([binary, "--model", model, "--flash-attn", "on",
                             "--host", "127.0.0.1", "--port", str(port)] + extra,
                            env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    wait_health(proc, port, logf, tag)
    if not identity_ok(port, os.path.basename(model)):
        proc.kill(); logf.close()
        raise RuntimeError(f"{tag}: wrong model on port")
    log(f"  {tag} server up (GPU{gpu}:{port}), identity OK")
    return proc, logf


def stop_server(proc, logf):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    logf.close()
    time.sleep(3)


def speed_probe(port, max_tokens=256):
    import http.client
    best = 0.0
    for _ in range(2):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        payload = json.dumps({"messages": [{"role": "user", "content":
            "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False})
        t0 = time.time()
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        d = json.loads(conn.getresponse().read())
        dt = time.time() - t0
        toks = (d.get("usage") or {}).get("completion_tokens")
        conn.close()
        if not toks:
            return {"error": "no completion_tokens"}
        best = max(best, toks / dt)
    return {"decode_tps": round(best, 2)}


def run_sanity(model_file, name, gpu):
    glog = os.path.join(LOGS, f"newmodels_{re.sub(r'[^a-zA-Z0-9]+','_',name)[:30]}_gate.log")
    with open(glog, "w") as lf:
        subprocess.run([BENCH_PY, GATE, "--model-file", model_file,
                        "--name", name, "--gpu", gpu, "--benchmarks", "sanity:25"],
                       stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    frac = re.findall(r"(\d+)\s*/\s*25", open(glog).read())
    return {"sanity_passed": int(frac[-1]), "sanity_pct": round(100 * int(frac[-1]) / 25, 1)} if frac else {"error": "no score"}


def run_lcb(lcb_model, out_name, port, tag):
    out_dir = os.path.join(LCB_DIR, "output", out_name)
    lenv = dict(os.environ)
    lenv.update({"OPENAI_KEY": "none", "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                 "HF_ALLOW_CODE_EVAL": "1", "LCB_DISABLE_THINKING": "1"})
    llog = os.path.join(LOGS, f"newmodels_{tag}_lcb.log")
    t0 = time.time()
    with open(llog, "w") as lf:
        subprocess.run([BENCH_PY, "-m", "lcb_runner.runner.main",
                        "--model", lcb_model, "--scenario", "codegeneration",
                        "--release_version", "release_latest", "--n", "1",
                        "--temperature", "0.0", "--max_tokens", "4096",
                        "--num_problems", "75", "--openai_timeout", "300", "--evaluate"],
                       stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR, env=lenv, timeout=4 * 3600)
    elapsed = time.time() - t0
    pass1 = None
    for f in glob.glob(os.path.join(out_dir, "*_eval.json")):
        if f.endswith("_all.json"):
            continue
        try:
            data = json.load(open(f))
            if isinstance(data, list) and data:
                pass1 = data[0].get("pass@1")
        except Exception:
            pass
    n = empty = 0
    for f in glob.glob(os.path.join(out_dir, "Scenario.codegeneration_1_*.json")):
        if f.endswith("_all.json") or f.endswith("_eval.json"):
            continue
        for g in json.load(open(f)):
            n += 1
            if not (g.get("output_list") or [""])[0].strip():
                empty += 1
    log(f"  LCB {tag}: pass@1={pass1} ({elapsed/60:.0f} min) empty={empty}/{n}")
    return pass1, empty, n, round(elapsed, 1)


def run_tau2(agent_name, agent_port, user_gpu):
    """Canonical tau2: airline/15/seed42/conc2/30. User sim on user_gpu:8081."""
    safe = re.sub(r"[/()\[\]]", "", agent_name.replace(" ", "_"))
    cmd = ["uv", "run", "tau2", "run", "--domain", "airline",
           "--agent-llm", f"openai/{safe}",
           "--agent-llm-args", json.dumps({"api_key": "none",
               "api_base": f"http://127.0.0.1:{agent_port}/v1", "temperature": 0.0}),
           "--user-llm", f"openai/{USER_SIM_NAME}",
           "--user-llm-args", json.dumps({"api_key": "none",
               "api_base": "http://127.0.0.1:8081/v1"}),
           "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
           "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
           "--seed", "42", "--save-to", f"tau2_{safe}"]
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"
    env["CUDA_VISIBLE_DEVICES"] = str(user_gpu)
    t0 = time.time()
    with open(os.path.join(LOGS, f"newmodels_tau2_{safe[:40]}.log"), "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=86400,
                       env=env, cwd=TAU2_DIR)
    content = open(os.path.join(LOGS, f"newmodels_tau2_{safe[:40]}.log")).read()
    m = re.search(r"Average Reward\s+([\d.]+)", content)
    m2 = re.search(r"Task Pass Rate\s+([\d.]+)", content)
    log(f"  tau2 {agent_name}: reward={m.group(1) if m else None} ({(time.time()-t0)/60:.0f} min)")
    return {"reward": float(m.group(1)) if m else None,
            "task_pass_rate": float(m2.group(1)) if m2 else None,
            "wall_time_s": round(time.time() - t0, 1)}


def user_sim_proc(gpu):
    cmd = [BINARY, "--model", os.path.join(LLMS, USER_SIM_FILE),
           "--flash-attn", "on", "--host", "127.0.0.1", "--port", "8081",
           "--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
           "--threads", "4", "--threads-batch", "4", "--cache-type-k", "q8_0",
           "--cache-type-v", "q8_0", "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(os.path.join(LOGS, "newmodels_usersim.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    wait_health(proc, 8081, logf, "usersim")
    log(f"  user sim up on GPU{gpu}:8081")
    return proc, logf


# ---------------------------------------------------------------- lane A
def lane_a():
    progress = load_progress()

    # 0. Bonsai dspark retry - stop production service FIRST (root cause of last OOM)
    log("Lane A: stopping caimlas-qwythos for clean V100...")
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(5)
    dspark_result = {}
    try:
        proc, logf = serve(PRISMML, os.path.join(LLMS, "Ternary-Bonsai-27B-Q2_0.gguf"), 0, 18096,
                           ["--gpu-layers", "99", "--ctx-size", "16384", "--ubatch-size", "512",
                            "--threads", "8", "--threads-batch", "8",
                            "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
                            "-fit", "off",
                            "--spec-draft-model", os.path.join(LLMS, "Ternary-Bonsai-27B-dspark-Q4_1.gguf"),
                            "--spec-draft-n-max", "4", "--jinja"], "bonsai_dspark")
        r = speed_probe(18096)
        dspark_result.update(r)
        log(f"  bonsai dspark probe: {r} (prior 36.3)")
        stop_server(proc, logf)
    except Exception as e:
        dspark_result = {"error": str(e)[:300]}
        log(f"  bonsai dspark FAILED: {str(e)[:200]}")
    progress = load_progress()
    e = get_entry(progress, "Ternary-Bonsai-27B Q2_0 (dspark)")
    if e:
        e["ecc_dspark_retry"] = {**dspark_result, "timestamp": TS}
        save_progress(progress)

    # 1. DogukanUrker BTL-4 full lane
    log("Lane A: DogukanUrker-BTL-4 Q4_K_M full lane (V100)...")
    name = "DogukanUrker-BTL-4 Q4_K_M"
    proc, logf = serve(BINARY, os.path.join(LLMS, "dogukanurker-btl4-q4km/BTL-4-Q4_K_M.gguf"), 0, 18096,
                       ["--gpu-layers", "99", "--ctx-size", "32768", "--batch-size", "2048",
                        "--ubatch-size", "512", "--threads", "8", "--threads-batch", "8",
                        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                        "--parallel", "2", "--temp", "0.0", "-n", "4096", "--jinja"], "du_btl4")
    sp = speed_probe(18096)
    log(f"  speed: {sp}")
    stop_server(proc, logf)

    progress = load_progress()
    e = get_entry(progress, name, create={"file": "dogukanurker-btl4-q4km/BTL-4-Q4_K_M.gguf",
                                          "category": "35B-MoE", "gpu": "V100",
                                          "engine": "llama.cpp", "thinking": False, "template": "stock"})
    e["decode_tps"] = sp.get("decode_tps")
    save_progress(progress)

    san = run_sanity("dogukanurker-btl4-q4km/BTL-4-Q4_K_M.gguf", name, "0")
    log(f"  sanity: {san}")
    progress = load_progress()
    e = get_entry(progress, name)
    e["benchkit"] = {"sanity": san}
    save_progress(progress)

    proc, logf = serve(BINARY, os.path.join(LLMS, "dogukanurker-btl4-q4km/BTL-4-Q4_K_M.gguf"), 0, 18096,
                       ["--gpu-layers", "99", "--ctx-size", "32768", "--batch-size", "2048",
                        "--ubatch-size", "512", "--threads", "8", "--threads-batch", "8",
                        "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                        "--parallel", "2", "--temp", "0.0", "-n", "4096", "--jinja"], "du_btl4")
    usim, ulogf = user_sim_proc(1)
    try:
        pass1, empty, n, wall = run_lcb("local/dogukanurker-btl4-q4km", "DogukanUrker-BTL-4 Q4_K_M", 18096, "du_btl4")
        tau = run_tau2(name, 18096, 1)
    finally:
        stop_server(proc, logf)
        stop_server(usim, ulogf)

    progress = load_progress()
    e = get_entry(progress, name)
    e["livecodebench"] = {"pass_at_1": pass1, "empty": empty, "of": n, "wall_time_s": wall}
    if pass1 is not None:
        e["tau2"] = tau
    else:
        e.setdefault("failures", []).append({"benchmark": "lcb", "error": "no pass@1", "timestamp": TS})
    save_progress(progress)
    log("Lane A COMPLETE")


# ---------------------------------------------------------------- lane B
def lane_b():
    progress = load_progress()
    log("Lane B: MiniCPM5-2B Q8_0 full lane (3060)...")
    if not os.path.exists(os.path.join(LLMS, "MiniCPM5-2B-Q8_0.gguf")):
        log("  MiniCPM5 GGUF missing - download did not finish; aborting lane B")
        return
    name = "MiniCPM5-2B Q8_0 (3060)"
    # USER-SUPPLIED config: -ngl 99 -c 131072 -fa on --jinja -np 1 -ctk f16 -ctv f16
    # -b 2048 -ub 1024 --temp 1.0 --top-p 0.95
    proc, logf = serve(BINARY, os.path.join(LLMS, "MiniCPM5-2B-Q8_0.gguf"), 1, 18095,
                       ["--gpu-layers", "99", "--ctx-size", "131072", "-np", "1",
                        "--batch-size", "2048", "--ubatch-size", "1024",
                        "--cache-type-k", "f16", "--cache-type-v", "f16",
                        "--temp", "1.0", "--top-p", "0.95", "--jinja",
                        "--threads", "8", "--threads-batch", "8"], "minicpm3060")
    sp = speed_probe(18095)
    log(f"  speed (user config): {sp}")
    stop_server(proc, logf)

    e = get_entry(progress, name, create={"file": "MiniCPM5-2B-Q8_0.gguf", "category": "2B-dense",
                                          "gpu": "3060", "engine": "llama.cpp",
                                          "thinking": False, "template": "stock"})
    e["decode_tps"] = sp.get("decode_tps")
    save_progress(progress)

    san = run_sanity("MiniCPM5-2B-Q8_0.gguf", name, "1")
    log(f"  sanity: {san}")
    progress = load_progress()
    e = get_entry(progress, name)
    e["benchkit"] = {"sanity": san}
    save_progress(progress)

    proc, logf = serve(BINARY, os.path.join(LLMS, "MiniCPM5-2B-Q8_0.gguf"), 1, 18095,
                       ["--gpu-layers", "99", "--ctx-size", "131072", "-np", "1",
                        "--batch-size", "2048", "--ubatch-size", "1024",
                        "--cache-type-k", "f16", "--cache-type-v", "f16",
                        "--temp", "1.0", "--top-p", "0.95", "--jinja",
                        "--threads", "8", "--threads-batch", "8"], "minicpm3060")
    usim, ulogf = user_sim_proc(0)  # user sim on V100 (Lane A tau2 done by then)
    try:
        pass1, empty, n, wall = run_lcb("local/minicpm5-2b-q8-3060", "MiniCPM5-2B Q8_0 (3060)", 18095, "minicpm3060")
        tau = run_tau2(name, 18095, 0)
    finally:
        stop_server(proc, logf)
        stop_server(usim, ulogf)

    progress = load_progress()
    e = get_entry(progress, name)
    e["livecodebench"] = {"pass_at_1": pass1, "empty": empty, "of": n, "wall_time_s": wall}
    if pass1 is not None:
        e["tau2"] = tau
    else:
        e.setdefault("failures", []).append({"benchmark": "lcb", "error": "no pass@1", "timestamp": TS})
    save_progress(progress)
    log("Lane B COMPLETE")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    # Production V100 service must stay down across BOTH lanes: BTL-4 needs the
    # whole V100, and lane B's user sim runs there too. Restart only at the end
    # (also on failure - never leave production down).
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(5)
    try:
        if which in ("a", "both"):
            lane_a()
        if which in ("b", "both"):
            lane_b()
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "caimlas-qwythos"], capture_output=True)
        time.sleep(15)
    log("ALL DONE")
