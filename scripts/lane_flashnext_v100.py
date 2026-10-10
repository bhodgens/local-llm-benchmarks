#!/usr/bin/env python3
"""
Qwen3.8 Flash Next GSQ-RCO Q2_0 - FULL LANE on V100 (2026-10-10).

Best config from probe ladder: --n-cpu-moe 8 -fit on --gpu-layers 99
(12.21 t/s decode, 31.1 GB VRAM at 8K ctx; N=16: 8.02, N=24: 5.7,
cpu-moe: 1.63, full offload OOMs at 37.3 GB; MTP head not in this quant).

Legs: speed -> sanity:25 -> LCB 75 (thinking off, allowlist patched
'flash-next') -> tau2 airline/15/seed42/conc2, LFM user sim on 3060
(caimlas-gemma4-3060 stopped for the tau2 leg only).

Prod: caimlas-btl4-v100 + caimlas-lfm25-v100 stopped ONCE for the whole run,
restored in a single top-level finally; gemma4-3060 stopped/restored around
the tau2 leg only.

Merges each leg into /tmp/coding-bench/progress.json as it completes;
regenerates report.html at the end. Never overwrites a score with None.
"""
import subprocess, json, time, os, re, urllib.request, http.client, glob

XING4 = "/home/caimlas/git/llama.cpp-xing4/build/bin/llama-server"
UPSTREAM = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
MODEL = "/home/files/llms/flashnext-q20/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs/flashnext_lane"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
RESULTS_LOG = "/tmp/coding-bench/flashnext_lane_results.json"
ENTRY_NAME = "Qwen3.8-Flash-Next GSQ-RCO Q2_0 (V100)"
PORT = 18096
USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"
V100_PRODS = ["caimlas-btl4-v100", "caimlas-lfm25-v100"]
G30_PROD = "caimlas-gemma4-3060"

BEST_EXTRA = ["--n-cpu-moe", "8", "-fit", "on", "--gpu-layers", "99"]

os.makedirs(LOGS, exist_ok=True)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def sudo(*args):
    return subprocess.run(["sudo", "-n"] + list(args), capture_output=True, text=True)

def save_results(r):
    with open(RESULTS_LOG + ".tmp", "w") as f:
        json.dump(r, f, indent=2)
    os.replace(RESULTS_LOG + ".tmp", RESULTS_LOG)

def merge_progress(r):
    """Merge lane results into progress.json after each leg (never None-overwrite)."""
    try:
        prog = json.load(open(PROGRESS_FILE))
    except Exception:
        return
    entry = None
    for m in prog["models"]:
        if m["name"] == ENTRY_NAME:
            entry = m
            break
    if entry is None:
        entry = {"name": ENTRY_NAME}
        prog["models"].append(entry)
    entry["gpu"] = "V100"
    entry["engine"] = XING4
    entry["file"] = MODEL
    entry["thinking"] = False
    entry["serve_flags"] = " ".join(BEST_EXTRA)
    if r.get("speed", {}).get("decode_tps"):
        entry["decode_tps"] = r["speed"]["decode_tps"]
    if "sanity" in r and r["sanity"].get("sanity_pct") is not None:
        entry.setdefault("benchkit", {})["sanity"] = r["sanity"]
    lcb = r.get("lcb") or {}
    if lcb.get("pass_at_1") is not None:
        entry["livecodebench"] = {"pass_at_1": lcb["pass_at_1"],
                                  "wall_time_s": lcb.get("wall_time_s"),
                                  "empty": lcb.get("empty"), "of": lcb.get("of")}
    elif "lcb" in r and lcb.get("error"):
        entry.setdefault("failures", []).append(
            {"benchmark": "livecodebench", "attempt": 1,
             "error": str(lcb["error"])[:400],
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    tau = r.get("tau2") or {}
    if tau.get("reward") is not None:
        entry["tau2"] = {"reward": tau["reward"],
                         "wall_time_s": tau.get("wall_time_s"),
                         "user_sim": tau.get("user_sim"), "canonical": True}
    elif "tau2" in r and tau.get("error"):
        entry.setdefault("failures", []).append(
            {"benchmark": "tau2", "attempt": 1,
             "error": str(tau["error"])[:400],
             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
    entry["status"] = r.get("status", "running")
    entry["end_time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(PROGRESS_FILE + ".tmp", "w") as f:
        json.dump(prog, f, indent=1)
    os.replace(PROGRESS_FILE + ".tmp", PROGRESS_FILE)

def serve(extra, tag, ctx=32768, parallel=1, timeout_s=2400):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(LOGS, f"{tag}_server.log"), "w")
    cmd = [XING4, "--model", MODEL, "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(PORT),
           "--ctx-size", str(ctx), "--batch-size", "2048", "--ubatch-size", "512",
           "--threads", "8", "--threads-batch", "8",
           "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
           "--parallel", str(parallel), "--temp", "0.0", "-n", "4096",
           "--jinja"] + list(extra)
    log(f"serve {tag}: ctx={ctx} parallel={parallel} extra={' '.join(extra)}")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        if proc.poll() is not None:
            tail = open(logf.name).read()[-600:]
            logf.close()
            raise RuntimeError(f"{tag} died:\n{tail}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
            if "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0" not in props:
                proc.kill(); logf.close()
                raise RuntimeError(f"{tag} identity mismatch")
            log(f"  {tag} up (V100:{PORT})")
            return proc, logf
        except RuntimeError:
            raise
        except Exception:
            pass
    proc.kill(); logf.close()
    raise RuntimeError(f"{tag} timeout")

def stop(proc, logf):
    if not proc:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    if logf:
        logf.close()
    time.sleep(5)

def speed_probe():
    best = 0.0
    last = None
    for _ in range(2):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
            payload = {"messages": [{"role": "user", "content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens": 256, "temperature": 0.0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
            t0 = time.time()
            conn.request("POST", "/v1/chat/completions", json.dumps(payload),
                         {"Content-Type": "application/json"})
            d = json.loads(conn.getresponse().read())
            dt = time.time() - t0
            toks = (d.get("usage") or {}).get("completion_tokens")
            conn.close()
            if toks:
                best = max(best, toks / dt)
        except Exception as e:
            last = str(e)
    return {"decode_tps": round(best, 2)} if best else {"error": last or "no tokens"}

def sanity(name, binary, model, gpu, extra):
    gate = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
    glog = os.path.join(LOGS, f"{re.sub(r'[^A-Za-z0-9]+', '_', name)}_gate.log")
    cmd = [BENCH_PY, gate, "--model-file", model, "--name", name,
           "--gpu", str(gpu), "--benchmarks", "sanity:25", "--binary", binary]
    if extra:
        cmd += ["--args", " ".join(extra) if isinstance(extra, list) else extra]
    with open(glog, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=7200)
    txt = open(glog).read()
    m = re.search(r"sanity:25:\s+([\d.]+)%\s+\(?(\d+)/(\d+)", txt)
    if m:
        return {"sanity_pct": float(m.group(1)),
                "passed": int(m.group(2)), "total": int(m.group(3))}
    return {"error": txt[-300:]}

def lcb(name, lcb_model, port, thinking_off=True):
    lenv = dict(os.environ)
    lenv.update({"OPENAI_KEY": "none",
                 "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                 "HF_ALLOW_CODE_EVAL": "1"})
    if thinking_off:
        lenv["LCB_DISABLE_THINKING"] = "1"
    out_dir = os.path.join(LCB_DIR, "output", name)
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir])
    t0 = time.time()
    with open(os.path.join(LOGS, f"lcb_{re.sub(r'[^A-Za-z0-9]+', '_', lcb_model)}.log"), "w") as lf:
        subprocess.run([BENCH_PY, "-m", "lcb_runner.runner.main",
                        "--model", lcb_model, "--scenario", "codegeneration",
                        "--release_version", "release_latest", "--n", "1",
                        "--temperature", "0.0", "--max_tokens", "4096",
                        "--num_problems", "75", "--openai_timeout", "300",
                        "--evaluate"],
                       stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR, env=lenv,
                       timeout=24 * 3600)
    wall = round(time.time() - t0, 1)
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
    if pass1 is None:
        txt = open(lf.name).read().strip()
        if txt:
            m = re.search(r"^([\d.]+)$", txt.splitlines()[-1].strip())
            if m:
                pass1 = float(m.group(1))
    n = empty = 0
    for f in glob.glob(os.path.join(out_dir, "Scenario.codegeneration_1_*.json")):
        if f.endswith("_all.json") or f.endswith("_eval.json"):
            continue
        try:
            for g in json.load(open(f)):
                n += 1
                if not (g.get("output_list") or [""])[0].strip():
                    empty += 1
        except Exception:
            pass
    return {"pass_at_1": pass1, "empty": empty, "of": n, "wall_time_s": wall}

def start_usersim(port=8081, gpu="1"):
    uenv = dict(os.environ)
    uenv["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ulogf = open(os.path.join(LOGS, "usersim.log"), "w")
    usim = subprocess.Popen([UPSTREAM, "--model", os.path.join(LLMS, USER_SIM_FILE),
                             "--flash-attn", "on", "--host", "127.0.0.1",
                             "--port", str(port), "--gpu-layers", "99",
                             "--ctx-size", "131072", "--ubatch-size", "512",
                             "--threads", "4", "--threads-batch", "4",
                             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                             "--parallel", "2", "--temp", "0.0", "-n", "4096"],
                            env=uenv, stdout=ulogf, stderr=subprocess.STDOUT, text=True)
    for _ in range(150):
        time.sleep(2)
        if usim.poll() is not None:
            raise RuntimeError("user sim died: " + open(ulogf.name).read()[-300:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            log("  user sim up (3060:8081)")
            return usim, ulogf
        except Exception:
            pass
    usim.kill(); ulogf.close()
    raise RuntimeError("user sim timeout")

def tau2(name, agent_port, usersim_port=8081):
    safe = re.sub(r"[/()\[\]]", "", name.replace(" ", "_"))
    sdir = os.path.join(TAU2_DIR, "data", "simulations", f"tau2_{safe}")
    if os.path.isdir(sdir):
        subprocess.run(["rm", "-rf", sdir])
    cmd = ["uv", "run", "tau2", "run", "--domain", "airline",
           "--agent-llm", f"openai/{safe}",
           "--agent-llm-args", json.dumps({"api_key": "none",
               "api_base": f"http://127.0.0.1:{agent_port}/v1",
               "temperature": 0.0}),
           "--user-llm", f"openai/{USER_SIM_NAME}",
           "--user-llm-args", json.dumps({"api_key": "none",
               "api_base": f"http://127.0.0.1:{usersim_port}/v1"}),
           "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
           "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
           "--seed", "42", "--save-to", f"tau2_{safe}"]
    tenv = dict(os.environ)
    tenv["OPENAI_API_KEY"] = "none"
    t0 = time.time()
    with open(os.path.join(LOGS, f"tau2_{re.sub(r'[^A-Za-z0-9]+', '_', safe)}.log"), "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=86400,
                       env=tenv, cwd=TAU2_DIR)
    elapsed = round(time.time() - t0, 1)
    d = json.load(open(os.path.join(sdir, "results.json")))
    per_task = [(s.get("reward_info") or {}).get("reward") for s in d["simulations"]]
    scored = [r for r in per_task if r is not None]
    return {"reward": sum(r or 0 for r in per_task) / len(per_task),
            "wall_time_s": elapsed, "user_sim": USER_SIM_NAME, "canonical": True,
            "sims_scored": f"{len(scored)}/15",
            "passes": sum(1 for r in scored if r == 1.0),
            "infra_errors": sum(1 for s in d["simulations"]
                                if s.get("termination_reason") == "infrastructure_error")}

def main():
    r = {"engine": XING4, "gpu": "V100",
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    v100_stopped = []
    try:
        for svc in V100_PRODS:
            sudo("systemctl", "stop", svc)
            v100_stopped.append(svc)
        time.sleep(10)
        subprocess.run(["pkill", "-f", "BTL-4"], capture_output=True)
        subprocess.run(["pkill", "-f", "LFM2.5"], capture_output=True)
        time.sleep(3)

        # --- speed (already probed; re-confirm at lane ctx) ---
        try:
            proc, logf = serve(BEST_EXTRA, "speed", ctx=8192)
            try:
                r["speed"] = speed_probe()
                log(f"speed: {r['speed']}")
            finally:
                stop(proc, logf)
        except Exception as e:
            r["speed"] = {"error": str(e)[:300]}
            log(f"speed FAILED: {str(e)[:200]}")
        r["status"] = "speed done"
        save_results(r); merge_progress(r)

        # --- sanity:25 ---
        try:
            r["sanity"] = sanity(ENTRY_NAME, XING4, MODEL, 0, BEST_EXTRA)
            log(f"sanity: {r['sanity']}")
        except Exception as e:
            r["sanity"] = {"error": str(e)[:300]}
            log(f"sanity FAILED: {str(e)[:200]}")
        r["status"] = "sanity done"
        save_results(r); merge_progress(r)

        # --- LCB 75, thinking off ---
        try:
            proc, logf = serve(BEST_EXTRA, "lcb", ctx=32768)
            try:
                r["lcb"] = lcb(ENTRY_NAME, "local/flashnext-qwen38-q20", PORT)
                log(f"lcb: {r['lcb']}")
            finally:
                stop(proc, logf)
        except Exception as e:
            r["lcb"] = {"error": str(e)[:300]}
            log(f"lcb FAILED: {str(e)[:200]}")
        r["status"] = "lcb done"
        save_results(r); merge_progress(r)

        # --- tau2 (agent V100, user sim 3060; stop gemma for the leg) ---
        usim = ulogf = None
        g30_stopped = False
        try:
            sudo("systemctl", "stop", G30_PROD)
            g30_stopped = True
            time.sleep(8)
            usim, ulogf = start_usersim(gpu="1")
            proc, logf = serve(BEST_EXTRA, "tau2", ctx=32768, parallel=2)
            try:
                conn = http.client.HTTPConnection("127.0.0.1", 8081, timeout=120)
                conn.request("POST", "/v1/chat/completions", json.dumps(
                    {"messages": [{"role": "user", "content": "hi"}],
                     "max_tokens": 8, "temperature": 0.0}),
                    {"Content-Type": "application/json"})
                conn.getresponse().read(); conn.close()
                r["tau2"] = tau2(ENTRY_NAME, PORT)
                log(f"tau2: {r['tau2']}")
            finally:
                stop(proc, logf)
        except Exception as e:
            r["tau2"] = {"error": str(e)[:300]}
            log(f"tau2 FAILED: {str(e)[:200]}")
        finally:
            if usim:
                usim.terminate()
                try:
                    usim.wait(timeout=10)
                except Exception:
                    usim.kill()
            if ulogf:
                ulogf.close()
            if g30_stopped:
                sudo("systemctl", "start", G30_PROD)
                time.sleep(5)
                log(f"prod {G30_PROD}: {sudo('systemctl','is-active',G30_PROD).stdout.strip()}")
        r["status"] = "completed"
        save_results(r); merge_progress(r)
    finally:
        for svc in v100_stopped:
            sudo("systemctl", "start", svc)
        time.sleep(5)
        for svc in v100_stopped:
            log(f"prod {svc}: {sudo('systemctl','is-active',svc).stdout.strip()}")

    # regenerate report
    try:
        subprocess.run([BENCH_PY, "/home/caimlas/llm-benchmarks/scripts/generate_report.py"],
                       capture_output=True, text=True, timeout=300)
        log("report regenerated")
    except Exception as e:
        log(f"report regen failed: {e}")
    log("LANE DONE")

if __name__ == "__main__":
    main()
