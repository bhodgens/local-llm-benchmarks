#!/usr/bin/env python3
"""
bench-0918 staged batch executor (2026-10-06).

Execution order per QUEUE.md:
  1. CRACK PQ2_0 (V100)          - full lane
  2. Bonsai2 official PQ2_0 (V100) - full lane
  3. F arms: PTQ1_0 baseline + PTQ1_0-mtp n-max 1 (V100) - speed + sanity + LCB
  4. Xing4.0-29B IQ4_NL (V100)   - full lane, then MTP arm if head present
  5. Xing4.0 3060 tok/s probe    - tok/s only (weights 20.1G > 12G VRAM -> cpu-moe)
  6. K2-Horizon-7B-Uno Q4_K_M (3060) - full lane, 128K-ctx fit check first
  7. C gate decision: 1bit CRACK benched on 3060 only if CRACK PQ2_0 vs official
     shows no substantial decay (recorded in results; bench only if triggered)

Prod handling: caimlas-nail + caimlas-nail-v100 stopped ONCE at batch start
(user-directed: unload services from GPUs for the batch), restarted in final
finally. V100 = GPU0, 3060 = GPU1.

Engines:
  PrismML prebuilt  : /var/tmp/llms/bench-0918/prismml-bin/llama-prism-b10685-7dffb15/llama-server  (A,B,C)
  sudoingx fork     : ~/git/llama.cpp-sudoingx/build/bin/llama-server   (F: PTQ1_0 + mtp + kernel arm)
  xing4 fork        : ~/git/llama.cpp-xing4/build/bin/llama-server      (D)
  k2horizon fork    : ~/git/llama.cpp-k2horizon/build/bin/llama-server  (E)
"""
import subprocess, json, time, os, re, urllib.request, glob, sys, http.client

STAGE = "/var/tmp/llms/bench-0918"
PRISMML = f"{STAGE}/prismml-bin/llama-prism-b10685-7dffb15/llama-server"
SUDOINGX = "/home/caimlas/git/llama.cpp-sudoingx/build/bin/llama-server"
XING4 = "/home/caimlas/git/llama.cpp-xing4/build/bin/llama-server"
K2FH = "/home/caimlas/git/llama.cpp-k2horizon/build/bin/llama-server"
UPSTREAM = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs/bench0918"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
RESULTS_LOG = "/tmp/coding-bench/bench0918_results.json"

USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"

V100_PROD = "caimlas-nail-v100"
G30_PROD = "caimlas-nail"

os.makedirs(LOGS, exist_ok=True)

RESULTS: dict = {}


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def save_results():
    with open(RESULTS_LOG + ".tmp", "w") as f:
        json.dump(RESULTS, f, indent=2)
    os.replace(RESULTS_LOG + ".tmp", RESULTS_LOG)


def sudo(*args):
    return subprocess.run(["sudo", "-n"] + list(args), capture_output=True, text=True)


def stop_prods():
    for svc in (V100_PROD, G30_PROD):
        sudo("systemctl", "stop", svc)
    time.sleep(8)
    # kill any stragglers
    subprocess.run(["pkill", "-f", "Nail-Qwen3.6"], capture_output=True)
    time.sleep(3)
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    log(f"GPU compute apps after prod stop: {out.strip() or '(none)'}")


def start_prods():
    for svc in (G30_PROD, V100_PROD):
        sudo("systemctl", "start", svc)
    log("prod services restarted")


def serve(binary, model, extra, gpu, tag, port, ctx=32768, parallel=2,
          kv="q8_0", threads=8):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(os.path.join(LOGS, f"{tag}_server.log"), "w")
    cmd = [binary, "--model", model, "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(port),
           "--gpu-layers", "99", "--ctx-size", str(ctx),
           "--batch-size", "2048", "--ubatch-size", "512",
           "--threads", str(threads), "--threads-batch", str(threads),
           "--cache-type-k", kv, "--cache-type-v", kv,
           "--parallel", str(parallel), "--temp", "0.0", "-n", "4096",
           "--jinja"] + list(extra)
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                            text=True)
    for _ in range(300):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"{tag} died: " + open(logf.name).read()[-400:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            props = urllib.request.urlopen(f"http://127.0.0.1:{port}/props",
                                           timeout=5).read().decode()
            base = os.path.basename(model)
            key = base.split("-0000")[0]
            if key not in props:
                proc.kill(); logf.close()
                raise RuntimeError(f"{tag} identity mismatch: props lacks {key}")
            log(f"  {tag} up (GPU{gpu}:{port})")
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
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    if logf:
        logf.close()
    time.sleep(3)


def speed_probe(port, thinking_off=True):
    best = 0.0
    last = None
    for _ in range(2):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
            payload = {"messages": [{"role": "user", "content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens": 256, "temperature": 0.0, "stream": False}
            if thinking_off:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            t0 = time.time()
            conn.request("POST", "/v1/chat/completions",
                         json.dumps(payload), {"Content-Type": "application/json"})
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
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
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
                       timeout=6 * 3600)
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
    # fallback: score printed at end of runner log (eval crashes can precede json)
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
        for g in json.load(open(f)):
            n += 1
            if not (g.get("output_list") or [""])[0].strip():
                empty += 1
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
                            env=uenv, stdout=ulogf, stderr=subprocess.STDOUT,
                            text=True)
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


def full_lane(name, lcb_model, binary, model, extra, gpu, port, thinking_off=True,
              ctx=32768, usersim_gpu=None):
    """speed -> sanity -> LCB -> tau2. Caller handles prod stop/start."""
    r = {"engine": os.path.dirname(binary), "gpu": f"GPU{gpu}", "ts":
         time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    usim = ulogf = None
    proc = logf = None
    try:
        # speed (agent server)
        proc, logf = serve(binary, model, extra, gpu, f"{name}_speed", port, ctx=ctx)
        r["speed"] = speed_probe(port, thinking_off)
        log(f"  speed: {r['speed']}")
        stop(proc, logf); proc = logf = None

        # sanity
        r["sanity"] = sanity(name, binary, model, gpu, extra)
        log(f"  sanity: {r['sanity']}")

        # LCB (agent server again)
        proc, logf = serve(binary, model, extra, gpu, f"{name}_lcb", port, ctx=ctx)
        r["lcb"] = lcb(name, lcb_model, port, thinking_off)
        log(f"  lcb: {r['lcb']}")
        stop(proc, logf); proc = logf = None

        # tau2 (user sim on the other GPU)
        if usersim_gpu is not None:
            usim, ulogf = start_usersim(gpu=usersim_gpu)
        proc, logf = serve(binary, model, extra, gpu, f"{name}_tau2", port, ctx=ctx)
        # verify user sim answers before firing tasks (PITFALL: all-15 connection fail)
        conn = http.client.HTTPConnection("127.0.0.1", 8081, timeout=120)
        conn.request("POST", "/v1/chat/completions", json.dumps(
            {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
             "temperature": 0.0}), {"Content-Type": "application/json"})
        conn.getresponse().read(); conn.close()
        r["tau2"] = tau2(name, port)
        log(f"  tau2: {r['tau2']}")
    finally:
        stop(proc, logf)
        if usim:
            usim.terminate()
            try:
                usim.wait(timeout=10)
            except Exception:
                usim.kill()
            if ulogf:
                ulogf.close()
    return r


# ---------------------------------------------------------------- lanes

def lane_crack():
    name = "Bonsai-2-27B Ternary CRACK PQ2_0 (V100)"
    return full_lane(name, "local/bonsai2-crack-pq20-v100", PRISMML,
                     f"{STAGE}/crack/Bonsai-2-27B-PQ2_0-CRACK.gguf",
                     ["-fit", "off"], 0, 18096, usersim_gpu="1")


def lane_bonsai2():
    name = "Ternary-Bonsai-2-27B PQ2_0 (V100)"
    return full_lane(name, "local/bonsai2-27b-pq20-v100", PRISMML,
                     f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf",
                     ["-fit", "off"], 0, 18096, usersim_gpu="1")


def lane_ptq10_baseline():
    name = "Bonsai-2-27B PTQ1_0 (V100)"
    return full_lane(name, "local/bonsai2-crack-1bit", SUDOINGX,
                     f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
                     [], 0, 18096, usersim_gpu="1")


def lane_ptq10_mtp():
    name = "Bonsai-2-27B PTQ1_0 + MTP n1 (V100)"
    r = {"engine": SUDOINGX, "gpu": "GPU0",
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    extra = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1",
             "-ctk", "q4_0", "-ctv", "q4_0", "-c", "131072"]
    proc, logf = serve(SUDOINGX,
                       f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf",
                       extra, 0, "ptq10_mtp", 18096)
    try:
        r["speed"] = speed_probe(18096)
        r["sanity"] = sanity(name, SUDOINGX,
                             f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf",
                             0, " ".join(extra))
        r["lcb"] = lcb(name, "local/bonsai2-crack-1bit", 18096)
        r["note"] = "speed arm: MTP n-max 1 per F-arm economics; identity caveat (batch invariance) applies"
    finally:
        stop(proc, logf)
    return r


def lane_xing_v100():
    name = "Xing4.0-29B-A4B IQ4_NL (V100)"
    return full_lane(name, "local/xing40-29b-iq4nl-v100", XING4,
                     f"{STAGE}/xing/xing4_0-29b-IQ4_NL-00001-of-00003.gguf",
                     [], 0, 18096, usersim_gpu="1")


def lane_xing_3060():
    """tok/s probe only per user directive. 20.1G weights > 12G VRAM -> cpu-moe."""
    name = "Xing4.0-29B-A4B IQ4_NL (3060)"
    r = {"engine": XING4, "gpu": "GPU1", "probe_only": True,
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    for label, extra in [
        ("cpu-moe", ["--cpu-moe", "--ctx-size", "8192", "-ctk", "q4_0", "-ctv", "q4_0"]),
        ("cpu-moe-mtp", ["--cpu-moe", "--ctx-size", "8192", "-ctk", "q4_0", "-ctv", "q4_0",
                         "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]),
    ]:
        try:
            proc, logf = serve(XING4,
                               f"{STAGE}/xing/xing4_0-29b-IQ4_NL-00001-of-00003.gguf",
                               extra, 1, f"xing3060_{label}", 18097, parallel=1,
                               threads=6)
            r[label] = speed_probe(18097)
            stop(proc, logf)
            log(f"  xing 3060 {label}: {r[label]}")
        except Exception as e:
            r[label] = {"error": str(e)[:300]}
            log(f"  xing 3060 {label}: ERROR {str(e)[:200]}")
    return r


def lane_k2_uno():
    """E: 3060 full lane. User sim moved to V100 (k2 needs the whole 3060:
    fit check showed even 32K beside the LFM sim OOMs on 12G)."""
    name = "K2-Horizon-7B-Uno Q4_K_M (3060)"
    model = f"{STAGE}/k2-uno-q4km/K2-Horizon-7B-Uno-Q4_K_M.gguf"
    r = {"engine": K2FH, "gpu": "GPU1", "user_sim_gpu": "V100",
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    usim = ulogf = None
    proc = logf = None
    try:
        usim, ulogf = start_usersim(gpu="0")
        # fit check on the FULL 3060: try 128K, fall back 64K, 32K
        chosen = None
        for ctx in (131072, 65536, 32768):
            try:
                proc, logf = serve(K2FH, model, [], 1, "k2_fit", 18097,
                                   ctx=ctx, parallel=2, kv="q8_0", threads=4)
                chosen = ctx
                break
            except RuntimeError as e:
                if "died" in str(e) or "timeout" in str(e):
                    log(f"  k2 ctx {ctx} no fit, stepping down")
                    continue
                raise
        if chosen is None:
            r["error"] = "no ctx size fits 3060 beside user sim"
            return r
        r["ctx"] = chosen
        r["speed"] = speed_probe(18097)
        log(f"  k2 speed @ctx{chosen}: {r['speed']}")
        stop(proc, logf); proc = logf = None

        r["sanity"] = sanity(name, K2FH, model, 1, "")
        log(f"  k2 sanity: {r['sanity']}")

        proc, logf = serve(K2FH, model, [], 1, "k2_lcb", 18097,
                           ctx=chosen, parallel=2, kv="q8_0", threads=4)
        r["lcb"] = lcb(name, "local/k2-horizon-7b-uno-q4km", 18097)
        log(f"  k2 lcb: {r['lcb']}")
        stop(proc, logf); proc = logf = None

        # tau2: agent on 3060:18097, user sim ALREADY on 3060:8081 (small model coexists)
        proc, logf = serve(K2FH, model, [], 1, "k2_tau2", 18097,
                           ctx=chosen, parallel=2, kv="q8_0", threads=4)
        r["tau2"] = tau2(name, 18097)
        log(f"  k2 tau2: {r['tau2']}")
    except Exception as e:
        r["error"] = str(e)[:400]
    finally:
        stop(proc, logf)
        if usim:
            usim.terminate()
            try:
                usim.wait(timeout=10)
            except Exception:
                usim.kill()
            if ulogf:
                ulogf.close()
    return r


def lane_crack1bit_3060():
    """C gate: 1bit CRACK on 3060 (PTQ1_0-class fits: gen-1 Q1_0 was 9.7G @262K)."""
    name = "Bonsai-2-27B 1bit CRACK PTQ1_0 (3060)"
    model = f"{STAGE}/crack1bit/Bonsai-2-27B-PTQ1_0-CRACK.gguf"
    r = {"engine": PRISMML, "gpu": "GPU1",
         "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    usim = ulogf = None
    proc = logf = None
    try:
        usim, ulogf = start_usersim()
        proc, logf = serve(PRISMML, model, ["-fit", "off"], 1, "crack1bit", 18097,
                           ctx=32768, parallel=2, kv="q4_0", threads=6)
        r["speed"] = speed_probe(18097)
        log(f"  crack1bit speed: {r['speed']}")
        stop(proc, logf); proc = logf = None
        r["sanity"] = sanity(name, PRISMML, model, 1, "-fit off")
        proc, logf = serve(PRISMML, model, ["-fit", "off"], 1, "crack1bit_lcb",
                           18097, ctx=32768, parallel=2, kv="q4_0", threads=6)
        r["lcb"] = lcb(name, "local/bonsai2-crack-1bit", 18097)
        log(f"  crack1bit lcb: {r['lcb']}")
        stop(proc, logf); proc = logf = None
        proc, logf = serve(PRISMML, model, ["-fit", "off"], 1, "crack1bit_tau2",
                           18097, ctx=32768, parallel=2, kv="q4_0", threads=6)
        r["tau2"] = tau2(name, 18097)
        log(f"  crack1bit tau2: {r['tau2']}")
    except Exception as e:
        r["error"] = str(e)[:400]
    finally:
        stop(proc, logf)
        if usim:
            usim.terminate()
            try:
                usim.wait(timeout=10)
            except Exception:
                usim.kill()
            if ulogf:
                ulogf.close()
    return r


# ---------------------------------------------------------------- driver

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    stages = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    log("stopping prod services (user-directed batch)")
    stop_prods()
    try:
        steps = [
            ("crack", lane_crack),
            ("bonsai2", lane_bonsai2),
            ("ptq10_baseline", lane_ptq10_baseline),
            ("ptq10_mtp", lane_ptq10_mtp),
            ("xing_v100", lane_xing_v100),
            ("xing_3060", lane_xing_3060),
            ("k2_uno", lane_k2_uno),
        ]
        if stages:
            steps = [(k, f) for k, f in steps if k in stages]
        for key, fn in steps:
            if only and key != only:
                continue
            log(f"=== LANE {key} ===")
            try:
                RESULTS[key] = fn()
            except Exception as e:
                RESULTS[key] = {"error": str(e)[:500]}
                log(f"  LANE {key} ERROR: {str(e)[:300]}")
            save_results()

        # C gate: substantial-decay verdict from crack vs official + PTQ1_0
        try:
            crack = RESULTS.get("crack", {})
            off = RESULTS.get("bonsai2", {})
            lcb_c = crack.get("lcb", {}).get("pass_at_1")
            lcb_o = off.get("lcb", {}).get("pass_at_1")
            tau_c = crack.get("tau2", {}).get("reward")
            tau_o = off.get("tau2", {}).get("reward")
            decay = (lcb_o is not None and lcb_c is not None and
                     (lcb_o - lcb_c) > 0.10) or \
                    (tau_o is not None and tau_c is not None and
                     (tau_o - tau_c) > 0.10)
            RESULTS["c_gate"] = {
                "crack_lcb": lcb_c, "official_lcb": lcb_o,
                "crack_tau2": tau_c, "official_tau2": tau_o,
                "substantial_decay": bool(decay)}
            save_results()
            if only not in ("crack", "bonsai2") and not decay:
                if RESULTS.get("ptq10_baseline", {}).get("lcb", {}).get("pass_at_1") is not None:
                    log("C gate: no substantial decay -> benching 1bit CRACK on 3060")
                    RESULTS["crack1bit_3060"] = lane_crack1bit_3060()
                    save_results()
            else:
                log(f"C gate: decay={decay} -> 1bit CRACK skipped")
        except Exception as e:
            RESULTS["c_gate_error"] = str(e)[:300]
            save_results()
    finally:
        start_prods()
        log("batch complete, results in " + RESULTS_LOG)
        print(json.dumps(RESULTS, indent=2, default=str))


if __name__ == "__main__":
    main()
