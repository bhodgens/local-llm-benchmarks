#!/usr/bin/env python3
"""
bench-0918 thinking-ON rerun (2026-10-07).

Lanes: BTL-4 Q4_K_M, Muse-Glimmer-30B, gemma-4-12B-it-QAT, Bonsai2 PQ2_0,
CRACK PQ2_0, PTQ1_0, Xing4.0-29B IQ4_NL — all with thinking ENABLED.

Template audit results:
  BTL-4 / Bonsai2 / CRACK / PTQ1_0 / Xing4.0 : qwen3-style, enable_thinking kwarg -> toggleable
  gemma-4-12B                                 : template has enable_thinking kwarg -> toggleable
  Muse-Glimmer                                : harmony/gpt-oss channels, reasoning ALWAYS on,
                                                no kwarg -> benchmarked as-is (reasoning extracted)

Thinking control: LCB env LCB_DISABLE_THINKING is simply NOT set; speed probes omit
chat_template_kwargs. Server-side: no --reasoning off; --reasoning-format deepseek keeps
<think> out of content so LCB's answer extraction sees clean text.

Prod handling: current prods (caimlas-btl4-v100, caimlas-lfm25-v100, caimlas-gemma4-3060)
stopped at start, restarted in finally. User sim (LFM2.5) runs on whichever GPU the lane
isn't using.

max_tokens for LCB raised 4096 -> 12288 (thinking consumes budget; off-runs used 4096).
"""
import json, os, re, subprocess, sys, time, glob, http.client, urllib.request

BENCH_PY  = "/home/caimlas/bench-venv/bin/python"
LCB_DIR   = "/home/caimlas/git/LiveCodeBench"
STAGE     = "/var/tmp/llms/bench-0918"
LLMS      = "/home/files/llms"
LOGS      = "/tmp/coding-bench/logs/bench0918_think"
RESULTS   = "/tmp/coding-bench/bench0918_think_results.json"
TAU2_DIR  = "/home/caimlas/git/tau2-bench"

PRISMML   = f"{STAGE}/prismml-bin/llama-prism-b10685-7dffb15/llama-server"
SUDOINGX  = "/home/caimlas/git/llama.cpp-sudoingx/build/bin/llama-server"
XING4     = "/home/caimlas/git/llama.cpp-xing4/build/bin/llama-server"
STOCK     = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
UPSTREAM  = STOCK

USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"
USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"

PRODS = ["caimlas-btl4-v100", "caimlas-lfm25-v100", "caimlas-gemma4-3060"]

os.makedirs(LOGS, exist_ok=True)

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "batch.log"), "a") as f:
        f.write(line + "\n")

def sudo(*args):
    return subprocess.run(["sudo", "-n"] + list(args), capture_output=True, text=True)

def stop_prods():
    for svc in PRODS:
        sudo("systemctl", "stop", svc)
    time.sleep(8)
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout
    log(f"GPU compute apps after prod stop: {out.strip() or '(none)'}")

def start_prods():
    # big alloc first (btl4), then lfm (ordered start; gemma independent)
    for svc in ("caimlas-btl4-v100", "caimlas-lfm25-v100", "caimlas-gemma4-3060"):
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
           "--parallel", str(parallel), "--temp", "0.0", "-n", "12288",
           "--jinja", "--reasoning-format", "deepseek"] + list(extra)
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(300):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"{tag} died: " + open(logf.name).read()[-400:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            props = urllib.request.urlopen(f"http://127.0.0.1:{port}/props", timeout=5).read().decode()
            key = os.path.basename(model).split("-0000")[0]
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
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    if logf:
        logf.close()

def start_usersim(port=8081, gpu="1"):
    uenv = dict(os.environ)
    uenv["CUDA_VISIBLE_DEVICES"] = str(gpu)
    ulogf = open(os.path.join(LOGS, "usersim.log"), "w")
    usim = subprocess.Popen([UPSTREAM, "--model", os.path.join(LLMS, USER_SIM_FILE),
                             "--flash-attn", "on", "--host", "127.0.0.1",
                             "--port", str(port), "--gpu-layers", "99",
                             "--ctx-size", "131072", "--ubatch-size", "512",
                             "--threads", "4", "--threads-batch", "4",
                             "--cache-type-k", "q8_0", "--jinja",
                             "--temp", "0.0", "-n", "4096"],
                            env=uenv, stdout=ulogf, stderr=subprocess.STDOUT, text=True)
    for _ in range(150):
        time.sleep(2)
        if usim.poll() is not None:
            ulogf.close()
            raise RuntimeError("user sim died: " + open(ulogf.name).read()[-300:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            log(f"  user sim up (GPU{gpu}:{port})")
            return usim, ulogf
        except Exception:
            pass
    usim.kill(); ulogf.close()
    raise RuntimeError("user sim timeout")

def speed_probe(port):
    """Thinking-ON decode t/s. Single-stream. No enable_thinking override."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        payload = {"messages": [{"role": "user", "content":
            "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
            "max_tokens": 1024, "temperature": 0.0, "stream": False}
        t0 = time.time()
        conn.request("POST", "/v1/chat/completions", json.dumps(payload),
                     {"Content-Type": "application/json"})
        d = json.loads(conn.getresponse().read())
        dt = time.time() - t0
        m = d["choices"][0]["message"]
        toks = (d.get("usage") or {}).get("completion_tokens") or 0
        rtoks = len(m.get("reasoning_content") or "") // 4
        conn.close()
        if toks:
            return {"decode_tps": round(toks / dt, 2),
                    "reasoning_est_tokens": rtoks,
                    "reasoning_share_pct": round(100 * rtoks / max(toks, 1), 1)}
        return {"error": "no tokens"}
    except Exception as e:
        return {"error": str(e)[:300]}

def sanity(name, binary, model, gpu, extra):
    """25-problem benchkit gate (its own server). Thinking-ON: pass --jinja +
    --reasoning-format deepseek so <think> is extracted to reasoning_content and
    BenchKit sees clean content."""
    gate = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
    glog = os.path.join(LOGS, f"{re.sub(r'[^A-Za-z0-9]+', '_', name)}_gate.log")
    jargs = list(extra) + ["--jinja", "--reasoning-format", "deepseek"]
    cmd = [BENCH_PY, gate, "--model-file", model, "--name", name + " think",
           "--gpu", str(gpu), "--benchmarks", "sanity:25",
           "--binary", binary, "--args", " ".join(f"'{a}'" if " " in a else a for a in jargs)]
    with open(glog, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    txt = open(glog).read()
    m = re.search(r"sanity:25:\s*([0-9.]+)%\s*\((\d+)/(\d+)\)", txt)
    if m:
        return round(int(m.group(2)) / int(m.group(3)), 2)
    return None

def lcb(name, lcb_model, port, max_tokens=12288):
    out_name = DISPLAY.get(lcb_model, name)
    """Thinking-ON LCB: no LCB_DISABLE_THINKING. Bigger budget for reasoning."""
    lenv = dict(os.environ)
    lenv.update({"OPENAI_KEY": "none",
                 "OPENAI_BASE_URL": f"http://127.0.0.1:{port}/v1",
                 "HF_ALLOW_CODE_EVAL": "1"})
    # LCB_DISABLE_THINKING deliberately NOT set
    out_dir = os.path.join(LCB_DIR, "output", out_name)
    if os.path.isdir(out_dir):
        subprocess.run(["rm", "-rf", out_dir])
    t0 = time.time()
    llog = os.path.join(LOGS, f"lcb_{re.sub(r'[^A-Za-z0-9]+','_',lcb_model)}.log")
    with open(llog, "w") as lf:
        subprocess.run([BENCH_PY, "-m", "lcb_runner.runner.main",
                        "--model", lcb_model, "--scenario", "codegeneration",
                        "--release_version", "release_latest", "--n", "1",
                        "--temperature", "0.0", "--max_tokens", str(max_tokens),
                        "--num_problems", "75", "--openai_timeout", "600",
                        "--evaluate"],
                       stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR, env=lenv,
                       timeout=8 * 3600)
    wall = round(time.time() - t0, 1)
    pass1 = None
    empty = 0
    for f in glob.glob(os.path.join(out_dir, "*_eval.json")):
        if f.endswith("_all.json"):
            continue
        try:
            data = json.load(open(f))
            if isinstance(data, list) and data:
                pass1 = data[0].get("pass@1")
                empty = sum(1 for row in data
                            if not (row.get("code_list") or [None])[0])
        except Exception:
            pass
    if pass1 is None:
        txt = open(llog).read().strip()
        if txt:
            m = re.findall(r"pass@1[:= ]+([0-9.]+)", txt[-3000:]) or re.findall(r"^([01]\.\d+)$", txt[-200:], re.M)
            if m:
                pass1 = float(m[-1])
    return {"pass@1": pass1, "empty_outputs": empty, "wall_s": wall} if pass1 is not None \
        else {"pass@1": None, "wall_s": wall, "log": llog}

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
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"
    env["NO_PROXY"] = "*"; env["no_proxy"] = "*"
    tlog = os.path.join(LOGS, f"tau2_{re.sub(r'[^A-Za-z0-9]+', '_', safe)}.log")
    try:
        with open(tlog, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=TAU2_DIR, env=env, timeout=8 * 3600)
    except subprocess.TimeoutExpired:
        log("  tau2 timeout")
    return parse_tau2(sdir, tlog)

def parse_tau2(sdir, tlog):
    """Canonical format: results.json['simulations'][i]['reward_info']['reward'],
    infra = termination_reason == 'infrastructure_error'. Denominator always 15."""
    res = {"reward": None, "infra_errors": 0}
    rj = os.path.join(sdir, "results.json")
    if os.path.exists(rj):
        try:
            d = json.load(open(rj))
            sims = d.get("simulations", [])
            per = [(s.get("reward_info") or {}).get("reward") for s in sims]
            scored = [x for x in per if x is not None]
            res = {
                "reward": round(sum(x or 0 for x in per) / 15, 4) if sims else None,
                "sims_scored": f"{len(scored)}/15",
                "passes": sum(1 for x in scored if x == 1.0),
                "infra_errors": sum(1 for s in sims
                                    if s.get("termination_reason") == "infrastructure_error"),
            }
        except Exception as e:
            res["parse_error"] = str(e)[:200]
    else:
        txt = open(tlog).read()[-4000:] if os.path.exists(tlog) else ""
        m = re.findall(r"reward[:= ]+([0-9.]+)", txt)
        if m:
            res["reward"] = round(sum(float(x) for x in m) / len(m), 4)
    return res

def full_lane_think(name, lcb_model, binary, model, extra, gpu, port,
                    ctx=32768, usersim_gpu=None, max_tokens=12288):
    r = {"engine": os.path.dirname(binary), "gpu": f"GPU{gpu}", "thinking": True, "ts":
         time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    usim = ulogf = None
    proc = logf = None
    try:
        proc, logf = serve(binary, model, extra, gpu, f"{name}_speed", port, ctx=ctx)
        r["speed"] = speed_probe(port)
        log(f"  speed: {r['speed']}")
        stop(proc, logf); proc = logf = None

        r["sanity"] = sanity(name, binary, model, gpu, extra)
        log(f"  sanity: {r['sanity']}")

        proc, logf = serve(binary, model, extra, gpu, f"{name}_lcb", port, ctx=ctx)
        r["lcb"] = lcb(name, lcb_model, port, max_tokens)
        log(f"  lcb: {r['lcb']}")
        stop(proc, logf); proc = logf = None

        if usersim_gpu is not None:
            usim, ulogf = start_usersim(gpu=usersim_gpu)
        proc, logf = serve(binary, model, extra, gpu, f"{name}_tau2", port, ctx=ctx)
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

# ------------------------------------------------------------------ lanes

def lane_btl4():
    return full_lane_think(
        "BTL-4 Q4_K_M (think)", "local/btl4-q4km-think", STOCK,
        f"{LLMS}/dogukanurker-btl4-q4km/BTL-4-Q4_K_M.gguf",
        [], 0, 18096, usersim_gpu="1")

def lane_muse():
    return full_lane_think(
        "Muse-Glimmer-30B-UD-Q4_K_XL (think)", "local/muse-glimmer-think", STOCK,
        f"{LLMS}/Muse-Glimmer-30B-UD-Q4_K_XL.gguf",
        [], 0, 18096, usersim_gpu="1")

def lane_gemma():
    return full_lane_think(
        "gemma-4-12B-it-QAT Q4_0 (think)", "local/gemma4-12b-think", STOCK,
        f"{LLMS}/gemma-4-12B-it-QAT-Q4_0.gguf",
        [], 0, 18096, usersim_gpu="1")

def lane_bonsai2():
    return full_lane_think(
        "Ternary-Bonsai-2-27B PQ2_0 (think)", "local/bonsai2-27b-pq20-think", PRISMML,
        f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf",
        ["-fit", "off"], 0, 18096, usersim_gpu="1")

def lane_crack():
    return full_lane_think(
        "Bonsai-2-27B Ternary CRACK PQ2_0 (think)", "local/bonsai2-crack-pq20-think", PRISMML,
        f"{STAGE}/crack/Bonsai-2-27B-PQ2_0-CRACK.gguf",
        ["-fit", "off"], 0, 18096, usersim_gpu="1")

def lane_ptq10():
    return full_lane_think(
        "Bonsai-2-27B PTQ1_0 (think)", "local/bonsai2-crack-1bit-think", SUDOINGX,
        f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
        [], 0, 18096, usersim_gpu="1")

def lane_xing():
    return full_lane_think(
        "Xing4.0-29B-A4B IQ4_NL (think)", "local/xing40-29b-iq4nl-think", XING4,
        f"{STAGE}/xing/xing4_0-29b-IQ4_NL-00001-of-00003.gguf",
        [], 0, 18096, usersim_gpu="1")

LANES = {
    "btl4": lane_btl4,
    "muse": lane_muse,
    "gemma": lane_gemma,
    "bonsai2": lane_bonsai2,
    "crack": lane_crack,
    "ptq10": lane_ptq10,
    "xing": lane_xing,
}

def save(results):
    with open(RESULTS, "w") as f:
        json.dump(results, f, indent=2)

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    lanes = [k for k in LANES if not only or k in only.split(",")]
    log(f"thinking-ON batch: {lanes}")
    results = {}
    if os.path.exists(RESULTS):
        try:
            results = json.load(open(RESULTS))
        except Exception:
            results = {}
    stop_prods()
    try:
        for k in lanes:
            log(f"=== lane {k} ===")
            try:
                results[k] = LANES[k]()
            except Exception as e:
                results[k] = {"error": str(e)[:400]}
                log(f"  lane {k} ERROR: {str(e)[:300]}")
            save(results)
    finally:
        start_prods()
    save(results)
    log("batch complete")

if __name__ == "__main__":
    main()
