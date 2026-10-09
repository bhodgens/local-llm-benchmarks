#!/usr/bin/env python3
"""
bench-0918 think-rerun POST-PASS (2026-10-09).
Fills the gaps the main batch left:
  - 4 sanity gates with vendor-fork-safe args (ctx 32768, -fit off where applicable)
  - 7 tau2 lanes with the canonical flags (--max-concurrency, --save-to)
Prod services stopped at start, restarted in finally. Setsid-safe.
"""
import json, os, re, subprocess, sys, time, http.client, urllib.request

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

USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"
USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
PRODS = ["caimlas-btl4-v100", "caimlas-lfm25-v100", "caimlas-gemma4-3060"]

os.makedirs(LOGS, exist_ok=True)

def log(msg):
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(os.path.join(LOGS, "postpass.log"), "a") as f:
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
    for svc in ("caimlas-btl4-v100", "caimlas-lfm25-v100", "caimlas-gemma4-3060"):
        sudo("systemctl", "start", svc)
    log("prod services restarted")

# ---------- sanity gates (fixed args) ----------
GATE_JOBS = [
    # (lane_key, display_name, binary, model, extra_args)
    ("bonsai2", "Ternary-Bonsai-2-27B PQ2_0 (think)", PRISMML,
     f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf",
     "-fit off --ctx-size 32768"),
    ("crack", "Bonsai-2-27B Ternary CRACK PQ2_0 (think)", PRISMML,
     f"{STAGE}/crack/Bonsai-2-27B-PQ2_0-CRACK.gguf",
     "-fit off --ctx-size 32768"),
    ("xing", "Xing4.0-29B-A4B IQ4_NL (think)", XING4,
     f"{STAGE}/xing/xing4_0-29b-IQ4_NL-00001-of-00003.gguf",
     "--ctx-size 32768"),
    # bonsai2/ptq10 sanity: ptq10 already passed (0.84). btl4/muse/gemma passed in-batch.
]

def run_gate(name, binary, model, extra_args):
    gate = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
    glog = os.path.join(LOGS, f"{re.sub(r'[^A-Za-z0-9]+', '_', name)}_gate2.log")
    args = extra_args + " --jinja --reasoning-format deepseek"
    cmd = [BENCH_PY, gate, "--model-file", model, "--name", name + " think",
           "--gpu", "0", "--benchmarks", "sanity:25",
           "--binary", binary, "--args", args]
    with open(glog, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    txt = open(glog).read()
    m = re.search(r"sanity:25:\s*([0-9.]+)%\s*\((\d+)/(\d+)\)", txt)
    return round(int(m.group(2)) / int(m.group(3)), 2) if m else None

# ---------- tau2 (canonical flags) ----------
TAU2_JOBS = [
    # (lane_key, name, agent server args-builder, port)
    ("btl4",    "BTL-4 Q4_K_M (think)", STOCK, f"{LLMS}/dogukanurker-btl4-q4km/BTL-4-Q4_K_M.gguf", []),
    ("muse",    "Muse-Glimmer-30B-UD-Q4_K_XL (think)", STOCK, f"{LLMS}/Muse-Glimmer-30B-UD-Q4_K_XL.gguf", []),
    ("gemma",   "gemma-4-12B-it-QAT Q4_0 (think)", STOCK, f"{LLMS}/gemma-4-12B-it-QAT-Q4_0.gguf", []),
    ("bonsai2", "Ternary-Bonsai-2-27B PQ2_0 (think)", PRISMML, f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf", ["-fit", "off"]),
    ("crack",   "Bonsai-2-27B Ternary CRACK PQ2_0 (think)", PRISMML, f"{STAGE}/crack/Bonsai-2-27B-PQ2_0-CRACK.gguf", ["-fit", "off"]),
    ("ptq10",   "Bonsai-2-27B PTQ1_0 (think)", SUDOINGX, f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", []),
    ("xing",    "Xing4.0-29B-A4B IQ4_NL (think)", XING4, f"{STAGE}/xing/xing4_0-29b-IQ4_NL-00001-of-00003.gguf", []),
]

def serve(binary, model, extra, gpu, tag, port, ctx=32768, parallel=2, kv="q8_0", threads=8):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(os.path.join(LOGS, f"{tag}_pp_server.log"), "w")
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
                raise RuntimeError(f"{tag} identity mismatch")
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
    ulogf = open(os.path.join(LOGS, "usersim_pp.log"), "w")
    usim = subprocess.Popen([STOCK, "--model", os.path.join(LLMS, USER_SIM_FILE),
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
            raise RuntimeError("user sim died")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            log(f"  user sim up (GPU{gpu}:{port})")
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
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"
    env["NO_PROXY"] = "*"; env["no_proxy"] = "*"
    tlog = os.path.join(LOGS, f"tau2pp_{re.sub(r'[^A-Za-z0-9]+', '_', safe)}.log")
    try:
        with open(tlog, "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                           cwd=TAU2_DIR, env=env, timeout=8 * 3600)
    except subprocess.TimeoutExpired:
        log("  tau2 timeout")
    return parse_tau2(sdir)

def parse_tau2(sdir):
    rj = os.path.join(sdir, "results.json")
    if not os.path.exists(rj):
        return {"reward": None, "error": "no results.json"}
    try:
        d = json.load(open(rj))
        sims = d.get("simulations", [])
        per = [(s.get("reward_info") or {}).get("reward") for s in sims]
        scored = [x for x in per if x is not None]
        return {
            "reward": round(sum(x or 0 for x in per) / 15, 4) if sims else None,
            "sims_scored": f"{len(scored)}/15",
            "passes": sum(1 for x in scored if x == 1.0),
            "infra_errors": sum(1 for s in sims
                                if s.get("termination_reason") == "infrastructure_error"),
        }
    except Exception as e:
        return {"reward": None, "error": str(e)[:200]}

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    results = {}
    if os.path.exists(RESULTS):
        results = json.load(open(RESULTS))
    stop_prods()
    try:
        if not only or "gates" in only:
            for key, name, binary, model, extra in GATE_JOBS:
                log(f"=== gate {key} ===")
                try:
                    sc = run_gate(name, binary, model, extra)
                except Exception as e:
                    sc = None
                    log(f"  gate error: {str(e)[:200]}")
                results.setdefault(key, {})["sanity"] = sc
                log(f"  sanity: {sc}")
                json.dump(results, open(RESULTS, "w"), indent=2)

        if not only or "tau2" in only:
            usim = ulogf = None
            proc = logf = None
            for key, name, binary, model, extra in TAU2_JOBS:
                log(f"=== tau2 {key} ===")
                try:
                    usim, ulogf = start_usersim(gpu="1")
                    proc, logf = serve(binary, model, extra, 0, f"{name}_pp", 18096)
                    conn = http.client.HTTPConnection("127.0.0.1", 8081, timeout=120)
                    conn.request("POST", "/v1/chat/completions", json.dumps(
                        {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8,
                         "temperature": 0.0}), {"Content-Type": "application/json"})
                    conn.getresponse().read(); conn.close()
                    r = tau2(name, 18096)
                    results.setdefault(key, {})["tau2"] = r
                    log(f"  tau2: {r}")
                except Exception as e:
                    results.setdefault(key, {})["tau2"] = {"error": str(e)[:300]}
                    log(f"  tau2 ERROR: {str(e)[:200]}")
                finally:
                    stop(proc, logf); proc = logf = None
                    if usim:
                        usim.terminate()
                        try:
                            usim.wait(timeout=10)
                        except Exception:
                            usim.kill()
                        if ulogf:
                            ulogf.close()
                        usim = ulogf = None
                json.dump(results, open(RESULTS, "w"), indent=2)
    finally:
        start_prods()
    log("post-pass complete")

if __name__ == "__main__":
    main()
