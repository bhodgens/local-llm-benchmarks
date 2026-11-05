#!/usr/bin/env python3
"""Two-GPU benchmark orchestrator: Coding-Monkey-Gemma + Qwopus Coder-Flash (V100)
+ LFM3 base + LFM3 uncensored (3060). Full criteria per model: in/out tok/s (speed
+ prefill), sanity:25, LCB 75, human_eval, tau2 (user sim LFM2.5 on the OPPOSITE GPU).
Stops prod services, restores in finally. Records to results/intake_<ts>.json.

Engine: new origin/master build at /var/tmp/llms/llama-lfm3-build/build/bin (has
gemma4 + qwen35 + lfm3 arches). Agent+sim never share a GPU; sim = LFM2.5-8B on
the other card at :8081.
"""
import subprocess, json, time, os, re, glob, urllib.request, shutil, sys, http.client
from datetime import datetime, timezone

BIN = "/var/tmp/llms/llama-lfm3-build/build/bin"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
LLMS = "/home/files/llms"
HE_PROXY = "/home/caimlas/llm-benchmarks/scripts/borg/run_humaneval_local.py"
SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
AGENT_PORT_V100 = 18095   # V100 agent
AGENT_PORT_3060 = 18096   # 3060 agent
SIM_PORT = 8081
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_FILE = f"/home/caimlas/llm-benchmarks/results/intake_{TS}.json"
OUT = {}

# Each model: name (display, GPU-suffixed), key (LCB alias), file, engine_cards,
# agent port, sim port, extra server args.
V100_MODELS = [
    {
        "name": "Coding-Monkey-Gemma gemma-coder Q4_K_M (V100)",
        "key": "local/coding-monkey-gemma4-q4",
        "file": "/var/tmp/llms/coding-monkey-gemma/gemma-coder-Q4_K_M.gguf",
        "port": AGENT_PORT_V100, "gpu": "0", "category": "12B",
        "args": ["-c", "65536", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1",
                 "--chat-template-kwargs", '{"enable_thinking": false}'],
    },
    {
        "name": "Qwopus3.8-27B-Coder-Flash Q4_K_M (V100)",
        "key": "local/qwopus38-coderflash-q4km",
        "file": "/var/tmp/llms/qwopus-coder-flash/Qwopus3.8-27B-Coder-Flash-Q4_K_M.gguf",
        "port": AGENT_PORT_V100, "gpu": "0", "category": "27B",
        "args": ["-c", "65536", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1",
                 "--chat-template-kwargs", '{"enable_thinking": false}'],
    },
]
GPU3060_MODELS = [
    {
        "name": "LFM3-1.7B Q4_K_M (3060)",
        "key": "local/lfm3-17b-q4km",
        "file": "/var/tmp/llms/lfm3-base/LFM3-1.7B-Q4_K_M.gguf",
        "port": AGENT_PORT_3060, "gpu": "1", "category": "1.7B",
        "args": ["-c", "65536", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1",
                 "--chat-template-kwargs", '{"enable_thinking": false}'],
    },
    {
        "name": "LFM3-1.7B-Uncensored AD-Q6_K (3060)",
        "key": "local/lfm3-17b-unc-q6km",
        "file": "/var/tmp/llms/lfm3-uncensored/LFM3-1.7B-Uncensored-AD-Q6_K.gguf",
        "port": AGENT_PORT_3060, "gpu": "1", "category": "1.7B",
        "args": ["-c", "65536", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1",
                 "--chat-template-kwargs", '{"enable_thinking": false}'],
    },
]

class L:
    def __init__(self): self.buf = []
    def p(self, s=""):
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{ts}] {s}"
        self.buf.append(line); print(line, flush=True)

log = L()

def sh(cmd, timeout=3600, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)

def stop_prod():
    # V100 prod (gemma-4-26b) + 3060 prod (bonsai-2) both released for lanes.
    sh("sudo -n systemctl stop caimlas-gemma-4-26b-v100")
    sh("sudo -n systemctl stop caimlas-bonsai-3060")
    time.sleep(5)
    log.p("[prod] stopped gemma-4-26b-v100 + bonsai-3060")

def start_prod():
    sh("sudo -n systemctl start caimlas-gemma-4-26b-v100")
    sh("sudo -n systemctl start caimlas-bonsai-3060")
    time.sleep(5)
    log.p("[prod] restart issued gemma-4-26b-v100 + bonsai-3060")

def wait_health(port, tries=240):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok": return True
        except Exception: time.sleep(5)
    return False

def props(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/props", timeout=3) as r:
            return json.load(r).get("model_path", "")
    except Exception: return ""

def start_srv(m):
    env = {"CUDA_VISIBLE_DEVICES": m["gpu"], "PATH": "/usr/bin:/bin"}
    cmd = [f"{BIN}/llama-server", "-m", m["file"], "--port", str(m["port"])] + m["args"]
    safe = re.sub(r'[^A-Za-z0-9]+', '_', m["name"])[:60]
    p = subprocess.Popen(cmd, stdout=open(f"/tmp/agent_{safe}.log", "w"),
                         stderr=subprocess.STDOUT, env=env)
    if not wait_health(m["port"]):
        log.p(f"  agent {m['name']} FAILED to start"); p.terminate(); return None
    log.p(f"  agent up on :{m['port']} props={props(m['port'])}")
    return p

def start_sim(gpu):
    env = {"CUDA_VISIBLE_DEVICES": gpu, "PATH": "/usr/bin:/bin"}
    cmd = [f"{BIN}/llama-server", "-m", os.path.join(LLMS, SIM_FILE), "--port", str(SIM_PORT),
           "-ngl", "99", "-c", "32768", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1"]
    p = subprocess.Popen(cmd, stdout=open("/tmp/user_sim.log", "w"), stderr=subprocess.STDOUT, env=env)
    if not wait_health(SIM_PORT):
        log.p("  sim FAILED to start"); p.terminate(); return None
    log.p(f"  sim up on :{SIM_PORT} (gpu {gpu})")
    return p

def speed_probe(port):
    best = 0.0; last = None
    for _ in range(2):
        try:
            body = json.dumps({"messages": [{"role": "user", "content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens": 256, "temperature": 0.0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}})
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
            t0 = time.time()
            conn.request("POST", "/v1/chat/completions", body, {"Content-Type": "application/json"})
            d = json.loads(conn.getresponse().read())
            dt = time.time() - t0
            toks = (d.get("usage") or {}).get("completion_tokens")
            conn.close()
            if toks: best = max(best, toks / dt)
        except Exception as e:
            last = str(e)
    return {"decode_tps": round(best, 2)} if best else {"error": last or "no tokens"}

def prefill_probe(m):
    """llama-bench -p 512 -n 1 canonic for input tok/s (pp512). Schema in
    b12126+ is a flat list of test objects; pp512 value = avg_ts of the n_prompt==512 test."""
    cmd = f"CUDA_VISIBLE_DEVICES={m['gpu']} {BIN}/llama-bench -m {m['file']} -p 512 -n 1 -fa 1 -t 8 -o json 2>/dev/null"
    r = sh(cmd, timeout=1800)
    try:
        data = json.loads(r.stdout)
        if isinstance(data, list):
            for row in data:
                if row.get("n_prompt") == 512:
                    return {"prompt_tps": round(row.get("avg_ts", 0), 2)}
    except Exception:
        pass
    return {"prompt_tps": None, "raw": r.stdout[-200:]}

def run_sanity(m):
    gate = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
    safe = re.sub(r'[^A-Za-z0-9]+', '_', m["name"]).strip('_')
    cmd = [BENCH_PY, gate, "--model-file", m["file"], "--name", m["name"],
           "--gpu", str(m["gpu"]), "--benchmarks", "sanity:25", "--binary",
           f"{BIN}/llama-server"]
    r = sh(" ".join(["'".join(["'" + x.replace("'", "'\\''") + "'"]) for x in cmd] if cmd else [""]), timeout=7200)
    out = r.stdout + r.stderr
    m2 = re.search(r'([\d.]+)%\s*\((\d+)/(\d+)\)', out)
    if m2:
        return {"sanity_pct": float(m2.group(1)), "passed": int(m2.group(2)), "total": int(m2.group(3))}
    return {"error": out[-300:] if out else "empty"}

def run_lcb(m):
    out_dir = os.path.join(LCB_DIR, "output", m["name"])
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)
    lenv = dict(os.environ); lenv["LCB_DISABLE_THINKING"] = "1"; lenv["HF_ALLOW_CODE_EVAL"] = "1"
    lenv["OPENAI_BASE_URL"] = f"http://127.0.0.1:{m['port']}/v1"; lenv["OPENAI_KEY"] = "none"
    logf = f"/tmp/lcb_{m['key'].split('/')[-1]}.log"
    t0 = time.time()
    with open(logf, "w") as lf:
        subprocess.run([BENCH_PY, "-m", "lcb_runner.runner.main",
                        "--model", m["key"], "--scenario", "codegeneration",
                        "--release_version", "release_latest", "--n", "1",
                        "--temperature", "0.0", "--max_tokens", "4096",
                        "--num_problems", "75", "--openai_timeout", "900",
                        "--evaluate"],
                       stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR, env=lenv,
                       timeout=24*3600)
    wall = round(time.time()-t0, 1)
    pass1 = None
    for f in glob.glob(os.path.join(out_dir, "*_eval.json")):
        if f.endswith("_all.json"): continue
        try:
            data = json.load(open(f))
            if isinstance(data, list) and data:
                pass1 = data[0].get("pass@1")
        except Exception: pass
    if pass1 is None:
        txt = open(logf).read().strip()
        m2 = re.search(r"^([\d.]+)$", txt.splitlines()[-1].strip())
        if m2: pass1 = float(m2.group(1))
    # count empty outputs
    return {"pass_at_1": pass1, "wall_s": wall, "log": logf}

def run_humaneval(m):
    logf = f"/tmp/he_{m['key'].split('/')[-1]}.log"
    out_jsonl = f"/home/caimlas/llm-benchmarks/results/humaneval_{m['key'].split('/')[-1]}.jsonl"
    safe = re.sub(r'[^A-Za-z0-9]+', '_', m["name"]).strip('_')
    t0 = time.time()
    with open(logf, "w") as lf:
        subprocess.run([BENCH_PY, HE_PROXY, "--port", str(m["port"]),
                        "--model", m["name"], "--label", safe, "--out", out_jsonl],
                       stdout=lf, stderr=subprocess.STDOUT, timeout=24*3600)
    wall = round(time.time()-t0, 1)
    pass1 = None
    txt = open(logf).read()
    for line in txt.splitlines():
        m2 = re.search(r'"humaneval_pass@1":\s*([\d.]+)', line)
        if m2: pass1 = float(m2.group(1))
    return {"pass_at_1": pass1, "wall_s": wall, "log": logf}

def run_tau2(m, sim_port):
    safe = re.sub(r'[^A-Za-z0-9]+', '_', m["name"]).strip('_')
    save_dir = os.path.join(TAU2_DIR, "data", "simulations", f"tau2_{safe}")
    if os.path.isdir(save_dir): shutil.rmtree(save_dir)
    agent_model = f"openai/{safe}"
    user_model = "openai/LFM2.5-8B-A1B-Clean-RealWorld-v2"
    cmd = ["uv", "run", "tau2", "run", "--domain", "airline",
           "--agent-llm", agent_model,
           "--agent-llm-args", json.dumps({"api_key": "none", "api_base": f"http://127.0.0.1:{m['port']}/v1", "temperature": 0.0}),
           "--user-llm", user_model,
           "--user-llm-args", json.dumps({"api_key": "none", "api_base": f"http://localhost:{sim_port}/v1"}),
           "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
           "--max-steps", "30", "--max-errors", "5", "--timeout", "300", "--seed", "42",
           "--save-to", save_dir]
    env = dict(os.environ); env["OPENAI_API_KEY"] = "none"
    logf = f"/tmp/tau2_{safe}.log"
    t0 = time.time()
    with open(logf, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=TAU2_DIR, env=env, timeout=24*3600)
    wall = round(time.time()-t0, 1)
    reward = None; sims = 0; scored = 0; infra = 0
    rj = os.path.join(save_dir, "results.json")
    if os.path.exists(rj):
        try:
            data = json.load(open(rj))
            sims = data.get("simulations", [])
            vals = [(s.get("reward_info") or {}).get("reward") for s in sims]
            scored = sum(1 for v in vals if v is not None)
            infra = sum(1 for s in sims if s.get("termination_reason") == "infrastructure_error")
            reward = round(sum(v for v in vals if v is not None) / 15.0, 4) if len(vals) == 15 else None
        except Exception as e:
            log.p("  tau2 parse err: " + str(e))
    return {"tau2_reward": reward, "sims": len(sims) if isinstance(sims, list) else 0,
            "scored": scored, "infra": infra, "wall_s": wall, "log": logf}

def full_lane(m, sim_gpu):
    log.p(f"=== FULL LANE {m['name']} ===")
    res = {}
    ap = start_srv(m)
    if not ap: return {"error": "agent failed"}
    try:
        res["speed"] = speed_probe(m["port"])
        res["prefill"] = prefill_probe(m)
        log.p(f"  speed={res['speed'].get('decode_tps')} prefill={res['prefill'].get('prompt_tps')}")
        res["sanity"] = run_sanity(m)
        log.p(f"  sanity={res['sanity']}")
        res["lcb"] = run_lcb(m)
        log.p(f"  lcb={res['lcb'].get('pass_at_1')}")
        res["humaneval"] = run_humaneval(m)
        log.p(f"  humaneval={res['humaneval'].get('pass_at_1')}")
        # tau2 needs agent + sim simultaneously; restart agent after LCB window
        ap.terminate(); ap.wait(timeout=15); time.sleep(3)
        sp = start_sim(sim_gpu)
        if sp:
            ap = start_srv(m)
            if ap:
                res["tau2"] = run_tau2(m, SIM_PORT)
                ap.terminate(); ap.wait(timeout=15)
            sp.terminate(); sp.wait(timeout=15)
        time.sleep(3)
    except Exception as e:
        log.p(f"  ERROR in {m['name']}: {e}")
    finally:
        try: ap.terminate(); ap.wait(timeout=10)
        except Exception: pass
    OUT[m["name"]] = res
    json.dump(OUT, open(OUT_FILE, "w"), indent=1, default=str)
    log.p(f"  saved -> {OUT_FILE}")
    return res

def tok_probe_only(m):
    log.p(f"=== TOK PROBE {m['name']} ===")
    res = {}
    ap = start_srv(m)
    if not ap: return {"error": "agent failed"}
    try:
        res["speed"] = speed_probe(m["port"])
        res["prefill"] = prefill_probe(m)
        log.p(f"  speed={res['speed'].get('decode_tps')} prefill={res['prefill'].get('prompt_tps')}")
    finally:
        try: ap.terminate(); ap.wait(timeout=10)
        except Exception: pass
    OUT[m["name"]] = res
    json.dump(OUT, open(OUT_FILE, "w"), indent=1, default=str)
    return res

if __name__ == "__main__":
    missing = [m["file"] for v in (V100_MODELS + GPU3060_MODELS) for m in [v] if not os.path.exists(v["file"])]
    if missing:
        log.p(f"[abort] missing files: {missing}"); sys.exit(2)
    log.p("== INTAKE START ==")
    stop_prod()
    try:
        # Fast 1.7B LFM3 lanes first (3060) to validate plumbing quickly; then the
        # heavy V100 lanes (Coding-Monkey, Qwopus). Each tau2 sim runs on the OPPOSITE
        # GPU, so lanes are sequential, never concurrent.
        for m in GPU3060_MODELS:
            full_lane(m, sim_gpu="0")
        for m in V100_MODELS:
            full_lane(m, sim_gpu="1")
    finally:
        start_prod()
        json.dump(OUT, open(OUT_FILE, "w"), indent=1, default=str)
        log.p(f"[done] -> {OUT_FILE}")
