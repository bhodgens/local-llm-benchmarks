#!/usr/bin/env python3
"""LCB + tau2 lanes for Flint and Sage-A1.5 on the V100.
Agent on V100 (upstream master build), user sim LFM2.5-8B on 3060:8081.
Stops prod services, restores in finally. Records to results/flint_sage_lcb_tau2.json.
"""
import subprocess, json, time, os, re, glob, urllib.request, shutil, sys

BIN = "/tmp/upstream-master/build/bin"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
LLMS = "/home/files/llms"
SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
AGENT_PORT = 18095
SIM_PORT = 8081
OUT = {}

MODELS = [
    {"key": "local/flint-398b-q4kxl-v100", "name": "Flint-39.8B Q4_K_XL (V100)",
     "file": "/home/files/llms/flint/flint-UD-Q4_K_XL.gguf",
     "args": ["-ngl","99","-c","32768","-fa","1","--jinja","-sm","none","--parallel","1"]},
    {"key": "local/sage-a15-q4km-v100", "name": "Sage-A1.5 Q4_K_M (V100)",
     "file": "/home/files/llms/sage-a1.5/Sage-A1.5-Q4_K_M.gguf",
     "args": ["-ngl","99","-c","32768","-fa","1","--jinja","-sm","none","--parallel","1",
              "--mmproj","/home/files/llms/sage-a1.5/Sage-A1.5-mmproj.gguf"]},
]

def sh(cmd, timeout=3600, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)

def stop_prod():
    sh("sudo -n systemctl stop caimlas-minicpm-v100")
    sh("sudo -n systemctl stop caimlas-gemma")
    time.sleep(4)
    print("[prod] stopped minicpm + gemma")

def start_prod():
    sh("sudo -n systemctl start caimlas-minicpm-v100")
    sh("sudo -n systemctl start caimlas-gemma")
    print("[prod] restarted minicpm + gemma")

def wait_health(port, tries=180):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok": return True
        except Exception: time.sleep(5)
    return False

def start_agent(m):
    env = {"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"}
    cmd = [f"{BIN}/llama-server", "-m", m["file"], "--port", str(AGENT_PORT)] + m["args"]
    p = subprocess.Popen(cmd, stdout=open(f"/tmp/agent_{m['name'][:8]}.log","w"),
                         stderr=subprocess.STDOUT, env=env)
    if not wait_health(AGENT_PORT):
        print(f"  agent {m['name']} failed to start"); p.terminate(); return None
    return p

def start_sim():
    env = {"CUDA_VISIBLE_DEVICES": "1", "PATH": "/usr/bin:/bin"}
    cmd = [f"{BIN}/llama-server", "-m", os.path.join(LLMS, SIM_FILE), "--port", str(SIM_PORT),
           "-ngl","99","-c","32768","-fa","1","--jinja","-sm","none","--parallel","1"]
    p = subprocess.Popen(cmd, stdout=open("/tmp/sim.log","w"), stderr=subprocess.STDOUT, env=env)
    if not wait_health(SIM_PORT):
        print("  sim failed to start"); p.terminate(); return None
    return p

def run_lcb(m):
    name = m["name"]
    out_dir = os.path.join(LCB_DIR, "output", name)
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    lenv = dict(os.environ); lenv["LCB_DISABLE_THINKING"] = "1"; lenv["HF_ALLOW_CODE_EVAL"] = "1"
    lenv["OPENAI_BASE_URL"] = f"http://127.0.0.1:{AGENT_PORT}/v1"
    lenv["OPENAI_KEY"] = "none"
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
        if txt:
            m2 = re.search(r"^([\d.]+)$", txt.splitlines()[-1].strip())
            if m2: pass1 = float(m2.group(1))
    return {"lcb_pass1": pass1, "wall_s": wall, "log": logf}

def run_tau2(m):
    safe = re.sub(r'[^A-Za-z0-9]+', '_', m["name"]).strip('_')
    save_dir = os.path.join(TAU2_DIR, "data", "simulations", f"tau2_{safe}")
    if os.path.isdir(save_dir):
        shutil.rmtree(save_dir)
    agent_model = f"openai/{safe}"
    user_model = "openai/LFM2.5-8B-A1B-Clean-RealWorld-v2"
    cmd = ["uv","run","tau2","run","--domain","airline",
           "--agent-llm", agent_model,
           "--agent-llm-args", json.dumps({"api_key":"none","api_base":f"http://127.0.0.1:{AGENT_PORT}/v1","temperature":0.0}),
           "--user-llm", user_model,
           "--user-llm-args", json.dumps({"api_key":"none","api_base":f"http://localhost:{SIM_PORT}/v1"}),
           "--num-tasks","15","--num-trials","1","--max-concurrency","2",
           "--max-steps","30","--max-errors","5","--timeout","300","--seed","42",
           "--save-to", save_dir]
    env = dict(os.environ); env["OPENAI_API_KEY"] = "none"
    logf = f"/tmp/tau2_{safe}.log"
    t0 = time.time()
    with open(logf, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=TAU2_DIR, env=env, timeout=24*3600)
    wall = round(time.time()-t0, 1)
    # parse reward from results.json
    reward = None; sims = 0; scored = 0; infra = 0
    rj = os.path.join(save_dir, "results.json")
    if os.path.exists(rj):
        try:
            data = json.load(open(rj))
            sims = data.get("simulations", [])
            vals = [(s.get("reward_info") or {}).get("reward") for s in sims]
            scored = sum(1 for v in vals if v is not None)
            infra = sum(1 for s in sims if s.get("termination_reason") == "infrastructure_error")
            reward = round(sum(v for v in vals if v is not None) / 15.0, 4)
        except Exception as e:
            print("  tau2 parse err:", e)
    return {"tau2_reward": reward, "sims": len(sims) if isinstance(sims, list) else 0,
            "scored": scored, "infra": infra, "wall_s": wall, "log": logf}

if __name__ == "__main__":
    stop_prod()
    try:
        for m in MODELS:
            print(f"== {m['name']} ==")
            ap = start_agent(m)
            if not ap: continue
            OUT[m["name"]] = {"lcb": run_lcb(m)}
            ap.terminate(); ap.wait(timeout=15)
            time.sleep(3)
            sp = start_sim()
            if sp:
                ap = start_agent(m)
                if ap:
                    OUT[m["name"]]["tau2"] = run_tau2(m)
                    ap.terminate(); ap.wait(timeout=15)
                sp.terminate(); sp.wait(timeout=15)
            time.sleep(3)
    finally:
        start_prod()
        with open("/home/caimlas/llm-benchmarks/results/flint_sage_lcb_tau2.json","w") as f:
            json.dump(OUT, f, indent=1)
        print("[done] -> results/flint_sage_lcb_tau2.json")
