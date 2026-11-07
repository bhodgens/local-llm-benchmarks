#!/usr/bin/env python3
"""TwIL-8B intake lane Nov 2026. V100 = FULL standard suite (speed, prefill,
sanity:25, LCB 75 thinking-off, HumanEval, tau2 w/ LFM2.5.8B sim on the 3060).
3060 = tok/s + prefill probe only (comparative, per user).
Stops prod (gemma-4-26b-v100 + bonsai-3060), restores in finally.
Engine: V100 build-bench (has granite + lfm2moe arches). Records to
results/twil_intake_<ts>.json. 
"""
import subprocess, json, time, os, re, glob, urllib.request, shutil, sys, http.client
from datetime import datetime, timezone

BIN = "/home/caimlas/git/llama.cpp/build-bench/bin"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
LLMS = "/home/files/llms"
HE_PROXY = "/home/caimlas/llm-benchmarks/scripts/borg/run_humaneval_local.py"
SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
MODEL_FILE = "/home/files/llms/twil-8b/TwIL-8B-Q4_K_M-balanced.gguf"
AGENT_PORT_V100 = 18095
AGENT_PORT_3060 = 18096
SIM_PORT = 8081
TS = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT_FILE = f"/home/caimlas/llm-benchmarks/results/twil_intake_{TS}.json"
OUT = {}

V100_MODEL = {
    "name": "TwIL-8B Q4_K_M balanced (V100)",
    "key": "local/twil-8b-q4km-v100",
    "file": MODEL_FILE, "port": AGENT_PORT_V100, "gpu": "0", "category": "8B",
    "args": ["-c", "65536", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1",
             "--chat-template-kwargs", '{"enable_thinking": false}'],
}
M3060 = {
    "name": "TwIL-8B Q4_K_M balanced (3060)",
    "key": "local/twil-8b-q4km-3060",
    "file": MODEL_FILE, "port": AGENT_PORT_3060, "gpu": "1", "category": "8B",
    "args": ["-c", "65536", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1",
             "--chat-template-kwargs", '{"enable_thinking": false}'],
}

class L:
    def __init__(self): self.buf=[]
    def p(self, s=""):
        ts=datetime.now(timezone.utc).strftime("%H:%M:%S")
        line=f"[{ts}] {s}"; self.buf.append(line); print(line, flush=True)
log=L()

def sh(cmd, timeout=3600, cwd=None):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd)

def sudo(cmd, timeout=120):
    return subprocess.run(f"sudo -n {cmd}", shell=True, capture_output=True, text=True, timeout=timeout)

def stop_prod():
    sudo("systemctl stop caimlas-gemma-4-26b-v100")
    sudo("systemctl stop caimlas-bonsai-3060")
    time.sleep(6)
    log.p("[prod] stopped gemma-4-26b-v100 + bonsai-3060")

def start_prod():
    sudo("systemctl start caimlas-gemma-4-26b-v100")
    sudo("systemctl start caimlas-bonsai-3060")
    time.sleep(6)
    log.p("[prod] restart issued gemma-4-26b-v100 + bonsai-3060")

def wait_health(port, tries=300):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                if json.load(r).get("status")=="ok": return True
        except Exception: time.sleep(5)
    return False

def props(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/props", timeout=3) as r:
            return json.load(r).get("model_path","")
    except Exception: return ""

def start_srv(m):
    env={"CUDA_VISIBLE_DEVICES": m["gpu"], "PATH":"/usr/bin:/bin"}
    cmd=[f"{BIN}/llama-server","-m",m["file"],"--port",str(m["port"])]+m["args"]
    safe=re.sub(r'[^A-Za-z0-9]+','_',m["name"])[:60]
    p=subprocess.Popen(cmd, stdout=open(f"/tmp/agent_{safe}.log","w"), stderr=subprocess.STDOUT, env=env)
    if not wait_health(m["port"]):
        log.p(f"  agent {m['name']} FAILED to start"); 
        try: p.terminate()
        except Exception: pass
        return None
    log.p(f"  agent up on :{m['port']} props={props(m['port'])}")
    return p

def start_sim(gpu):
    env={"CUDA_VISIBLE_DEVICES": gpu, "PATH":"/usr/bin:/bin"}
    cmd=[f"{BIN}/llama-server","-m",os.path.join(LLMS,SIM_FILE),"--port",str(SIM_PORT),
         "-ngl","99","-c","32768","-fa","1","--jinja","-sm","none","--parallel","1"]
    p=subprocess.Popen(cmd, stdout=open("/tmp/user_sim.log","w"), stderr=subprocess.STDOUT, env=env)
    if not wait_health(SIM_PORT):
        log.p("  sim FAILED to start"); 
        try: p.terminate()
        except Exception: pass
        return None
    log.p(f"  sim up on :{SIM_PORT} (gpu {gpu})")
    return p

def speed_probe(port):
    best=0.0; last=None
    for _ in range(2):
        try:
            body=json.dumps({"messages":[{"role":"user","content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens":256,"temperature":0.0,"stream":False,
                "chat_template_kwargs":{"enable_thinking":False}})
            conn=http.client.HTTPConnection("127.0.0.1",port,timeout=900)
            t0=time.time()
            conn.request("POST","/v1/chat/completions",body,{"Content-Type":"application/json"})
            d=json.loads(conn.getresponse().read())
            dt=time.time()-t0
            toks=(d.get("usage") or {}).get("completion_tokens")
            conn.close()
            if toks: best=max(best, toks/dt)
        except Exception as e:
            last=str(e)
    return {"decode_tps": round(best,2)} if best else {"error": last or "no tokens"}

def prefill_probe(m):
    cmd=f"CUDA_VISIBLE_DEVICES={m['gpu']} {BIN}/llama-bench -m {m['file']} -p 512 -n 1 -fa 1 -t 8 -o json 2>/dev/null"
    r=sh(cmd, timeout=1800)
    try:
        data=json.loads(r.stdout)
        if isinstance(data,list):
            for row in data:
                if row.get("n_prompt")==512:
                    return {"prompt_tps": round(row.get("avg_ts",0),2)}
    except Exception: pass
    return {"prompt_tps": None, "raw": r.stdout[-200:]}

def run_sanity(m):
    gate="/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
    safe=re.sub(r'[^A-Za-z0-9]+','_',m["name"]).strip('_')
    cmd=[BENCH_PY,gate,"--model-file",m["file"],"--name",m["name"],
         "--gpu",str(m["gpu"]),"--benchmarks","sanity:25","--binary",f"{BIN}/llama-server"]
    q=" ".join(["'"+x.replace("'","'\\''")+"'" for x in cmd])
    r=sh(q, timeout=7200)
    out=r.stdout+r.stderr
    m2=re.search(r'([\d.]+)%\s*\((\d+)/(\d+)\)', out)
    if m2:
        return {"sanity_pct": float(m2.group(1)), "passed": int(m2.group(2)), "total": int(m2.group(3))}
    return {"error": out[-300:] if out else "empty"}

def run_lcb(m):
    out_dir=os.path.join(LCB_DIR,"output",m["name"])
    if os.path.isdir(out_dir): shutil.rmtree(out_dir)
    lenv=dict(os.environ); lenv["LCB_DISABLE_THINKING"]="1"; lenv["HF_ALLOW_CODE_EVAL"]="1"
    lenv["OPENAI_BASE_URL"]=f"http://127.0.0.1:{m['port']}/v1"; lenv["OPENAI_KEY"]="none"
    logf=f"/tmp/lcb_{m['key'].split('/')[-1]}.log"
    t0=time.time()
    with open(logf,"w") as lf:
        subprocess.run([BENCH_PY,"-m","lcb_runner.runner.main",
                        "--model",m["key"],"--scenario","codegeneration",
                        "--release_version","release_latest","--n","1",
                        "--temperature","0.0","--max_tokens","4096",
                        "--num_problems","75","--openai_timeout","900","--evaluate"],
                       stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR, env=lenv, timeout=24*3600)
    wall=round(time.time()-t0,1)
    pass1=None
    for f in glob.glob(os.path.join(out_dir,"*_eval.json")):
        if f.endswith("_all.json"): continue
        try:
            data=json.load(open(f))
            if isinstance(data,list) and data:
                pass1=data[0].get("pass@1")
        except Exception: pass
    if pass1 is None:
        txt=open(logf).read().strip()
        m2=re.search(r"^([\d.]+)$", txt.splitlines()[-1].strip())
        if m2: pass1=float(m2.group(1))
    # empty-output audit
    empties=0; total_files=0
    for f in glob.glob(os.path.join(out_dir,"*.json")):
        if f.endswith("_all.json") or f.endswith("_eval.json"): continue
        total_files+=1
        try:
            d=json.load(open(f))
            if isinstance(d,list) and d and not d[0].get("output"): empties+=1
        except Exception: pass
    return {"pass_at_1": pass1, "wall_s": wall, "log": logf,
            "empty_outputs": empties, "gen_files": total_files}

def run_humaneval(m):
    logf=f"/tmp/he_{m['key'].split('/')[-1]}.log"
    out_jsonl=f"/home/caimlas/llm-benchmarks/results/humaneval_{m['key'].split('/')[-1]}.jsonl"
    safe=re.sub(r'[^A-Za-z0-9]+','_',m["name"]).strip('_')
    t0=time.time()
    with open(logf,"w") as lf:
        subprocess.run([BENCH_PY,HE_PROXY,"--port",str(m["port"]),
                        "--model",m["name"],"--label",safe,"--out",out_jsonl],
                       stdout=lf, stderr=subprocess.STDOUT, timeout=24*3600)
    wall=round(time.time()-t0,1)
    pass1=None
    txt=open(logf).read()
    for line in txt.splitlines():
        m2=re.search(r'"humaneval_pass@1":\s*([\d.]+)', line)
        if m2: pass1=float(m2.group(1))
    return {"pass_at_1": pass1, "wall_s": wall, "log": logf}

def run_tau2(m, sim_port):
    safe=re.sub(r'[^A-Za-z0-9]+','_',m["name"]).strip('_')
    save_dir=os.path.join(TAU2_DIR,"data","simulations",f"tau2_{safe}")
    if os.path.isdir(save_dir): shutil.rmtree(save_dir)
    agent_model=f"openai/{safe}"
    user_model="openai/LFM2.5-8B-A1B-Clean-RealWorld-v2"
    cmd=["uv","run","tau2","run","--domain","airline",
         "--agent-llm",agent_model,
         "--agent-llm-args",json.dumps({"api_key":"none","api_base":f"http://127.0.0.1:{m['port']}/v1","temperature":0.0}),
         "--user-llm",user_model,
         "--user-llm-args",json.dumps({"api_key":"none","api_base":f"http://localhost:{sim_port}/v1"}),
         "--num-tasks","15","--num-trials","1","--max-concurrency","2",
         "--max-steps","30","--max-errors","5","--timeout","300","--seed","42",
         "--save-to",save_dir]
    env=dict(os.environ); env["OPENAI_API_KEY"]="none"
    logf=f"/tmp/tau2_{safe}.log"
    t0=time.time()
    with open(logf,"w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=TAU2_DIR, env=env, timeout=24*3600)
    wall=round(time.time()-t0,1)
    reward=None; sims=0; scored=0; infra=0
    rj=os.path.join(save_dir,"results.json")
    if os.path.exists(rj):
        try:
            data=json.load(open(rj))
            sim_list=data.get("simulations",[])
            vals=[(s.get("reward_info") or {}).get("reward") for s in sim_list]
            scored=sum(1 for v in vals if v is not None)
            infra=sum(1 for s in sim_list if s.get("termination_reason")=="infrastructure_error")
            reward=round(sum(v for v in vals if v is not None)/15.0,4) if len(vals)==15 else None
            sims=len(sim_list)
        except Exception as e:
            log.p("  tau2 parse err: "+str(e))
    return {"tau2_reward": reward, "sims": sims, "scored": scored, "infra": infra,
            "wall_s": wall, "log": logf}

def full_lane_v100():
    m=V100_MODEL
    log.p(f"=== FULL LANE {m['name']} ===")
    res={}
    ap=start_srv(m)
    if not ap: return {"error":"agent failed"}
    try:
        res["speed"]=speed_probe(m["port"])
        res["prefill"]=prefill_probe(m)
        log.p(f"  speed={res['speed'].get('decode_tps')} prefill={res['prefill'].get('prompt_tps')}")
        res["sanity"]=run_sanity(m)
        log.p(f"  sanity={res['sanity']}")
        res["lcb"]=run_lcb(m)
        log.p(f"  lcb={res['lcb'].get('pass_at_1')} empties={res['lcb'].get('empty_outputs')}")
        res["humaneval"]=run_humaneval(m)
        log.p(f"  humaneval={res['humaneval'].get('pass_at_1')}")
        # tau2: agent + sim on 3060
        ap.terminate(); ap.wait(timeout=15); time.sleep(3)
        sp=start_sim("1")
        if sp:
            ap=start_srv(m)
            if ap:
                res["tau2"]=run_tau2(m, SIM_PORT)
                ap.terminate(); ap.wait(timeout=15)
            sp.terminate(); sp.wait(timeout=15)
        time.sleep(3)
    except Exception as e:
        log.p(f"  ERROR in {m['name']}: {e}")
    finally:
        try: ap.terminate(); ap.wait(timeout=10)
        except Exception: pass
    OUT[m["name"]]=res
    json.dump(OUT, open(OUT_FILE,"w"), indent=1, default=str)
    log.p(f"  saved -> {OUT_FILE}")
    return res

def tok_probe_3060():
    m=M3060
    log.p(f"=== TOK PROBE {m['name']} ===")
    res={}
    ap=start_srv(m)
    if not ap: return {"error":"agent failed"}
    try:
        res["speed"]=speed_probe(m["port"])
        res["prefill"]=prefill_probe(m)
        log.p(f"  speed={res['speed'].get('decode_tps')} prefill={res['prefill'].get('prompt_tps')}")
    finally:
        try: ap.terminate(); ap.wait(timeout=10)
        except Exception: pass
    OUT[m["name"]]=res
    json.dump(OUT, open(OUT_FILE,"w"), indent=1, default=str)
    return res

if __name__=="__main__":
    if not os.path.exists(MODEL_FILE):
        log.p(f"[abort] missing {MODEL_FILE}"); sys.exit(2)
    log.p("== TwIL-8B INTAKE START ==")
    stop_prod()
    try:
        full_lane_v100()
        tok_probe_3060()
    finally:
        start_prod()
        json.dump(OUT, open(OUT_FILE,"w"), indent=1, default=str)
        log.p(f"[done] -> {OUT_FILE}")
