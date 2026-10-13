#!/usr/bin/env python3
"""
LFM2.5-24B-A2B full benchmark on BOTH cards (V100 + 3060) at 128K context.

Stages per card, run sequentially (shared host CPU):
  1. start_model  -> llama-server at 131072 ctx
  2. throughput   -> prompt-eval + decode tok/s (via /completion)
  3. LCB          -> LiveCodeBench codegeneration, 75 probs
  4. tau2         -> agentic (agent on this card, LFM2.5-8B-A1B sim on the other)

Output: /tmp/coding-bench/results/lfm24_<CARD>.json (incremental)
Card 0 (V100): full GPU offload, -ngl 99, flash-attn, ubatch 512
Card 1 (3060): --cpu-moe (experts->CPU), -ngl 99
Sim (LFM2.5-8B-A1B-Clean-RealWorld-v2) runs on the OPPOSITE card during tau2.
"""
import subprocess, json, time, os, urllib.request, shutil

BINARY = "/home/caimlas/git/llama.cpp/build-bench/bin/llama-server"
SIM_FILE = "/home/files/llms/LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
MODEL_FILE = "/var/tmp/llms/lfm25-24b-a2b/LFM2.5-24B-A2B-Q4_K_M.gguf"
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"

CTX = 131072
LCB_NUM = 75
LCB_MAX_TOK = 8192  # LFM is not a thinking arch; plain budget

# Ports: agent card 0 -> 18098, agent card 1 -> 18099, sim -> 18097 (moves to other card)
PORTS = {0: 18098, 1: 18099}
SIM_PORT = 18097

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def healthy(port, timeout=4):
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/health")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r).get("status") == "ok"
    except Exception:
        return False

def wait_port(port, secs=300):
    t0 = time.time()
    while time.time() - t0 < secs:
        if healthy(port):
            return True
        time.sleep(3)
    return False

def start_server(card, model_file, port, extra=None):
    os.makedirs(LOGS, exist_ok=True)
    cmd = [BINARY, "--model", model_file,
           "--flash-attn", "on",
           "--ctx-size", str(CTX),
           "--batch-size", "2048",
           "--ubatch-size", "512",
           "--threads", "6", "--threads-batch", "6",
           "--host", "127.0.0.1", "--port", str(port),
           "--parallel", "2", "--temp", "0.0", "-n", str(LCB_MAX_TOK)]
    if card == 0:
        cmd += ["-ngl", "99"]
    else:
        cmd += ["-ngl", "99", "--cpu-moe"]
    if extra:
        cmd += extra
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(card)
    logf = open(os.path.join(LOGS, f"lfm24_card{card}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    return proc, logf

def stop_server(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=15)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def throughput(port, card):
    """prompt-eval + decode via /completion."""
    pe_prompt = "Write a detailed multi-paragraph essay on the causes and consequences of the industrial revolution." * 8
    res = {}
    # prompt eval: large prompt, tiny generation
    body = json.dumps({"prompt": pe_prompt, "n_predict": 16, "temperature": 0.0}).encode()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                     headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r); t = d["timings"]
            res["prompt_eval_tps"] = round(t["prompt_n"]/(t["prompt_ms"]/1000),1)
            res["prompt_tokens"] = t["prompt_n"]
    except Exception as e:
        res["prompt_eval_tps"] = None; res["prompt_err"] = str(e)
    # decode: short prompt, long gen, x2
    dec = []
    for p in ["Write a Python binary search with type hints.", "Explain merge sort with pseudocode."]:
        body = json.dumps({"prompt": p, "n_predict": 256, "temperature": 0.0}).encode()
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=body,
                                         headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r); t = d["timings"]
                dec.append(t["predicted_n"]/(t["predicted_ms"]/1000))
        except Exception as e:
            dec.append(None); res["decode_err"]=str(e)
    ok = [x for x in dec if x]
    res["decode_tps_run1"] = round(dec[0],1) if dec and dec[0] else None
    res["decode_tps_run2"] = round(dec[1],1) if len(dec)>1 and dec[1] else None
    res["decode_tps_avg"] = round(sum(ok)/len(ok),1) if ok else None
    log(f"  card{card} throughput: prompt_eval={res['prompt_eval_tps']}, decode={res['decode_tps_avg']}")
    return res

def run_lcb(port, card, model_name):
    lcb_model = f"local/lfm24-a2b-card{card}"
    outdir = os.path.join(LCB_DIR, "output", lcb_model.replace("/","_"))
    shutil.rmtree(outdir, ignore_errors=True)  # fresh (no thinking-cache issue, but clean anyway)
    cmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
           "--model", lcb_model, "--scenario", "codegeneration",
           "--release_version", "release_latest", "--n", "1", "--temperature", "0.0",
           "--max_tokens", str(LCB_MAX_TOK), "--num_problems", str(LCB_NUM),
           "--openai_timeout", "1800", "--evaluate"]
    env = dict(os.environ)
    env["OPENAI_KEY"]="none"; env["OPENAI_BASE_URL"]=f"http://127.0.0.1:{port}/v1"
    env["HF_ALLOW_CODE_EVAL"]="1"
    logpath = os.path.join(LOGS, f"lfm24_card{card}_lcb.log")
    start = time.time()
    try:
        with open(logpath,"w") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=72000, env=env, cwd=LCB_DIR)
        elapsed = time.time()-start
        exit_code = r.returncode
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 72000, "error":"timeout"}
    except Exception as e:
        return {"pass_at_1": None, "wall_time_s": time.time()-start, "error":str(e)}
    # parse
    pass1 = None
    dst = os.path.join(RESULTS, f"lcb_lfm24_card{card}")
    backup_dir = dst
    os.makedirs(backup_dir, exist_ok=True)
    for root,_,files in os.walk(outdir):
        for f in files:
            src=os.path.join(root,f); rel=os.path.relpath(src,outdir)
            shutil.copy2(src, os.path.join(backup_dir,rel.replace("/","_")))
            if f.endswith("_eval.json"):
                try:
                    data=json.load(open(src))
                    if isinstance(data,list) and data and isinstance(data[0],dict):
                        pass1 = data[0].get("pass@1")
                except: pass
    log(f"  card{card} LCB: pass@1={pass1} ({elapsed/60:.0f}min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed,1), "exit_code": exit_code}

def run_tau2(port, card, sim_port):
    env = dict(os.environ)
    model = f"openai/LFM2.5-24B-A2B_Q4_K_M"
    sim = f"openai/LFM2.5-8B-A1B-Clean-RealWorld-v2"
    cmd = ["uv","run","tau2","run","--domain","airline",
           "--agent-llm", model,
           "--agent-llm-args", json.dumps({"api_key":"x","api_base":f"http://127.0.0.1:{port}/v1","temperature":0.0}),
           "--user-llm", sim,
           "--user-llm-args", json.dumps({"api_key":"x","api_base":f"http://127.0.0.1:{sim_port}/v1","temperature":0.0}),
           "--num-tasks","15","--num-trials","1","--max-concurrency","2","--max-steps","30",
           "--max-errors","5","--timeout","300","--seed","42",
           "--save-to", f"tau2_LFM2.5-24B-A2B_card{card}"]
    logpath = os.path.join(LOGS, f"lfm24_card{card}_tau2.log")
    log(f"  card{card} tau2 start (sim on card-{1 if card==0 else 0} :{sim_port})")
    start=time.time()
    try:
        with open(logpath,"w") as lf:
            r=subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                             timeout=18000, env=env, cwd=TAU2_DIR)
        elapsed=time.time()-start
    except Exception as e:
        return {"average_reward": None, "error": str(e)}
    # parse Average Reward from Rich output log
    import re
    avg=None
    try:
        txt=open(logpath).read()
        m=re.findall(r"Average\s+Reward[:\s]*([0-9.]+)", txt, re.I)
        if m: avg=float(m[-1])
    except Exception: pass
    log(f"  card{card} tau2: avg_reward={avg} ({elapsed/60:.0f}min)")
    return {"average_reward": avg, "wall_time_s": round(elapsed,1), "exit_code": r.returncode}

def main():
    os.makedirs(RESULTS, exist_ok=True); os.makedirs(LOGS, exist_ok=True)
    allres = {}
    for card in [0,1]:
        key = f"card{card}"
        log(f"=== LFM2.5-24B-A2B card{card} ({'V100' if card==0 else '3060'}) ===")
        proc, logf = start_server(card, MODEL_FILE, PORTS[card])
        if not wait_port(PORTS[card]):
            stop_server(proc, logf)
            allres[key] = {"error": "server did not become ready"}
            continue
        log(f"  server up on :{PORTS[card]}")
        tp = throughput(PORTS[card], card)
        allres[key] = {"model":"LFM2.5-24B-A2B Q4_K_M","ctx":CTX,
                       "device":"V100" if card==0 else "3060",
                       "throughput": tp, "server":"up"}
        # save incrementally
        with open(os.path.join(RESULTS, f"lfm24_{key}.json"),"w") as f:
            json.dump(allres[key], f, indent=2)
        # LCB
        l = run_lcb(PORTS[card], card, f"LFM2.5-24B-A2B card{card}")
        allres[key]["lcb"] = l
        with open(os.path.join(RESULTS, f"lfm24_{key}.json"),"w") as f:
            json.dump(allres[key], f, indent=2)
        # leave card's server up for tau2 (agent here)
        # tau2: start sim on the OTHER card now
        other = 1 if card==0 else 0
        sim_proc, sim_logf = start_server(other, SIM_FILE, SIM_PORT)
        if wait_port(SIM_PORT):
            t = run_tau2(PORTS[card], card, SIM_PORT)
            allres[key]["tau2"] = t
        else:
            allres[key]["tau2"] = {"error": "sim server did not start"}
        stop_server(sim_proc, sim_logf)
        # stop agent server
        stop_server(proc, logf)
        with open(os.path.join(RESULTS, f"lfm24_{key}.json"),"w") as f:
            json.dump(allres[key], f, indent=2)
        log(f"card{card} DONE")

    with open(os.path.join(RESULTS,"lfm24_final.json"),"w") as f:
        json.dump(allres, f, indent=2)
    log("ALL DONE")

if __name__ == "__main__":
    main()
