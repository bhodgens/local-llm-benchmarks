#!/usr/bin/env python3
"""TwIL-8B re-probe: fix prefill (both GPUs) + 3060 decode (right engine).
V100: build-bench (sm_70 native). 3060: lfm3-build (sm_86 native).
Pauses prod, restores in finally. Writes redo numbers to a JSON.
"""
import subprocess, json, time, os, re, http.client, sys
from datetime import datetime, timezone

V100_BIN = "/home/caimlas/git/llama.cpp/build-bench/bin"
M3060_BIN = "/var/tmp/llms/llama-lfm3-build/build/bin"
MODEL = "/home/files/llms/twil-8b/TwIL-8B-Q4_K_M-balanced.gguf"
OUT = "/home/caimlas/llm-benchmarks/results/twil_redone_20261107.json"
PORT_V100 = 18095
PORT_3060 = 18096
TS = datetime.now().strftime("%H:%M:%S")
def log(s): print(f"[{TS}] {s}", flush=True)

def sudo(c, t=120): return subprocess.run(f"sudo -n {c}",shell=True,capture_output=True,text=True,timeout=t)

def wait_health(port, tries=240):
    for _ in range(tries):
        try:
            import urllib.request, json as j
            with urllib.request.urlopen(f"http://localhost:{port}/health",timeout=2) as r:
                if j.load(r).get("status")=="ok": return True
        except Exception: time.sleep(5)
    return False

def props(port):
    try:
        import urllib.request, json as j
        with urllib.request.urlopen(f"http://localhost:{port}/props",timeout=3) as r:
            return j.load(r).get("model_path","")
    except Exception: return ""

def start_srv(binp, port, gpu):
    env={"CUDA_VISIBLE_DEVICES":str(gpu), "PATH":"/usr/bin:/bin"}
    cmd=[binp+"/llama-server","-m",MODEL,"--port",str(port),"-c","65536","-fa","1",
         "--jinja","-sm","none","--parallel","1",
         "--chat-template-kwargs",'{"enable_thinking": false}']
    p=subprocess.Popen(cmd,stdout=open(f"/tmp/twil_rp_{port}.log","w"),stderr=subprocess.STDOUT,env=env)
    if not wait_health(port):
        try: p.terminate()
        except Exception: pass
        return None
    return p

def speed_probe(binp, port, gpu):
    # run 2 probes, take best 256-token chat completion
    env={"CUDA_VISIBLE_DEVICES":str(gpu)}
    best=0
    for _ in range(2):
        body=json.dumps({"messages":[{"role":"user","content":
            "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
            "max_tokens":256,"temperature":0.0,"stream":False,
            "chat_template_kwargs":{"enable_thinking":False}})
        try:
            import http.client
            c=http.client.HTTPConnection("127.0.0.1",port,timeout=900); t0=time.time()
            c.request("POST","/v1/chat/completions",body,{"Content-Type":"application/json"})
            d=json.loads(c.getresponse().read()); dt=time.time()-t0
            toks=(d.get("usage") or {}).get("completion_tokens"); c.close()
            if toks: best=max(best,toks/dt)
        except Exception as e: log(f"  speed probe err {e}")
    return {"decode_tps": round(best,2)} if best else {}

def prefill_probe(binp, gpu):
    env={"CUDA_VISIBLE_DEVICES":str(gpu)}
    cmd=f"{binp}/llama-bench -m {MODEL} -p 512 -n 1 -fa 1 -t 8 -o json"
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=1800,env=dict(os.environ,**env))
    # parse robustly
    raw=r.stdout.strip()
    try:
        data=json.loads(raw)
        rows=data if isinstance(data,list) else data.get("results",data.get("tests",[]))
        for row in rows if isinstance(rows,list) else []:
            if row.get("n_prompt")==512:
                return {"prompt_tps": round(row.get("avg_ts",row.get("t/s",0)),2)}
    except Exception:
        pass
    return {"prompt_tps": None, "raw": raw[-400:]}

if __name__=="__main__":
    log("== TwIL re-probe start ==")
    sudo("systemctl stop caimlas-gemma-4-26b-v100")
    sudo("systemctl stop caimlas-bonsai-3060")
    time.sleep(6)
    res={}
    try:
        # V100 prefill (build-bench native sm_70)
        rp = prefill_probe(V100_BIN, 0)
        res["V100_prefill"] = rp
        log(f"V100 prefill: {rp}")
        # 3060 speed + prefill (lfm3-build native sm_86)
        ap=start_srv(M3060_BIN, PORT_3060, 1)
        if ap:
            sp=speed_probe(M3060_BIN, PORT_3060, 1)
            res["3060_speed"]=sp
            log(f"3060 speed: {sp} props={props(PORT_3060)}")
            ap.terminate(); ap.wait(timeout=15)
            rp2=prefill_probe(M3060_BIN, 1)
            res["3060_prefill"]=rp2
            log(f"3060 prefill: {rp2}")
        else:
            res["3060_speed"]={"error":"agent failed to start"}
    finally:
        sudo("systemctl start caimlas-gemma-4-26b-v100")
        sudo("systemctl start caimlas-bonsai-3060")
        time.sleep(6)
        json.dump(res, open(OUT,"w"), indent=1)
        log(f"[done] -> {OUT}, prod restored")
