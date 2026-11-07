#!/usr/bin/env python3
"""TwIL pass3 final: honest numbers.
V100 pp512 prefill -> lfm3-build llama-bench (has sm_70 + granite).
3060 decode -> llama-server at FIT ctx (-c 8192, q4_0 KV) + 256-tok chat probe.
Pause prod, restore finally.
"""
import subprocess, json, time, os, http.client, urllib.request
MODEL="/home/files/llms/twil-8b/TwIL-8B-Q4_K_M-balanced.gguf"
OUT="/home/caimlas/llm-benchmarks/results/twil_final_20261107.json"
LLB="/var/tmp/llms/llama-lfm3-build/build/bin"
SG="/home/caimlas/git/llama.cpp/build-bench/bin/llama-server"
PORT=18096
def sudo(c,t=120): return subprocess.run(f"sudo -n {c}",shell=True,capture_output=True,text=True,timeout=t)
def log(s): print(s,flush=True)
def wait_health(port,tries=240):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health",timeout=2) as r:
                if json.load(r).get("status")=="ok": return True
        except Exception: time.sleep(5)
    return False
def lb(gpu,pp,tg,extra=""):
    cmd=f"{LLB}/llama-bench -m {MODEL} -p {pp} -n {tg} -fa 1 -t 8 {extra} -o json"
    env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=1800,env=env)
    try:
        d=json.loads(r.stdout)
        rows=d if isinstance(d,list) else d.get("results",[])
        return {"list":[ (x.get("n_prompt"),x.get("n_gen"),round(x.get("avg_ts",0),2)) for x in rows]}
    except Exception:
        return {"raw_stdout":r.stdout[-300:],"stderr":r.stderr[-300:]}
def chat_probe(port,max_tokens=256):
    body=json.dumps({"messages":[{"role":"user","content":
        "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
        "max_tokens":max_tokens,"temperature":0.0,"stream":False,
        "chat_template_kwargs":{"enable_thinking":False}})
    c=http.client.HTTPConnection("127.0.0.1",port,timeout=900); t0=time.time()
    c.request("POST","/v1/chat/completions",body,{"Content-Type":"application/json"})
    d=json.loads(c.getresponse().read()); dt=time.time()-t0; c.close()
    u=d.get("usage",{})
    return {"decode_tps": round(u.get("completion_tokens",0)/dt,2),
            "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens")}

if __name__=="__main__":
    log("== pass3 ==")
    sudo("systemctl stop caimlas-gemma-4-26b-v100"); sudo("systemctl stop caimlas-bonsai-3060")
    time.sleep(6); res={}
    try:
        # V100 pp512 via lfm3-build llama-bench (sm70 kernel present)
        res["V100_lb"]=lb(0,512,1)
        log(f"V100 pp512: {res['V100_lb']}")
        # 3060 fit-ctx server: -c 8192 q4_0 KV
        env={"CUDA_VISIBLE_DEVICES":"1","PATH":"/usr/bin:/bin"}
        cmd=[SG,"-m",MODEL,"--port",str(PORT),"-c","8192","--cache-type-k","q4_0",
             "--cache-type-v","q4_0","-fa","1","--jinja","-sm","none","--parallel","1",
             "--chat-template-kwargs",'{"enable_thinking": false}']
        p=subprocess.Popen(cmd,stdout=open("/tmp/twil_3060_fit.log","w"),stderr=subprocess.STDOUT,env=env)
        if wait_health(PORT):
            res["3060_fit_decode"]=chat_probe(PORT)
            log(f"3060 fit-ctx decode: {res['3060_fit_decode']}")
            p.terminate(); p.wait(timeout=15)
        else:
            res["3060_fit_decode"]={"error":"server failed"}
            log("3060 fit server FAILED; log:"+open("/tmp/twil_3060_fit.log").read()[-400:])
    finally:
        sudo("systemctl start caimlas-gemma-4-26b-v100"); sudo("systemctl start caimlas-bonsai-3060")
        time.sleep(6)
        json.dump(res,open(OUT,"w"),indent=1,default=str)
        log(f"[done] -> {OUT}")
