#!/usr/bin/env python3
"""TwIL pass4: 3060 decode with the CORRECT combo = lfm3-build server (native sm_86)
+ fitting ctx (q4_0 KV, 8192). This is the one missing measurement.
"""
import subprocess, json, time, os, http.client, urllib.request
MODEL="/home/files/llms/twil-8b/TwIL-8B-Q4_K_M-balanced.gguf"
SG="/var/tmp/llms/llama-lfm3-build/build/bin/llama-server"
PORT=18096
def sudo(c,t=120): return subprocess.run(f"sudo -n {c}",shell=True,capture_output=True,text=True,timeout=t)
def wait_health(port,tries=240):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health",timeout=2) as r:
                if json.load(r).get("status")=="ok": return True
        except Exception: time.sleep(5)
    return False
def chat(bin_server):
    # start lfm3-build server on 3060, fit ctx q4 KV
    env={"CUDA_VISIBLE_DEVICES":"1","PATH":"/usr/bin:/bin"}
    cmd=[SG,"-m",MODEL,"--port",str(PORT),"-c","8192","--cache-type-k","q4_0",
         "--cache-type-v","q4_0","-fa","1","--jinja","-sm","none","--parallel","1",
         "--chat-template-kwargs",'{"enable_thinking": false}']
    p=subprocess.Popen(cmd,stdout=open("/tmp/twil_3060_p4.log","w"),stderr=subprocess.STDOUT,env=env)
    res={}
    if wait_health(PORT):
        best=0
        for _ in range(2):
            body=json.dumps({"messages":[{"role":"user","content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens":256,"temperature":0.0,"stream":False,
                "chat_template_kwargs":{"enable_thinking":False}})
            c=http.client.HTTPConnection("127.0.0.1",PORT,timeout=900); t0=time.time()
            c.request("POST","/v1/chat/completions",body,{"Content-Type":"application/json"})
            d=json.loads(c.getresponse().read()); dt=time.time()-t0; c.close()
            u=d.get("usage",{}); toks=u.get("completion_tokens",0)
            if toks: best=max(best,toks/dt)
        res={"decode_tps":round(best,2)}
        p.terminate(); p.wait(timeout=15)
    else:
        res={"error":"server failed"}
        print("srvlog:",open("/tmp/twil_3060_p4.log").read()[-500:])
    return res

sudo("systemctl stop caimlas-gemma-4-26b-v100"); sudo("systemctl stop caimlas-bonsai-3060")
time.sleep(6)
try:
    r=chat(SG)
    print("3060 lfm3-build fit-ctx decode:", json.dumps(r))
    json.dump({"3060_final_decode":r},open("/home/caimlas/llm-benchmarks/results/twil_p4.json","w"),indent=1)
finally:
    sudo("systemctl start caimlas-gemma-4-26b-v100"); sudo("systemctl start caimlas-bonsai-3060")
    time.sleep(6)
    print("[done] prod restored")
