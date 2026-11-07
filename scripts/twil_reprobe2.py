#!/usr/bin/env python3
"""TwIL-8B probe pass 2: full llama-bench pp512 + tg128 on BOTH cards (capture all
output incl stderr), plus a verbose 3060 decode probe to explain the 13.5 t/s.
V100=build-bench(sm70), 3060=lfm3-build(sm86). Pause prod, restore finally.
"""
import subprocess, json, time, os, http.client
MODEL="/home/files/llms/twil-8b/TwIL-8B-Q4_K_M-balanced.gguf"
OUT="/home/caimlas/llm-benchmarks/results/twil_redone2_20261107.json"
V100_BIN="/home/caimlas/git/llama.cpp/build-bench/bin"
M3060_BIN="/var/tmp/llms/llama-lfm3-build/build/bin"
def sudo(c,t=120): return subprocess.run(f"sudo -n {c}",shell=True,capture_output=True,text=True,timeout=t)
def log(s): print(s,flush=True)

def lb(binp,gpu,pp,tg):
    cmd=f"{binp}/llama-bench -m {MODEL} -p {pp} -n {tg} -fa 1 -t 8 -o json"
    env=dict(os.environ); env["CUDA_VISIBLE_DEVICES"]=str(gpu)
    r=subprocess.run(cmd,shell=True,capture_output=True,text=True,timeout=1800,env=env)
    # return stdout + stderr
    return {"rc":r.returncode,"stdout":r.stdout,"stderr":r.stderr[-500:]}

if __name__=="__main__":
    log("== pass2 start ==")
    sudo("systemctl stop caimlas-gemma-4-26b-v100"); sudo("systemctl stop caimlas-bonsai-3060")
    time.sleep(6); res={}
    try:
        # V100 full bench capture
        res["V100_lb"] = lb(V100_BIN,0,512,1)
        log(f"V100 LB stdout={res['V100_lb']['stdout'][-1200:]}")
        log(f"V100 LB stderr={res['V100_lb']['stderr']}")
        # 3060 full bench capture
        res["G3060_lb"] = lb(M3060_BIN,1,512,1)
        log(f"3060 LB stdout={res['G3060_lb']['stdout'][-1200:]}")
        log(f"3060 LB stderr={res['G3060_lb']['stderr']}")
        # 3060 tg128 to see decode scaling
        res["G3060_tg128"] = lb(M3060_BIN,1,512,128)
        log(f"3060 LB tg128 stdout={res['G3060_tg128']['stdout'][-1000:]}")
    finally:
        sudo("systemctl start caimlas-gemma-4-26b-v100"); sudo("systemctl start caimlas-bonsai-3060")
        time.sleep(6)
        json.dump(res,open(OUT,"w"),indent=1,default=str)
        log(f"[done] -> {OUT}")
