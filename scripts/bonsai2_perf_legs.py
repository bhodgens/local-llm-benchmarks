#!/usr/bin/env python3
"""Bonsai-2 perf legs 1-3 on V100 (prism HEAD build). STAGED — run manually after user gate.
Baselines to beat (bench-0918, thinking-off): PQ2_0 44.35 t/s tg | PTQ1_0 43.4 | PTQ1_0+MTP 54.4.
Experiments: (1) PTQ1_0 vs PQ2_0 llama-bench A/B  (2) MTP server draft acceptance probe  (3) ubatch sweep.
Queue rule: STOPS prod V100 service (caimlas-minicpm-v100) first, restores in finally.
"""
import json, subprocess, time, urllib.request, sys

BIN = "/tmp/prism-head/build/bin"
MODELS = "/var/tmp/llms/bench-0918/bonsai2"
PORT = 18096
RESULTS = {}

def sh(cmd, timeout=1800):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def wait_health(port, tries=120):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(5)
    return False

def stop_prod():
    sh("sudo -n systemctl stop caimlas-minicpm-v100")
    time.sleep(3)
    out = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0").stdout.strip()
    print(f"[prod] V100 compute procs now: {out or 'none'}")

def start_prod():
    sh("sudo -n systemctl start caimlas-minicpm-v100")
    print("[prod] caimlas-minicpm-v100 restart issued")

# ---- leg 1: llama-bench A/B PQ2_0 vs PTQ1_0 (x ubatch variants) ----
def leg1():
    m_pq2 = f"{MODELS}/Ternary-Bonsai-2-27B-PQ2_0.gguf"
    m_ptq = f"{MODELS}/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    cmd = (f"{BIN}/llama-bench -m {m_pq2} -m {m_ptq} "
           f"-p 512,4096 -n 128 -fa 1 -ctk q8_0 -ctv q8_0 -r 3 -t 8 -ub 512,2048 "
           f"-o json")
    print("[leg1] running llama-bench A/B...")
    r = sh(cmd, timeout=3600)
    RESULTS["leg1_bench"] = r.stdout
    print(r.stdout[-2000:] if r.stdout else r.stderr[-1000:])

# ---- leg 2: MTP acceptance probe ----
def leg2():
    m = f"{MODELS}/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
    srv = (f"{BIN}/llama-server -m {m} --port {PORT} -ngl 99 -c 32768 -fa 1 "
           f"-ctk q8_0 -ctv q8_0 -ub 1024 --jinja -sm none --main-gpu 0 "
           f"--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0")
    env = {"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"}
    print("[leg2] starting MTP server...")
    p = subprocess.Popen(srv.split(), stdout=open("/tmp/leg2_server.log","w"), stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_health(PORT):
            print("[leg2] server failed to come up"); return
        body = json.dumps({"messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],
                           "max_tokens":512, "temperature":0}).encode()
        req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=body,
                                     headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        RESULTS["leg2_mtp"] = {"timings": d.get("timings"), "usage": d.get("usage"),
                               "content_head": (d["choices"][0]["message"].get("content") or "")[:200]}
        print(json.dumps(RESULTS["leg2_mtp"], indent=1)[:1200])
        # non-MTP baseline on same file
        p.terminate(); p.wait(); time.sleep(3)
        srv2 = srv.replace("--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k q8_0 --spec-draft-type-v q8_0", "")
        print("[leg2] baseline (no MTP) on same file...")
        p = subprocess.Popen(srv2.split(), stdout=open("/tmp/leg2b_server.log","w"), stderr=subprocess.STDOUT)
        if wait_health(PORT):
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            RESULTS["leg2_nomtp"] = {"timings": d.get("timings"), "usage": d.get("usage")}
            print(json.dumps(RESULTS["leg2_nomtp"], indent=1)[:600])
    finally:
        p.terminate()
        try: p.wait(timeout=10)
        except Exception: p.kill()

# ---- leg 3: ubatch sweep (short-ctx decode + prefill matrix) ----
def leg3():
    m = f"{MODELS}/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
    cmd = (f"{BIN}/llama-bench -m {m} -p 512,16384 -n 256 -fa 1 -ctk q8_0 -ctv q8_0 "
           f"-ub 512,1024,2048 -r 2 -t 8 -o json")
    print("[leg3] ubatch sweep...")
    r = sh(cmd, timeout=3600)
    RESULTS["leg3_ub"] = r.stdout
    print(r.stdout[-1500:] if r.stdout else r.stderr[-800:])

if __name__ == "__main__":
    legs = sys.argv[1:] or ["1","2","3"]
    stop_prod()
    try:
        if "1" in legs: leg1()
        if "2" in legs: leg2()
        if "3" in legs: leg3()
    finally:
        start_prod()
        with open("/home/caimlas/llm-benchmarks/results/bonsai2_perf_legs.json","w") as f:
            json.dump(RESULTS, f, indent=1)
        print("[done] results -> results/bonsai2_perf_legs.json")
