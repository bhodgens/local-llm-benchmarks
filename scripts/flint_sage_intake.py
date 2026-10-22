#!/usr/bin/env python3
"""Flint + Sage-A1.5 intake lanes on V100 (upstream master build). Stops prod, restores after."""
import subprocess, time, json, urllib.request, sys

BIN = "/tmp/upstream-master/build/bin"
PORT = 18095
FLINT = "/home/files/llms/flint/flint-UD-Q4_K_XL.gguf"
SAGE = "/home/files/llms/sage-a1.5/Sage-A1.5-Q4_K_M.gguf"
SAGE_MM = "/home/files/llms/sage-a1.5/Sage-A1.5-mmproj.gguf"
OUT = {}

def sh(cmd, timeout=1800):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def stop_prod():
    sh("sudo -n systemctl stop caimlas-minicpm-v100")
    time.sleep(3)
    out = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0").stdout.strip()
    print(f"[prod] GPU0 procs: {out or 'none'}")

def start_prod():
    sh("sudo -n systemctl start caimlas-minicpm-v100")
    print("[prod] caimlas-minicpm-v100 restart issued")

def wait_health(tries=120):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok": return True
        except Exception: time.sleep(5)
    return False

def props():
    try:
        with urllib.request.urlopen(f"http://localhost:{PORT}/props", timeout=3) as r:
            return json.load(r).get("model_path", "")
    except Exception: return ""

def probe(model, mmproj=None, extra=""):
    srv = [f"{BIN}/llama-server", "-m", model, "--port", str(PORT), "-ngl", "99",
           "-c", "32768", "-fa", "1", "--jinja", "-sm", "none"]
    if mmproj: srv += ["--mmproj", mmproj]
    if extra: srv += extra.split()
    env = {"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin:/bin"}
    p = subprocess.Popen(srv, stdout=open("/tmp/intake_server.log","w"), stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_health():
            print("  server failed"); return {"error": "server failed"}
        mp = props()
        print(f"  /props model_path: {mp}")
        # speed probe x2
        best = None
        for i in range(2):
            body = json.dumps({"messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],
                               "max_tokens":256, "temperature":0}).encode()
            req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=body,
                                         headers={"Content-Type":"application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.load(r)
            t = d.get("timings", {})
            row = {"prompt_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
                   "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
                   "content_head": (d["choices"][0]["message"].get("content") or "")[:120]}
            if best is None or (row["decode_tps"] or 0) > (best["decode_tps"] or 0): best = row
        return {"props": mp, "best": best}
    finally:
        p.terminate()
        try: p.wait(timeout=10)
        except Exception: p.kill()

def bench(model, mmproj=None):
    cmd = f"CUDA_VISIBLE_DEVICES=0 {BIN}/llama-bench -m {model} -p 512,4096 -n 128 -fa 1 -r 2 -t 8 -o md"
    if mmproj: cmd += f" -mmp {mmproj}"
    r = sh(cmd, timeout=1800)
    return r.stdout

if __name__ == "__main__":
    stop_prod()
    try:
        print("== FLINT lane ==")
        OUT["flint"] = probe(FLINT)
        OUT["flint_bench"] = bench(FLINT)
        print("== SAGE lane ==")
        OUT["sage"] = probe(SAGE, SAGE_MM)
        OUT["sage_bench"] = bench(SAGE, SAGE_MM)
        print("== SAGE MTP ==")
        OUT["sage_mtp"] = probe(SAGE, SAGE_MM, extra="--spec-type draft-mtp --spec-draft-n-max 2")
    finally:
        start_prod()
        with open("/home/caimlas/llm-benchmarks/results/flint_sage_intake.json","w") as f:
            json.dump(OUT, f, indent=1)
        print("[done] -> results/flint_sage_intake.json")
