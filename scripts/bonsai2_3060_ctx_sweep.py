#!/usr/bin/env python3
"""3060 larger-context sweep for Bonsai-2 PTQ1_0. Stops caimlas-gemma, restores after.
Tests: ctx 32K/64K/128K x KV q8_0/q4_0, with/without MTP, decode + prefill + VRAM.
"""
import subprocess, time, json, urllib.request, sys

BIN = "/tmp/prism-head/build/bin"
M = "/var/tmp/llms/bench-0918/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
MMTP = "/var/tmp/llms/bench-0918/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
PORT = 18097
OUT = {}

def sh(cmd, timeout=1800):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def stop_gemma():
    sh("sudo -n systemctl stop caimlas-gemma")
    time.sleep(3)
    out = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 1").stdout.strip()
    print(f"[prod] GPU1 procs: {out or 'none'}")

def start_gemma():
    sh("sudo -n systemctl start caimlas-gemma")
    print("[prod] caimlas-gemma restart issued")

def bench(model, ctx, kv, ub=512, extra=""):
    cmd = (f"CUDA_VISIBLE_DEVICES=1 {BIN}/llama-bench -m {model} -c {ctx} "
           f"-p 512,4096 -n 128 -fa 1 -ctk {kv} -ctv {kv} -ub {ub} -r 2 -t 8 -o md {extra}")
    r = sh(cmd, timeout=1800)
    return r.stdout

def parse_tg(md):
    """extract tg128 t/s from llama-bench md output"""
    import re
    lines = md.splitlines()
    for l in lines:
        if 'tg128' in l or 'tg256' in l:
            m = re.search(r'(\d+\.\d+)\s*±', l)
            if m: return float(m.group(1))
    return None

def parse_pp(md):
    import re
    for l in md.splitlines():
        if 'pp4096' in l:
            m = re.search(r'(\d+\.\d+)\s*±', l)
            if m: return float(m.group(1))
    return None

def vram_used():
    r = sh("nvidia-smi --query-compute-apps=used_memory --format=csv,noheader -i 1")
    return r.stdout.strip()

# ---- sweep: ctx x kv on PTQ1_0 ----
def sweep():
    for ctx in [32768, 65536, 131072]:
        for kv in ['q8_0', 'q4_0']:
            key = f"ctx{ctx//1024}k_{kv}"
            print(f"[sweep] {key} ...")
            md = bench(M, ctx, kv)
            OUT[key] = {"tg128": parse_tg(md), "pp4096": parse_pp(md), "vram": vram_used(), "md": md}
            print(f"  tg128={OUT[key]['tg128']} pp4096={OUT[key]['pp4096']} vram={OUT[key]['vram']}")

# ---- MTP at larger ctx ----
def mtp_probe(ctx, kv):
    srv = (f"{BIN}/llama-server -m {MMTP} --port {PORT} -ngl 99 -c {ctx} -fa 1 "
           f"-ctk {kv} -ctv {kv} -ub 512 --jinja -sm none "
           f"--spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k {kv} --spec-draft-type-v {kv}")
    env = {"CUDA_VISIBLE_DEVICES": "1", "PATH": "/usr/bin:/bin"}
    p = subprocess.Popen(srv.split(), stdout=open("/tmp/mtp3060.log","w"), stderr=subprocess.STDOUT, env=env)
    try:
        for _ in range(120):
            try:
                with urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=2) as r:
                    if json.load(r).get("status") == "ok": break
            except Exception: time.sleep(5)
        else:
            print(f"[mtp] ctx{ctx//1024}k {kv} server failed"); return None
        body = json.dumps({"messages":[{"role":"user","content":"Write a Python function that merges two sorted lists."}],
                           "max_tokens":512, "temperature":0}).encode()
        req = urllib.request.Request(f"http://localhost:{PORT}/v1/chat/completions", data=body,
                                     headers={"Content-Type":"application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            d = json.load(r)
        t = d.get("timings", {})
        res = {"prompt_tps": t.get("prompt_per_second"), "decode_tps": t.get("predicted_per_second"),
               "draft_n": t.get("draft_n"), "draft_acc": t.get("draft_n_accepted"),
               "vram": vram_used()}
        print(f"[mtp] ctx{ctx//1024}k {kv}: decode={res['decode_tps']} acc={res['draft_acc']}/{res['draft_n']} vram={res['vram']}")
        return res
    finally:
        p.terminate()
        try: p.wait(timeout=10)
        except Exception: p.kill()

if __name__ == "__main__":
    stop_gemma()
    try:
        sweep()
        for ctx in [32768, 65536]:
            for kv in ['q8_0', 'q4_0']:
                OUT[f"mtp_ctx{ctx//1024}k_{kv}"] = mtp_probe(ctx, kv)
    finally:
        start_gemma()
        with open("/home/caimlas/llm-benchmarks/results/bonsai2_3060_ctx_sweep.json","w") as f:
            json.dump(OUT, f, indent=1)
        print("[done] -> results/bonsai2_3060_ctx_sweep.json")
