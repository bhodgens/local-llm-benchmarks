#!/usr/bin/env python3
"""
BTL-4 Q4_K_M Phase 2 on V100 (GPU0) - same harness as BTL-4 IQ2_XXS run.
Speed probe + BenchKit gate + LCB 75 (thinking suppressed via LCB_DISABLE_THINKING).
"""
import subprocess, json, time, os, re, urllib.request

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
MODEL = "/home/files/llms/badtheorylabs_BTL-4-Q4_K_M.gguf"
GPU_ID = "0"  # V100
PORT = 18099
LOGS = "/tmp/coding-bench/logs_btl4q4km"
os.makedirs(LOGS, exist_ok=True)
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
GATE = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
NAME = "BTL-4 Q4_K_M"

def load_progress():
    with open(PROGRESS_FILE) as f: return json.load(f)

def save_progress(p):
    with open(PROGRESS_FILE + ".tmp", "w") as f: json.dump(p, f, indent=2)
    os.replace(PROGRESS_FILE + ".tmp", PROGRESS_FILE)

subprocess.run(["pkill", "-f", "port 18099"], capture_output=True)
time.sleep(3)

print("Starting BTL-4 Q4_K_M on V100 (GPU0)...")
cmd = [BINARY, "--model", MODEL, "--flash-attn", "on", "--gpu-layers", "99",
       "--ctx-size", "32768", "--batch-size", "2048", "--ubatch-size", "512",
       "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "1", "--temp", "0.0",
       "-n", "4096", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
       "--threads", "8", "--threads-batch", "8", "--jinja",
       "--reasoning", "off", "--reasoning-format", "deepseek"]
env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = GPU_ID
logf = open(os.path.join(LOGS, "server.log"), "w")
proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
up = False
for _ in range(240):
    time.sleep(2)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
        up = True; break
    except Exception:
        if proc.poll() is not None: break
if not up:
    print("FATAL: server failed to start")
    print(open(os.path.join(LOGS, "server.log")).read()[-500:])
    raise SystemExit(1)
print("server up")

def speed_probe():
    prompt = ("Write a detailed essay about the history of computing, "
              "from Babbage to modern GPUs. Include key milestones.")
    # prefill: 512-token prompt
    long_prompt = prompt + " " + "Consider architecture, instruction sets, memory hierarchies, and parallelism. " * 8
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps({"messages":[{"role":"user","content":long_prompt}],
                         "max_tokens":1,"temperature":0.0}).encode(),
        headers={"Content-Type":"application/json"})
    t0=time.time(); d=json.loads(urllib.request.urlopen(req,timeout=120).read()); pt=time.time()-t0
    pp = d["usage"]["prompt_tokens"]/pt
    # decode: 256 tokens
    best = 0.0
    for _ in range(2):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps({"messages":[{"role":"user","content":prompt}],
                             "max_tokens":256,"temperature":0.0}).encode(),
            headers={"Content-Type":"application/json"})
        t0=time.time(); d=json.loads(urllib.request.urlopen(req,timeout=300).read()); dt=time.time()-t0
        best = max(best, d["usage"]["completion_tokens"]/dt)
    return {"pp512": round(pp,2), "tg128": round(best,2)}

print("speed..."); sp = speed_probe()
progress = load_progress()
entry = next((m for m in progress["models"] if m["name"]==NAME), None)
if entry is None:
    entry = {"name": NAME, "file": "badtheorylabs_BTL-4-Q4_K_M.gguf", "category": "35B-MoE",
             "gpu": "V100", "engine": "llama.cpp", "thinking": False, "template": "stock"}
    progress["models"].append(entry)
entry["prompt_tps"]=sp["pp512"]; entry["decode_tps"]=sp["tg128"]; entry["speed_norm"]=sp
save_progress(progress); print(" speed:", sp)

print("gate...")
# gate uses its own server mgmt; kill ours first
proc.terminate()
try: proc.wait(timeout=10)
except Exception: proc.kill()
time.sleep(3)
glog = os.path.join(LOGS, "gate.log")
with open(glog, "w") as lf:
    subprocess.run([BENCH_PY, GATE, "--model-file", "badtheorylabs_BTL-4-Q4_K_M.gguf",
                    "--name", NAME, "--gpu", GPU_ID, "--benchmarks", "sanity:25"],
                   stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
gtxt = open(glog).read()
frac = re.findall(r"(\d+)\s*/\s*25", gtxt)
gate = {"sanity_passed": int(frac[-1]), "sanity_pct": round(100*int(frac[-1])/25,1)} if frac else {"error": gtxt[-200:]}
entry["benchkit"] = {"sanity": {"score_pct": gate.get("sanity_pct"), "passed": gate.get("sanity_passed"), "total": 25}}
save_progress(progress); print(" gate:", gate)

print("lcb...")
lcb_srv = [BINARY, "--model", MODEL, "--flash-attn", "on", "--gpu-layers", "99",
           "--ctx-size", "32768", "--batch-size", "2048", "--ubatch-size", "512",
           "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "1", "--temp", "0.0",
           "-n", "4096", "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
           "--threads", "8", "--threads-batch", "8", "--jinja",
           "--reasoning", "off", "--reasoning-format", "deepseek"]
logf2 = open(os.path.join(LOGS, "lcb_server.log"), "w")
proc2 = subprocess.Popen(lcb_srv, env=env, stdout=logf2, stderr=subprocess.STDOUT, text=True)
for _ in range(240):
    time.sleep(2)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3); break
    except Exception: pass
lcmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", "local/btl4-q4km",
        "--scenario", "codegeneration", "--release_version", "release_latest",
        "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
        "--num_problems", "75", "--openai_timeout", "600", "--evaluate"]
lenv = dict(os.environ)
lenv.update({"OPENAI_KEY":"none","OPENAI_BASE_URL":f"http://127.0.0.1:{PORT}/v1",
             "HF_ALLOW_CODE_EVAL":"1","LCB_DISABLE_THINKING":"1"})
lcb_log = os.path.join(LOGS, "lcb.log")
with open(lcb_log, "w") as lf:
    subprocess.run(lcmd, stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR, env=lenv, timeout=5*3600)
proc2.terminate()
try: proc2.wait(timeout=10)
except Exception: proc2.kill()
txt = open(lcb_log).read()
m = re.search(r"pass@1[^\d]*([0-9.]+)", txt) or re.search(r"\n([01]\.[0-9]+)\s*$", txt)
lcb = (float(m.group(1)) if float(m.group(1))<=1.0 else float(m.group(1))/100.0) if m else None
entry["livecodebench"] = {"pass_at_1": lcb, "exit_code": 0 if lcb is not None else 1}
save_progress(progress)
print(" lcb:", lcb)
print("ALL DONE")
