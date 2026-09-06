#!/usr/bin/env python3
"""
gemma4-coding Q4_K_M (fable5-composer2.5) on 3060 via llama.cpp:
speed (llama-bench) + BenchKit sanity:25 + LCB 75. Appends to progress.json.
"""
import subprocess, json, time, os, sys, urllib.request, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LBENCH = "/home/caimlas/git/llama.cpp/build/bin/llama-bench"
LLMS_DIR = "/home/files/llms"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
GATE = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
PORT = 18099
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
LOGS = "/tmp/coding-bench/logs_gemma4coding"
os.makedirs(LOGS, exist_ok=True)

NAME = "gemma4-coding fable5-composer2.5 Q4_K_M"
FILE = "gemma4-coding-Q4_K_M.gguf"

def load_progress():
    with open(PROGRESS_FILE) as f:
        return json.load(f)

def save_progress(p):
    with open(PROGRESS_FILE + ".tmp", "w") as f:
        json.dump(p, f, indent=2)
    os.replace(PROGRESS_FILE + ".tmp", PROGRESS_FILE)

def run_speed():
    path = os.path.join(LLMS_DIR, FILE)
    cmd = [LBENCH, "-m", path, "-fa", "on", "-ngl", "99", "-ctk", "q8_0", "-ctv", "q8_0",
           "-p", "512", "-n", "128", "-d", "8192", "-r", "2", "-t", "8"]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "1"
    log = os.path.join(LOGS, "llamabench.log")
    with open(log, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=1200)
    txt = open(log).read()
    ts = re.findall(r"\|\s*([0-9.]+)\s*±\s*[0-9.]+\s*\|", txt)
    out = {}
    if len(ts) >= 2:
        out = {"pp512": float(ts[0]), "tg128": float(ts[1])}
    m = re.search(r"([0-9.]+)\s*bpw", txt)
    if m: out["actual_bpw"] = float(m.group(1))
    return out

def run_gate():
    log = os.path.join(LOGS, "gate.log")
    cmd = [BENCH_PY, GATE, "--model-file", FILE, "--name", NAME, "--gpu", "1",
           "--benchmarks", "sanity:25"]
    with open(log, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    txt = open(log).read()
    frac = re.findall(r"(\d+)\s*/\s*25", txt)
    return {"sanity_passed": int(frac[-1]), "sanity_pct": round(100*int(frac[-1])/25, 1)} if frac else {"error": txt[-200:]}

def run_lcb():
    path = os.path.join(LLMS_DIR, FILE)
    cmd = [BINARY, "--model", path, "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "1", "--temp", "0.0",
           "-n", "4096", "--gpu-layers", "99", "--ctx-size", "32768", "--ubatch-size", "512",
           "--threads", "8", "--threads-batch", "8",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja"]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "1"
    srvlog = os.path.join(LOGS, "lcb_server.log")
    logf = open(srvlog, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                break
        except Exception:
            pass
    else:
        return {"error": "server timeout"}
    lcmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
            "--model", "local/gemma4-coding-q4",
            "--scenario", "codegeneration", "--release_version", "release_latest",
            "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
            "--num_problems", "75", "--openai_timeout", "300",
            "--evaluate", "--use_cache"]
    lenv = dict(os.environ)
    lenv.update({"OPENAI_KEY": "none", "OPENAI_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
                 "HF_ALLOW_CODE_EVAL": "1", "LCB_DISABLE_THINKING": "1"})
    lcb_log = os.path.join(LOGS, "lcb.log")
    with open(lcb_log, "w") as lf:
        r = subprocess.run(lcmd, stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR,
                           env=lenv, timeout=5*3600)
    proc.terminate()
    try: proc.wait(timeout=10)
    except Exception: proc.kill()
    txt = open(lcb_log).read()
    out = {}
    m = re.search(r"pass@1[^\d]*([0-9.]+)", txt)
    if not m:
        m = re.search(r"\n([01]\.[0-9]+)\s*$", txt)
    if m:
        v = float(m.group(1))
        out["lcb_pass1"] = v if v <= 1.0 else v/100.0
    if not out:
        out = {"error": txt[-200:]}
    return out

progress = load_progress()
entry = next((m for m in progress["models"] if m["name"] == NAME), None)
if entry is None:
    entry = {"name": NAME, "file": FILE, "category": "12-14B", "gpu": "3060",
             "engine": "llama.cpp"}
    progress["models"].append(entry)

print("speed..."); entry["speed_norm"] = run_speed()
e = entry["speed_norm"]
entry["decode_tps"] = e.get("tg128"); entry["prompt_tps"] = e.get("pp512")
save_progress(progress); print(" speed:", e)

print("gate..."); bk = run_gate()
entry["benchkit"] = {"sanity": {"score_pct": bk.get("sanity_pct"),
                                "passed": bk.get("sanity_passed"), "total": 25}}
save_progress(progress); print(" gate:", bk)

print("lcb..."); lc = run_lcb()
entry["livecodebench"] = {"pass_at_1": lc.get("lcb_pass1"), "exit_code": 0 if lc.get("lcb_pass1") else 1}
save_progress(progress); print(" lcb:", lc)
print("ALL DONE")
