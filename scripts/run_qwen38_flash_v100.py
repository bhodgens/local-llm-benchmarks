#!/usr/bin/env python3
"""
Qwopus3.8-27B-Flash MTP Q4_K_M on V100 - production-optimal:
speed (llama-bench) + MTP n3 decode+acceptance + BenchKit sanity:25 + LCB 75.
"""
import subprocess, json, time, os, re, urllib.request

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LBENCH = "/home/caimlas/git/llama.cpp/build/bin/llama-bench"
LLMS_DIR = "/home/files/llms"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
GATE = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
PORT = 18099
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
LOGS = "/tmp/coding-bench/logs_qwopusflash"
os.makedirs(LOGS, exist_ok=True)

NAME = "Qwopus3.8-27B-Flash MTP Q4_K_M"
FILE = "Qwopus3.8-27B-Flash-MTP-Q4_K_M.gguf"
SRV_BASE = ["-m", os.path.join(LLMS_DIR, FILE), "--flash-attn", "on",
            "--host", "127.0.0.1", "--port", str(PORT), "--parallel", "1", "--temp", "0.0",
            "-n", "4096", "--gpu-layers", "99", "--ctx-size", "65536", "--ubatch-size", "512",
            "--threads", "8", "--threads-batch", "8",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja"]

def load_progress():
    with open(PROGRESS_FILE) as f: return json.load(f)

def save_progress(p):
    with open(PROGRESS_FILE + ".tmp", "w") as f: json.dump(p, f, indent=2)
    os.replace(PROGRESS_FILE + ".tmp", PROGRESS_FILE)

def run_speed():
    cmd = [LBENCH, "-m", os.path.join(LLMS_DIR, FILE), "-fa", "on", "-ngl", "99",
           "-ctk", "q8_0", "-ctv", "q8_0", "-p", "512", "-n", "128", "-d", "8192",
           "-r", "2", "-t", "8"]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    log = os.path.join(LOGS, "llamabench.log")
    with open(log, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=1800)
    txt = open(log).read()
    ts = re.findall(r"\|\s*([0-9.]+)\s*±\s*[0-9.]+\s*\|", txt)
    return {"pp512": float(ts[0]), "tg128": float(ts[1])} if len(ts) >= 2 else {"error": txt[-200:]}

def mtp_probe():
    """MTP n=3 decode t/s + acceptance on a 256-token gen (no-spec skipped; baseline known 35.6)."""
    cmd = [BINARY] + SRV_BASE + ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(LOGS, "mtp_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(240):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception: pass
    else:
        proc.kill(); return {"error": "server fail"}
    prompt = ("Write a detailed essay about the history of computing, "
              "from Babbage to modern GPUs. Include key milestones.")
    d = None
    for _ in range(2):
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps({"messages":[{"role":"user","content":prompt}],
                             "max_tokens":256,"temperature":0.0}).encode(),
            headers={"Content-Type":"application/json"})
        t0 = time.time()
        d = json.loads(urllib.request.urlopen(req, timeout=300).read())
        dt = time.time() - t0
    ct = d["usage"]["completion_tokens"]
    res = {"MTP-n3": {"decode_tps": round(ct/dt, 2), "gen_tokens": ct}}
    proc.terminate()
    try: proc.wait(timeout=10)
    except Exception: proc.kill()
    time.sleep(3)
    txt = open(os.path.join(LOGS, "mtp_server.log")).read()
    accs = re.findall(r"accepted = ([0-9.]+)", txt)
    if accs:
        res["MTP-n3"]["acceptance_avg"] = round(sum(float(a) for a in accs)/len(accs), 3)
    return res

def run_gate():
    log = os.path.join(LOGS, "gate.log")
    cmd = [BENCH_PY, GATE, "--model-file", FILE, "--name", NAME, "--gpu", "0",
           "--mtp", "--benchmarks", "sanity:25"]
    with open(log, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    txt = open(log).read()
    frac = re.findall(r"(\d+)\s*/\s*25", txt)
    return {"sanity_passed": int(frac[-1]), "sanity_pct": round(100*int(frac[-1])/25, 1)} if frac else {"error": txt[-200:]}

def run_lcb():
    cmd = [BINARY] + SRV_BASE + ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(LOGS, "lcb_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(240):
        time.sleep(2)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3); break
        except Exception: pass
    else:
        return {"error": "server timeout"}
    lcmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
            "--model", "local/qwopus38-flash-q4km-mtp",
            "--scenario", "codegeneration", "--release_version", "release_latest",
            "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
            "--num_problems", "75", "--openai_timeout", "600",
            "--evaluate"]
    lenv = dict(os.environ)
    lenv.update({"OPENAI_KEY": "none", "OPENAI_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
                 "HF_ALLOW_CODE_EVAL": "1", "LCB_DISABLE_THINKING": "1"})
    lcb_log = os.path.join(LOGS, "lcb.log")
    with open(lcb_log, "w") as lf:
        subprocess.run(lcmd, stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR,
                       env=lenv, timeout=5*3600)
    proc.terminate()
    try: proc.wait(timeout=10)
    except Exception: proc.kill()
    txt = open(lcb_log).read()
    m = re.search(r"pass@1[^\d]*([0-9.]+)", txt) or re.search(r"\n([01]\.[0-9]+)\s*$", txt)
    if m:
        v = float(m.group(1))
        return {"lcb_pass1": v if v <= 1.0 else v/100.0}
    return {"error": txt[-200:]}

progress = load_progress()
entry = next((m for m in progress["models"] if m["name"] == NAME), None)
if entry is None:
    entry = {"name": NAME, "file": FILE, "category": "27B Dense", "gpu": "V100",
             "engine": "llama.cpp", "thinking": True, "mtp_enabled": True, "template": "stock"}
    progress["models"].append(entry)

print("speed..."); entry["speed_norm"] = run_speed()
s = entry["speed_norm"]
entry["decode_tps"] = s.get("tg128"); entry["prompt_tps"] = s.get("pp512")
save_progress(progress); print(" speed:", s)

print("mtp probe..."); entry["mtp_modes"] = mtp_probe()
mp = entry["mtp_modes"].get("MTP-n3", {})
if mp.get("acceptance_avg") is not None:
    entry["mtp_acceptance"] = {"MTP-n3": {"decode_tps": mp.get("decode_tps"),
                                          "acceptance_avg": mp.get("acceptance_avg")}}
save_progress(progress); print(" mtp:", entry["mtp_modes"])

print("gate..."); bk = run_gate()
entry["benchkit"] = {"sanity": {"score_pct": bk.get("sanity_pct"),
                                "passed": bk.get("sanity_passed"), "total": 25}}
save_progress(progress); print(" gate:", bk)

print("lcb..."); lc = run_lcb()
entry["livecodebench"] = {"pass_at_1": lc.get("lcb_pass1"),
                          "exit_code": 0 if lc.get("lcb_pass1") is not None else 1}
save_progress(progress); print(" lcb:", lc)
print("ALL DONE")
