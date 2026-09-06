#!/usr/bin/env python3
"""
Qwen3.8-27B GGUF fallback benchmark on the V100 (GPU 0), replacing the
unrunnable EXL3 quants (EXL3 kernels are Ampere+ only).

Per model (UD-IQ3_S ~3.5bpw, UD-Q4_K_S ~4.5bpw):
  1. llama-bench speed run (pp512 / tg128, 8K ctx, fa on, q8_0 KV)
  2. llama-server smoke + BenchKit sanity:25 gate (existing script, port 18052)
  3. LiveCodeBench 75 problems via lcb_runner (port 18099)

Results appended to /tmp/coding-bench/progress.json under qwen38_v100_gguf.
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
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs_qwen38gguf")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
os.makedirs(LOGS, exist_ok=True)

MODELS = [
    {"name": "Qwen3.8-27B UD-IQ3_S (GGUF, ~3.5bpw)",
     "file": "Qwen3.8-27B-UD-IQ3_S.gguf", "bpw": 3.52,
     "lcb_model": "local/qwen38-27b-ud-iq3s"},
    {"name": "Qwen3.8-27B UD-Q4_K_S (GGUF, ~4.5bpw)",
     "file": "Qwen3.8-27B-UD-Q4_K_S.gguf", "bpw": 4.49,
     "lcb_model": "local/qwen38-27b-ud-q4ks"},
]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)

def get_entry(progress, name):
    for m in progress.get("models", []):
        if m["name"] == name:
            return m
    entry = {"name": name, "gpu": "V100", "engine": "llama.cpp-upstream"}
    progress.setdefault("models", []).append(entry)
    return entry

def run_speed(model):
    """llama-bench: pp512/tg128, 8K depth, fa, q8_0 kv, V100 only."""
    path = os.path.join(LLMS_DIR, model["file"])
    safe = model["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("~", "")
    logpath = os.path.join(LOGS, f"{safe}_llamabench.log")
    cmd = [LBENCH, "-m", path, "-fa", "on", "-ngl", "99", "-ctk", "q8_0", "-ctv", "q8_0",
           "-p", "512", "-n", "128", "-d", "8192", "-r", "2", "-t", "8"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"  llama-bench starting...")
    with open(logpath, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=1200)
    if r.returncode != 0:
        with open(logpath) as f:
            return {"error": "llama-bench failed: " + f.read()[-300:]}
    with open(logpath) as f:
        txt = f.read()
    out = {}
    # parse the markdown table: t/s cells look like "578.80 ± 12.05"
    ts = re.findall(r"\|\s*([0-9.]+)\s*±\s*[0-9.]+\s*\|", txt)
    if len(ts) >= 2:
        out["pp512"] = float(ts[0])
        out["tg128"] = float(ts[1])
    elif len(ts) == 1:
        out["tg128"] = float(ts[0])
    mbpw = re.search(r"([0-9.]+)\s*bpw", txt)
    if mbpw:
        out["actual_bpw"] = float(mbpw.group(1))
    msize = re.search(r"\|\s+([0-9.]+)\s+GiB\s+\|", txt)
    if msize:
        out["size_GiB"] = float(msize.group(1))
    # vram from log
    m = re.search(r" CUDA0 model buffer size\s*=\s*([0-9.]+) MiB", txt)
    if m:
        out["weights_MiB"] = float(m.group(1))
    return out

def run_gate(model):
    """BenchKit sanity:25 gate using the existing house script."""
    safe = model["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("~", "")
    logpath = os.path.join(LOGS, f"{safe}_gate.log")
    cmd = [BENCH_PY, GATE, "--model-file", model["file"],
           "--name", model["name"], "--gpu", "0", "--benchmarks", "sanity:25"]
    print("  BenchKit gate starting (~10 min)...")
    with open(logpath, "w") as lf:
        r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
    with open(logpath) as f:
        txt = f.read()
    out = {}
    m = re.search(r"[Ss]core[:\s]+([0-9.]+)%?", txt)
    frac = re.findall(r"(\d+)\s*/\s*25", txt)
    if frac:
        out["sanity_passed"] = int(frac[-1])
        out["sanity_pct"] = round(100 * int(frac[-1]) / 25, 1)
    if m:
        out["sanity_score_raw"] = m.group(1)
    if r.returncode != 0 and not out:
        out["error"] = "gate failed: " + txt[-300:]
    return out

def run_lcb(model):
    """LiveCodeBench 75 via lcb_runner against a server we start here."""
    path = os.path.join(LLMS_DIR, model["file"])
    safe = model["name"].replace(" ", "_").replace("(", "").replace(")", "").replace(",", "").replace("~", "")
    cmd = [BINARY, "--model", path, "--device", "CUDA0", "--flash-attn", "on",
           "--batch-size", "2048", "--host", "127.0.0.1", "--port", str(PORT),
           "--parallel", "1", "--temp", "0.0", "-n", "4096",
           "--gpu-layers", "99", "--ctx-size", "65536", "--ubatch-size", "512",
           "--threads", "8", "--threads-batch", "8",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    srvlog = os.path.join(LOGS, f"{safe}_lcb_server.log")
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
        if proc.poll() is not None:
            logf.close()
            with open(srvlog) as f:
                return {"error": "LCB server died: " + f.read()[-300:]}
    else:
        proc.kill()
        return {"error": "LCB server timeout"}

    # smoke: one tiny completion
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps({"messages": [{"role": "user", "content": "Say OK."}],
                             "max_tokens": 16, "temperature": 0.0}).encode(),
            headers={"Content-Type": "application/json"})
        resp = json.loads(urllib.request.urlopen(req, timeout=120).read())
        smoke = resp["choices"][0]["message"].get("content", "")[:80]
        print(f"  smoke: {smoke!r}")
    except Exception as e:
        proc.terminate()
        return {"error": f"smoke failed: {e}"}

    lcb_log = os.path.join(LOGS, f"{safe}_lcb.log")
    lcmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
            "--model", model["lcb_model"],
            "--scenario", "codegeneration",
            "--release_version", "release_latest",
            "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
            "--num_problems", "75", "--openai_timeout", "300",
            "--evaluate", "--use_cache"]
    lenv = dict(os.environ)
    lenv["OPENAI_KEY"] = "none"
    lenv["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    lenv["HF_ALLOW_CODE_EVAL"] = "1"
    lenv["LCB_DISABLE_THINKING"] = "1"
    print("  LCB 75 running (long)...")
    with open(lcb_log, "w") as lf:
        r = subprocess.run(lcmd, stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR,
                           env=lenv, timeout=4 * 3600)
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    logf.close()

    out = {}
    with open(lcb_log) as f:
        txt = f.read()
    m = re.search(r"pass@1[^\d]*([0-9.]+)", txt)
    if m:
        v = float(m.group(1))
        out["lcb_pass1"] = v if v <= 1.0 else v / 100.0
    else:
        # lcb_runner prints the bare pass@1 float as the final line
        m2 = re.search(r"\n([01]\.[0-9]+)\s*$", txt)
        if m2:
            out["lcb_pass1"] = float(m2.group(1))
    if r.returncode != 0 and "lcb_pass1" not in out:
        out["error"] = "LCB failed: " + txt[-300:]
    return out

def vram_used():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] + " MiB"

if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    progress = load_progress()
    for model in MODELS:
        if only and only not in model["file"]:
            continue
        print("=" * 60)
        print(model["name"])
        print("=" * 60)
        entry = get_entry(progress, model["name"])
        entry["bpw"] = model["bpw"]

        if "speed" not in entry:
            sp = run_speed(model)
            entry["speed"] = sp
            save_progress(progress)
            print(f"  speed: {sp}")
        else:
            print(f"  speed (cached): {entry['speed']}")

        if "benchkit" not in entry:
            bk = run_gate(model)
            entry["benchkit"] = bk
            save_progress(progress)
            print(f"  gate: {bk}")
        else:
            print(f"  gate (cached): {entry['benchkit']}")

        if "lcb" not in entry:
            lc = run_lcb(model)
            entry["lcb"] = lc
            save_progress(progress)
            print(f"  lcb: {lc}")
        else:
            print(f"  lcb (cached): {entry['lcb']}")

        save_progress(progress)

    print("\n" + "=" * 70)
    print("SUMMARY (V100, llama.cpp, fa=on, ctx 8K probe / 64K LCB)")
    print("=" * 70)
    print(f"{'Model':<38} {'bpw':>5} {'pp512':>8} {'tg128':>8} {'sanity%':>8} {'LCB':>7}")
    for m in progress.get("models", []):
        if "qwen38" not in m["name"].lower() and "Qwen3.8" not in m["name"]:
            continue
        sp = m.get("speed", {})
        bk = m.get("benchkit", {})
        lc = m.get("lcb", {})
        print(f"{m['name']:<38} {m.get('bpw',0):>5} "
              f"{sp.get('pp512','-'):>8} {sp.get('tg128','-'):>8} "
              f"{bk.get('sanity_pct','-'):>8} {lc.get('lcb_pass1','-'):>7}")
