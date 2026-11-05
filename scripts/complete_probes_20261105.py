#!/usr/bin/env python3
"""Cross-GPU probe completion for intake_20261105. Runs each missing probe with the
target card's prod idle (one model at a time, no concurrent double-load), then restores
both prod units in finally. Batch by card to minimize prod downtime:
  V100 round (stop gemma-v100): Qwopus prefill+sanity, LFM3 base+unc V100 pp512/tg128
  3060 round (stop bonsai-3060): Coding-Monkey 3060 pp512/tg128
"""
import json, os, shutil, subprocess, sys, time

BIN = "/var/tmp/llms/llama-lfm3-build/build/bin"
BENCH = os.path.join(BIN, "llama-bench")
RES = "/home/caimlas/llm-benchmarks/results/intake_20261105-061928.json"
OUT = json.load(open(RES))

def sh(cmd, timeout=1800):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def bench_probe(gpu, model_file):
    """pp512 + tg128 via llama-bench on one card, card otherwise idle."""
    cmd = f"CUDA_VISIBLE_DEVICES={gpu} {BENCH} -m {model_file} -p 512 -n 128 -fa 1 -t 8 -o json 2>/dev/null"
    r = sh(cmd, timeout=1800)
    try:
        data = json.loads(r.stdout)
        if isinstance(data, list):
            pp = tg = None
            for row in data:
                if row.get("n_prompt") == 512 and pp is None:
                    pp = round(row.get("avg_ts", 0), 2)
                if row.get("n_gen") == 128 and tg is None:
                    tg = round(row.get("avg_ts", 0), 2)
            return {"prompt_tps": pp, "decode_tps": tg, "via": f"llama-bench-{BENCH.split('/')[-1]}"}
    except Exception as e:
        return {"error": str(e), "raw": r.stdout[-200:] + r.stderr[-200:]}
    return {"error": "no results", "raw": r.stdout[-200:]}

def bench_sanity(gpu, model_file, name):
    gate = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
    cmd = f"{sys.executable} {gate} --model-file {model_file} --name '{name}' --gpu {gpu} --benchmarks sanity:25 --binary {os.path.join(BIN,'llama-server')}"
    r = sh(cmd, timeout=3600)
    out = r.stdout + r.stderr
    import re
    m2 = re.search(r'([\d.]+)%\s*\((\d+)/(\d+)\)', out)
    if m2:
        return {"sanity_pct": float(m2.group(1)), "passed": int(m2.group(2)), "total": int(m2.group(3))}
    return {"error": out[-400:] if out else "empty"}

def stop(unit): return sh(f"sudo -n systemctl stop {unit}")
def start(unit): return sh(f"sudo -n systemctl start {unit}")
def wait200(port, tries=40):
    for _ in range(tries):
        time.sleep(5)
        try:
            r = sh(f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/v1/models", timeout=15)
            if r.stdout.strip() == "200": return True
        except Exception: pass
    return False

Q = "/var/tmp/llms/qwopus-coder-flash/Qwopus3.8-27B-Coder-Flash-Q4_K_M.gguf"
L_B, L_U = "/var/tmp/llms/lfm3-base/LFM3-1.7B-Q4_K_M.gguf", "/var/tmp/llms/lfm3-uncensored/LFM3-1.7B-Uncensored-AD-Q6_K.gguf"
CM = "/var/tmp/llms/coding-monkey-gemma/gemma-coder-Q4_K_M.gguf"

KW = ["Qwopus3.8-27B-Coder-Flash Q4_K_M (V100)", "LFM3-1.7B Q4_K_M (3060)",
      "LFM3-1.7B-Uncensored AD-Q6_K (3060)", "Coding-Monkey-Gemma gemma-coder Q4_K_M (V100)"]

try:
    # ---- V100 round ----
    print("== V100 round: stop gemma-v100 ==", flush=True)
    stop("caimlas-gemma-4-26b-v100")
    time.sleep(5)
    # Qwopus prefill (was None) + decode
    print("-- Qwopus V100 pp512/tg128 --", flush=True)
    OUT[KW[0]]["prefill"] = bench_probe(0, Q)
    print("  ", OUT[KW[0]]["prefill"], flush=True)
    # Qwopus sanity (was gate-0 OOM)
    print("-- Qwopus V100 sanity --", flush=True)
    OUT[KW[0]]["sanity"] = bench_sanity(0, Q, "Qwopus3.8-27B-Coder-Flash Q4_K_M (V100)")
    print("  ", OUT[KW[0]]["sanity"], flush=True)
    # LFM3 base V100 tok
    print("-- LFM3 base V100 pp512/tg128 --", flush=True)
    OUT[KW[1]]["v100_probe"] = bench_probe(0, L_B)
    print("  ", OUT[KW[1]]["v100_probe"], flush=True)
    # LFM3 unc V100 tok
    print("-- LFM3 unc V100 pp512/tg128 --", flush=True)
    OUT[KW[2]]["v100_probe"] = bench_probe(0, L_U)
    print("  ", OUT[KW[2]]["v100_probe"], flush=True)

    # ---- 3060 round ----
    print("== 3060 round: stop bonsai-3060 ==", flush=True)
    stop("caimlas-bonsai-3060")
    time.sleep(5)
    # Coding-Monkey 3060 probe (user: "prefill/toks for 3060 for comparison with a subagent")
    print("-- Coding-Monkey 3060 pp512/tg128 --", flush=True)
    OUT[KW[3]]["gpu3060_probe"] = bench_probe(1, CM)
    print("  ", OUT[KW[3]]["gpu3060_probe"], flush=True)

finally:
    print("== restoring prod ==", flush=True)
    start("caimlas-gemma-4-26b-v100")
    start("caimlas-bonsai-3060")
    w1 = wait200(8082); w2 = wait200(8083)
    print(f"gemma-v100 200={w1}  bonsai-3060 200={w2}", flush=True)
    json.dump(OUT, open(RES, "w"), indent=1, default=str)
    print("saved ->", RES, flush=True)
