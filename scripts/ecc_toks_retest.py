#!/usr/bin/env python3
"""
tok/s re-tests for top-5 LCB + top-5 tau2 models (V100 ECC disabled).
Queue item: measure ECC-off impact on decode/prompt tok/s.

Protocol: llama-bench pp512/tg128 (-r 3, fa on, q8_0 kv, same as original
speed_norm) + 256-token chat decode probe. Results stored in
bench_results.json under "<file> [ecc-off]"; deltas vs recorded speed_norm
printed at the end. 3060-lane rows are controls (ECC change is V100-only).

V100 service caimlas-qwythos is stopped for the duration and restored after.
"""
import subprocess, json, time, os, re, urllib.request, shutil, sys

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LBENCH = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18096
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
BENCH_RESULTS = "/home/caimlas/llm-benchmarks/bench_results.json"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

DRAFT = "Ternary-Bonsai-27B-dspark-Q4_1.gguf"

# name -> (file, gpu, mode)  mode: plain | mtp3 | dspark
MODELS = [
    # top-5 LCB (tie at 92% -> 6)
    ("Qwythos-27B-v1 Q4_K_M",                       "Qwythos-27B-MTP-Q4_K_M.gguf", 0, "plain"),   # v1 non-MTP file deleted earlier? use MTP file only if v1 missing
    ("Qwythos-27B-MTP Q4_K_M",                      "Qwythos-27B-MTP-Q4_K_M.gguf", 0, "mtp3"),
    ("Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL [Sharp]",     "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", 0, "plain"),
    ("BTL-4 Q4_K_M",                                "badtheorylabs_BTL-4-Q4_K_M.gguf", 0, "plain"),
    ("Ornith-1.5-35B-A3B Q4_K_M [Sharp]",           "Ornith-1.5-35B-Q4_K_M.gguf", 0, "plain"),
    ("gemma-4-12B-it-QAT Q4_0 (3060 128K)",         "gemma-4-12B-it-QAT-Q4_0.gguf", 1, "plain"),
    # top-5 tau2 (tie at 0.50 -> 6)
    ("Ternary-Bonsai-27B Q2_0 (dspark)",            "Ternary-Bonsai-27B-Q2_0.gguf", 0, "dspark"),
    ("Carnice-V3 Q4_K_M",                           "Carnice-V3-Q4_K_M.gguf", 0, "plain"),
    ("Qwopus3.6-27B-v2-MTP Q4_K_M",                 "Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf", 0, "plain"),
    ("LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M",     "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf", 1, "plain"),
    ("gemma4-coding Q4_K_M",                        "gemma4-coding-Q4_K_M.gguf", 1, "plain"),
    ("DeepSeek-Coder-V2-Lite IQ4_XS",               "DeepSeek-Coder-V2-Lite-Instruct-IQ4_XS.gguf", 1, "plain"),
]

MTP3 = ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]
DSPARK = ["--spec-draft-model", os.path.join(LLMS_DIR, DRAFT), "--spec-draft-n-max", "4"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_progress():
    with open(PROGRESS_FILE) as f:
        return json.load(f)


def llama_bench(path, gpu, extra_name):
    cmd = ["/home/caimlas/git/llama.cpp/build/bin/llama-bench",
           "-m", path, "-fa", "on", "-ngl", "99",
           "-ctk", "q8_0", "-ctv", "q8_0", "-p", "512", "-n", "128",
           "-d", "8192", "-r", "3", "-t", "8"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    blog = os.path.join(LOGS, f"ecc_{extra_name}_llamabench.log")
    with open(blog, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=3600)
    txt = open(blog).read()
    ts = re.findall(r"\|\s*([0-9.]+)\s*±\s*[0-9.]+\s*\|", txt)
    return ({"pp512": float(ts[0]), "tg128": float(ts[1])} if len(ts) >= 2
            else {"error": txt[-200:]})


def chat_probe(path, gpu, extra, name):
    cmd = [BINARY, "-m", path, "--flash-attn", "on", "--host", "127.0.0.1",
           "--port", str(PORT), "--parallel", "1", "--temp", "0.0", "-n", "256",
           "--gpu-layers", "99", "--ctx-size", "8192", "--ubatch-size", "512",
           "--threads", "8", "--threads-batch", "8",
           "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja"] + extra
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    slog = os.path.join(LOGS, f"ecc_{re.sub(r'[^a-zA-Z0-9]+','_',name)[:40]}_server.log")
    logf = open(slog, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(150):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            return {"error": "server died: " + open(slog).read()[-200:]}
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            break
        except Exception:
            pass
    else:
        proc.kill(); logf.close()
        return {"error": "server timeout"}
    try:
        props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
        if path.split("/")[-1].replace(".gguf", "") not in props:
            proc.kill(); logf.close()
            return {"error": "wrong server on port"}
        import http.client
        best = 0.0
        for _ in range(2):
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
            payload = json.dumps({
                "messages": [{"role": "user", "content":
                    "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens": 256, "temperature": 0.0, "stream": False})
            t0 = time.time()
            conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
            d = json.loads(conn.getresponse().read())
            dt = time.time() - t0
            toks = (d.get("usage") or {}).get("completion_tokens")
            if not toks:
                conn.close(); proc.kill(); logf.close()
                return {"error": "no completion_tokens"}
            best = max(best, toks / dt)
            conn.close()
        return {"decode_tps_probe": round(best, 2)}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        logf.close()
        time.sleep(3)


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(3)
    results = {}
    if os.path.exists(BENCH_RESULTS):
        with open(BENCH_RESULTS) as f:
            results = json.load(f)
    progress = load_progress()

    try:
        for name, fname, gpu, mode in MODELS:
            if only and name not in only:
                continue
            path = os.path.join(LLMS_DIR, fname)
            if not os.path.exists(path):
                # try Qwythos v1 fallback (non-MTP file deleted?)
                if "Qwythos-27B-v1" in name:
                    alt = os.path.join(LLMS_DIR, "Qwythos-27B-v1-Q4_K_M.gguf")
                    if os.path.exists(alt):
                        path = alt
                    else:
                        log(f"SKIP {name}: no non-MTP v1 file on disk"); continue
                else:
                    log(f"SKIP {name}: {fname} missing"); continue

            log(f"=== {name} (GPU{gpu}, mode={mode}) ===")
            extra = MTP3 if mode == "mtp3" else DSPARK if mode == "dspark" else []
            bench = llama_bench(path, gpu, re.sub(r"[^a-zA-Z0-9]+", "_", name)[:40])
            probe = chat_probe(path, gpu, extra, name)

            # prior recorded numbers
            prior = {}
            for m in progress["models"]:
                if m["name"] == name:
                    prior = m.get("speed_norm") or {}
                    break
            entry = {"ecc_off": {**bench, **probe},
                     "prior_speed_norm": prior,
                     "gpu": f"GPU{gpu}", "mode": mode, "timestamp": TS}
            results[f"{fname} [ecc-off]"] = entry
            with open(BENCH_RESULTS, "w") as f:
                json.dump(results, f, indent=2)

            if "tg128" in bench and prior.get("tg128"):
                d = 100 * (bench["tg128"] - prior["tg128"]) / prior["tg128"]
                log(f"  tg128: {prior['tg128']} -> {bench['tg128']} ({d:+.1f}%)  "
                    f"pp512: {prior.get('pp512')} -> {bench.get('pp512')}")
            else:
                log(f"  bench={bench} probe={probe} prior={prior}")
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "caimlas-qwythos"], capture_output=True)
        time.sleep(3)

    log("=== ECC tok/s re-tests COMPLETE ===")


if __name__ == "__main__":
    main()
