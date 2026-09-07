#!/usr/bin/env python3
"""
ECC followup 2 - corrected:
- Nail [Sharp] / Ornith [Sharp]: 256-token chat probe with ORIGINAL server
  flags (upstream binary, cpu-moe, 262K ctx) - comparable to recorded
  decode_tps (24.3 / 25.1). llama-bench cannot express cpu-moe.
- Bonsai dspark: PrismML server WITHOUT explicit -ngl (let memory fit logic
  work), draft n=4 probe, comparable to recorded 36.3.
"""
import subprocess, json, time, os, re, urllib.request, http.client

LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PORT = 18096
BENCH_RESULTS = "/home/caimlas/llm-benchmarks/bench_results.json"
PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def server_probe(binary, model, gpu, extra, tag, max_tokens=256):
    cmd = [binary, "--model", model, "--flash-attn", "on", "--host", "127.0.0.1",
           "--port", str(PORT), "--parallel", "1", "--temp", "0.0",
           "-n", str(max_tokens), "--jinja"] + extra
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(os.path.join(LOGS, f"ecc3_{tag}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(240):
            time.sleep(2)
            if proc.poll() is not None:
                return {"error": "server died: " + open(logf.name).read()[-300:]}
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                break
            except Exception:
                pass
        else:
            proc.kill()
            return {"error": "timeout"}
        best = 0.0
        for _ in range(2):
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
            payload = json.dumps({"messages": [{"role": "user", "content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens": max_tokens, "temperature": 0.0, "stream": False})
            t0 = time.time()
            conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
            d = json.loads(conn.getresponse().read())
            dt = time.time() - t0
            toks = (d.get("usage") or {}).get("completion_tokens")
            conn.close()
            if not toks:
                return {"error": "no completion_tokens"}
            best = max(best, toks / dt)
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
    results = json.load(open(BENCH_RESULTS))
    orig = ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144",
            "--ubatch-size", "512", "--threads", "8", "--threads-batch", "8",
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]

    log("Nail [Sharp]: probe with original flags (cpu-moe, 262K)...")
    r = server_probe("/home/caimlas/git/llama.cpp/build/bin/llama-server",
                     os.path.join(LLMS, "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), 0,
                     orig, "nail_origcfg")
    results["Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL [Sharp] [ecc-off-origcfg-probe]"] = {
        "probe": r, "prior_decode_tps": 24.3, "gpu": "GPU0", "timestamp": TS}
    log(f"  {r} (prior 24.3)")

    log("Ornith [Sharp]: probe with original flags...")
    r = server_probe("/home/caimlas/git/llama.cpp/build/bin/llama-server",
                     os.path.join(LLMS, "Ornith-1.5-35B-Q4_K_M.gguf"), 0,
                     orig, "ornith_origcfg")
    results["Ornith-1.5-35B-A3B Q4_K_M [Sharp] [ecc-off-origcfg-probe]"] = {
        "probe": r, "prior_decode_tps": 25.1, "gpu": "GPU0", "timestamp": TS}
    log(f"  {r} (prior 25.1)")

    log("Bonsai dspark: PrismML, fit-friendly (no explicit ngl), draft n=4...")
    r = server_probe(PRISMML, os.path.join(LLMS, "Ternary-Bonsai-27B-Q2_0.gguf"), 0,
                     ["--ctx-size", "8192", "--ubatch-size", "512",
                      "--threads", "8", "--threads-batch", "8",
                      "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
                      "--spec-draft-model", os.path.join(LLMS, "Ternary-Bonsai-27B-dspark-Q4_1.gguf"),
                      "--spec-draft-n-max", "4"],
                     "bonsai_dspark", max_tokens=256)
    results["Ternary-Bonsai-27B Q2_0 (dspark) [ecc-off-dspark-probe]"] = {
        "probe": r, "prior_decode_tps": 36.3, "gpu": "GPU0", "timestamp": TS}
    log(f"  {r} (prior 36.3)")

    with open(BENCH_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    log("DONE")


if __name__ == "__main__":
    main()
