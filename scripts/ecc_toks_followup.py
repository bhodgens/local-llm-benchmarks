#!/usr/bin/env python3
"""
ECC tok/s follow-ups: the three models whose first-pass measurements were
config-mismatched or load-failed.

1. Nail [Sharp] + Ornith-1.5 [Sharp]: rerun llama-bench in their ORIGINAL
   serving configs (262K ctx, cpu-moe, sharp-template server variant for
   Nail) so the delta vs recorded speed_norm is apples-to-apples.
2. Ternary-Bonsai-27B Q2_0 dspark: load via the PrismML fork (upstream
   llama.cpp cannot read Q2_0_g128 ternary tensors).
Results appended to bench_results.json with suffix [ecc-off-origcfg].
"""
import subprocess, json, time, os, re, urllib.request, sys

LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PORT = 18096
BENCH_RESULTS = "/home/caimlas/llm-benchmarks/bench_results.json"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
LBENCH_PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-bench"
LBENCH = "/home/caimlas/git/llama.cpp/build/bin/llama-bench"
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

NAIL_ORIG = ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144",
             "--ubatch-size", "512", "--threads", "8", "--threads-batch", "8",
             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def bench(binary, model, gpu, extra=None):
    cmd = [binary if binary.endswith("llama-bench") else binary,
           "-m", model, "-fa", "on", "-ngl", "99",
           "-ctk", "q8_0", "-ctv", "q8_0", "-p", "512", "-n", "128",
           "-d", "8192", "-r", "3", "-t", "8"] + (extra or [])
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    name = re.sub(r"[^a-zA-Z0-9]+", "_", model.split("/")[-1])[:40]
    with open(os.path.join(LOGS, f"ecc2_{name}.log"), "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=3600)
    txt = open(os.path.join(LOGS, f"ecc2_{name}.log")).read()
    ts = re.findall(r"\|\s*([0-9.]+)\s*±\s*[0-9.]+\s*\|", txt)
    return {"pp512": float(ts[0]), "tg128": float(ts[1])} if len(ts) >= 2 else {"error": txt[-200:]}


def main():
    results = json.load(open(BENCH_RESULTS))

    # 1. Nail original config (upstream binary, cpu-moe, 262K)
    log("Nail [Sharp] original-config bench...")
    r = bench(LBENCH, os.path.join(LLMS, "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"), 0, NAIL_ORIG)
    results["Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL [ecc-off-origcfg]"] = {
        "ecc_off_origcfg": r, "gpu": "GPU0", "note": "cpu-moe 262K ctx, matches original run config", "timestamp": TS}
    log(f"  {r}")

    # 2. Ornith original config
    log("Ornith [Sharp] original-config bench...")
    r = bench(LBENCH, os.path.join(LLMS, "Ornith-1.5-35B-Q4_K_M.gguf"), 0, NAIL_ORIG)
    results["Ornith-1.5-35B-A3B Q4_K_M [Sharp] [ecc-off-origcfg]"] = {
        "ecc_off_origcfg": r, "gpu": "GPU0", "note": "cpu-moe 262K ctx, matches original run config", "timestamp": TS}
    log(f"  {r}")

    # 3. Bonsai dspark via PrismML (llama-bench has no draft-model support for
    #    speculative pairs; measure the base model alone on PrismML, which is
    #    what the original speed_norm 36.3 was NOT - it was the dspark server
    #    decode probe. So record both: bench base, and probe dspark via server.)
    log("Bonsai Q2_0 via PrismML llama-bench (base, no draft)...")
    rb = bench(PRISMML.replace("llama-server", "llama-bench") if os.path.exists(
        PRISMML.replace("llama-server", "llama-bench")) else LBENCH,
        os.path.join(LLMS, "Ternary-Bonsai-27B-Q2_0.gguf"), 0)
    results["Ternary-Bonsai-27B Q2_0 [ecc-off-prismml-bench]"] = {
        "bench": rb, "gpu": "GPU0", "note": "PrismML fork (Q2_0_g128); llama-bench cannot do dspark pairs", "timestamp": TS}
    log(f"  bench: {rb}")

    # dspark server probe (draft n=4) - matches the original 36.3 t/s protocol
    log("Bonsai dspark server probe (PrismML, draft n=4)...")
    cmd = [PRISMML, "--model", os.path.join(LLMS, "Ternary-Bonsai-27B-Q2_0.gguf"),
           "--flash-attn", "on", "--host", "127.0.0.1", "--port", str(PORT),
           "--parallel", "1", "--temp", "0.0", "-n", "256", "--gpu-layers", "99",
           "--ctx-size", "8192", "--ubatch-size", "512", "--threads", "8", "--threads-batch", "8",
           "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
           "--spec-draft-model", os.path.join(LLMS, "Ternary-Bonsai-27B-dspark-Q4_1.gguf"),
           "--spec-draft-n-max", "4"]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(LOGS, "ecc2_bonsai_dspark_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    probe = {}
    try:
        for _ in range(150):
            time.sleep(2)
            if proc.poll() is not None:
                probe = {"error": "server died: " + open(logf.name).read()[-200:]}
                break
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                break
            except Exception:
                pass
        else:
            proc.kill(); probe = {"error": "timeout"}
        if not probe:
            import http.client
            best = 0.0
            for _ in range(2):
                conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
                payload = json.dumps({"messages": [{"role": "user", "content":
                    "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                    "max_tokens": 256, "temperature": 0.0, "stream": False})
                t0 = time.time()
                conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
                d = json.loads(conn.getresponse().read())
                dt = time.time() - t0
                toks = (d.get("usage") or {}).get("completion_tokens")
                if toks:
                    best = max(best, toks / dt)
                conn.close()
            probe = {"decode_tps_probe_dspark": round(best, 2)}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        logf.close()
    results["Ternary-Bonsai-27B Q2_0 (dspark) [ecc-off-dspark-probe]"] = {
        "probe": probe, "gpu": "GPU0", "note": "PrismML + dspark draft n=4, same protocol as original 36.3 t/s", "timestamp": TS}
    log(f"  dspark probe: {probe}")

    with open(BENCH_RESULTS, "w") as f:
        json.dump(results, f, indent=2)
    log("DONE - results in bench_results.json [ecc-off-origcfg] / [ecc-off-prismml-*] keys")


if __name__ == "__main__":
    main()
