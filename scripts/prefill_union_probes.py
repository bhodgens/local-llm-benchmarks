#!/usr/bin/env python3
"""
Prefill-only probes for the union of: top-12 LCB, top-10 decode tok/s, top-5 tau2
(unique set, one prefill measurement each). Staged per QUEUE.md convention -- NOT
auto-run; launch manually when the target GPU is idle.

Protocol: llama-bench pp512 only (-r 3, fa on, q8_0 kv, 8K depth -- same convention
as speed_norm/ecc-off probes so numbers are comparable with existing rows).
Results appended to bench_results.json under "<file-basename> [prefill]" with the
progress entry name recorded, so generate_report.py can pick them up via its
canonical-probe ranking.

Which models are NOT here: every union member that already has a prefill probe in
bench_results.json (11), plus MiniCPM5-2B (3060) as a file-dup of the V100 entry,
plus BTL-4 Q4_K_M (think) as a file-dup of BTL-4 Q4_K_M (probed).
"""
import subprocess, json, time, os, re, sys

LBENCH = "/home/caimlas/git/llama.cpp/build/bin/llama-bench"
BENCH_RESULTS = "/home/caimlas/llm-benchmarks/bench_results.json"
LOGS = "/tmp/coding-bench/logs"
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

# (progress entry name, model file, gpu_index)
MODELS = [
    # top-12 LCB (not already probed)
    ("Qwen3.8-Flash-Next GSQ-RCO Q2_0 (V100)",     "/home/files/llms/flashnext-q20/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", 0),
    ("Muse-Glimmer-30B-UD-Q4_K_XL (think)",        "/home/files/llms/Muse-Glimmer-30B-UD-Q4_K_XL.gguf", 0),
    ("Qwythos-27B-v1 Q4_K_M",                      "/home/files/llms/gguf/Qwythos-27B-Q4_K_M.gguf", 0),
    ("DogukanUrker-BTL-4 Q4_K_M",                  "/home/files/llms/dogukanurker-btl4-q4km/BTL-4-Q4_K_M.gguf", 0),
    ("Bonsai-2-27B Ternary CRACK PQ2_0 (think)",   "/var/tmp/llms/bench-0918/crack/Bonsai-2-27B-PQ2_0-CRACK.gguf", 0),
    ("gemma-4-26B-A4B-it-QAT Q4_0",                "/home/files/llms/gemma-4-26B-A4B-it-QAT-Q4_0.gguf", 0),
    # top-10 decode tok/s (not already probed)
    ("MiniCPM5-2B Q8_0 (V100)",                    "/home/files/llms/MiniCPM5-2B-Q8_0.gguf", 0),
    # top-5 tau2 (not already probed)
    ("Ternary-Bonsai-2-27B PQ2_0 (V100)",          "/var/tmp/llms/bench-0918/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf", 0),
    ("Qwen3.8-27B Q4_K_M",                         "/home/files/llms/Qwen3.8-27B-Q4_K_M.gguf", 0),
]

NOTE = ("Note: Qwen3.8-Flash-Next is a split (00001-of-00002) GGUF; llama-bench takes the "
        "first shard. All entries V100 (GPU0) -- none of the unprobed union members are "
        "3060-only. Approx 2-4 min per model.")

def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def pp512(path, gpu, tag):
    cmd = [LBENCH, "-m", path, "-fa", "on", "-ngl", "99",
           "-ctk", "q8_0", "-ctv", "q8_0", "-p", "512", "-n", "0",
           "-d", "8192", "-r", "3", "-t", "8"]
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    os.makedirs(LOGS, exist_ok=True)
    blog = os.path.join(LOGS, f"prefill_{tag}.log")
    with open(blog, "w") as lf:
        subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, timeout=1800)
    txt = open(blog).read()
    ts = re.findall(r"\|\s*([0-9.]+)\s*\u00b1\s*[0-9.]+\s*\|", txt)
    if not ts:
        return {"error": f"no llama-bench output (see {blog})"}
    return {"pp512": float(ts[0])}

def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    with open(BENCH_RESULTS) as f:
        results = json.load(f)
    for name, path, gpu in MODELS:
        if only and name not in only:
            continue
        if not os.path.exists(path):
            log(f"SKIP {name}: {path} missing"); continue
        tag = re.sub(r"[^a-zA-Z0-9]+", "_", name)[:40]
        log(f"=== {name} (GPU{gpu}) ===")
        r = pp512(path, gpu, tag)
        key = f"{os.path.basename(path)} [prefill]"
        results[key] = {"prefill": r, "entry_name": name, "gpu": f"GPU{gpu}",
                        "timestamp": TS}
        with open(BENCH_RESULTS, "w") as f:
            json.dump(results, f, indent=2)
        log(f"  {key}: {r}")
    log("=== prefill probes COMPLETE (regenerate report to pick up) ===")

if __name__ == "__main__":
    main()
