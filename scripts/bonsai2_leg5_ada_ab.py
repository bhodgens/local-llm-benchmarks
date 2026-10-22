#!/usr/bin/env python3
"""Leg 5 GPU phase: A/B llama-bench stock (prism HEAD) vs ADA-ported build, plus PPL-parity gate.
Stock = /tmp/prism-head/build (73bfe6c + nothing), Ported = /tmp/ada-port/build (HEAD + ADA patches,
D256 skipped). Bench: pp4096,16384 + tg128, fa 1, q8_0 KV, -ub 2048, -r 3, -t 8, -o md (json broken).
PPL gate: ported llama-perplexity --chunks 2 -c 512 on both quants vs stock's 8.7508 (delta <= 0.005).
Extra: stock -b 1024 chunks2 (ADA-comparability crosscheck; their Q=(256,256,24,2) shape).
"""
import fcntl, json, os, re, subprocess, time

STOCK = "/tmp/prism-head/build/bin"
PORT  = "/tmp/ada-port/build/bin"
MODELS = "/var/tmp/llms/bench-0918/bonsai2"
M_PTQ = f"{MODELS}/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
M_PQ2 = f"{MODELS}/Ternary-Bonsai-2-27B-PQ2_0.gguf"
WIKI = os.path.expanduser("~/.cache/wikitext-2/wiki.test.raw")
OUT = {}

def sh(cmd, timeout=3600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def stop_prod():
    sh("sudo -n systemctl stop caimlas-minicpm-v100")
    for _ in range(30):
        time.sleep(2)
        if not sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0").stdout.strip():
            print("[prod] V100 clear"); return

def start_prod():
    sh("sudo -n systemctl start caimlas-minicpm-v100")
    print("[prod] restart issued")

def bench(bin_dir, tag):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = (f"{bin_dir}/llama-bench -m {M_PTQ} -p 4096,16384 -n 128 -fa 1 -ctk q8_0 -ctv q8_0 "
           f"-ub 2048 -r 3 -t 8 -o md")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600, env=env)
    dt = time.time() - t0
    print(f"[{tag}] rc={r.returncode} {dt:.0f}s")
    md = r.stdout
    print(md)
    return {"rc": r.returncode, "seconds": round(dt), "md": md,
            "stderr_tail": (r.stderr or "")[-400:]}

def ppl(bin_dir, model, tag, extra=""):
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    cmd = (f"{bin_dir}/llama-perplexity -m {model} -f {WIKI} -c 512 --chunks 2 "
           f"-fa 1 -ctk q8_0 -ctv q8_0 -t 8 --no-warmup {extra}")
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800, env=env)
    dt = time.time() - t0
    m = re.findall(r"Final estimate: PPL = ([0-9.]+)", r.stdout + r.stderr)
    print(f"[{tag}] rc={r.returncode} {dt:.0f}s PPL={m[-1] if m else 'NONE'}")
    return {"rc": r.returncode, "seconds": round(dt), "ppl": m[-1] if m else None}

def main():
    lock = open("/tmp/gpu0.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    print("[lock] gpu0 acquired")
    try:
        stop_prod()
        OUT["bench_stock"] = bench(STOCK, "bench-stock")
        OUT["bench_ported"] = bench(PORT, "bench-ported")
        OUT["ppl_ported_ptq1_0_c2"] = ppl(PORT, M_PTQ, "ppl-ported-ptq")
        OUT["ppl_ported_pq2_0_c2"] = ppl(PORT, M_PQ2, "ppl-ported-pq2")
        OUT["ppl_stock_b1024_ptq1_0"] = ppl(STOCK, M_PTQ, "ppl-stock-b1024", "-b 1024")
    finally:
        start_prod()
        fcntl.flock(lock, fcntl.LOCK_UN)
        with open("/tmp/leg5_results.json", "w") as f:
            json.dump(OUT, f, indent=1)
        print("[done] -> /tmp/leg5_results.json")

if __name__ == "__main__":
    main()
