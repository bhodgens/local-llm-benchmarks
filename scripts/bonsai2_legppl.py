#!/usr/bin/env python3
"""Leg PPL: quant parity sanity - llama-perplexity on PQ2_0 + PTQ1_0 (V100, prism HEAD build).
ADA-comparable: -c 512 --chunks 2, fa 1, q8_0 KV (their reference: both 7.1131 bit-identical).
Then full corpus if it fits the 15-min budget. GPU lock held for the whole phase.
"""
import fcntl, json, os, re, subprocess, time

BIN = "/tmp/prism-head/build/bin"
MODELS = "/var/tmp/llms/bench-0918/bonsai2"
WIKI = os.path.expanduser("~/.cache/wikitext-2/wiki.test.raw")
QUANTS = {"ptq1_0": f"{MODELS}/Ternary-Bonsai-2-27B-PTQ1_0.gguf",
          "pq2_0": f"{MODELS}/Ternary-Bonsai-2-27B-PQ2_0.gguf"}
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

def run_ppl(model, chunks, tag, timeout=3600):
    cmd = (f"{BIN}/llama-perplexity -m {model} -f {WIKI} -c 512 --chunks {chunks} "
           f"-fa 1 -ctk q8_0 -ctv q8_0 -t 8 --no-warmup")
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=3600, env=env)
    dt = time.time() - t0
    txt = (r.stdout + r.stderr)
    m = re.findall(r"final estimate: PPL = ([0-9.]+) .*", txt)
    m2 = re.findall(r"perplexity: perplexity = ([0-9.]+)", txt)  # per-chunk lines
    print(f"[{tag}] rc={r.returncode} {dt:.1f}s final={m[-1] if m else 'NONE'}")
    if not m:
        print(txt[-1500:])
    return {"rc": r.returncode, "seconds": round(dt,1), "final_ppl": m[-1] if m else None,
            "per_chunk": m2, "tail": txt[-600:]}

def main():
    lock = open("/tmp/gpu0.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    print("[lock] gpu0 acquired")
    try:
        stop_prod()
        for name, path in QUANTS.items():
            OUT[f"{name}_chunks2"] = run_ppl(path, 2, f"{name}-c2")
        # full corpus (~280k tokens ≈ 550 chunks): pp-bound at ~750 t/s ≈ 6-7 min/model.
        # hard 870s timeout per model; skip PQ2_0 full if PTQ1_0 full timed out.
        for name, path in QUANTS.items():
            prev = OUT.get("ptq1_0_full") if name == "pq2_0" else None
            if prev and prev["rc"] != 0:
                print(f"[skip] {name} full (PTQ full failed/timed out)")
                continue
            OUT[f"{name}_full"] = run_ppl(path, -1, f"{name}-full", timeout=870)
    finally:
        start_prod()
        fcntl.flock(lock, fcntl.LOCK_UN)
        with open("/tmp/legppl_results.json", "w") as f:
            json.dump(OUT, f, indent=1)
        print("[done] -> /tmp/legppl_results.json")

if __name__ == "__main__":
    main()
