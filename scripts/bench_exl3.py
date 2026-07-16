#!/usr/bin/env python3
"""
Benchmark exllamav3 gemma-4-12B with continuous batching (batch=8).
Also benchmarks Bonsai on PrismML llama.cpp (--parallel 8).
"""
import os, sys, json, time, threading, subprocess, http.client, urllib.request

def bench_exl3_gemma(model_dir, gpu_id, batch_size=8):
    """Benchmark exllamav3 with gemma-4-12B EXL3 on specified GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ["EXLLAMA_NOCOMPILE"] = "1"

    from exllamav3 import Config, Model, Cache
    from exllamav3.generator import Generator
    from exllamav3.tokenizer import Tokenizer

    print(f"Loading {model_dir} on GPU {gpu_id}...")
    config = Config.from_directory(model_dir)

    model = Model.from_config(config)
    tokenizer = Tokenizer(config)

    # Allocate paged cache - enough for batch_size concurrent sequences
    # Each sequence needs up to ~4K tokens for prompt + 256 output
    max_tokens = batch_size * 8192
    cache = Cache(model, max_num_tokens=max_tokens)
    model.load_autoload(config, cache)

    print(f"Model loaded. Cache: {max_tokens} tokens.")

    generator = Generator(model, cache, tokenizer, max_batch_size=batch_size)

    prompt = ("Write a detailed essay about the history of computing, "
              "from Babbage to modern GPUs. Include key milestones, "
              "important figures, and technological breakthroughs.")

    # Warmup with batch=1
    print("Warmup (batch=1)...")
    generator.generate(prompt, max_new_tokens=16, sampler=None)

    # Timed batch run
    prompts = [prompt] * batch_size
    print(f"Running batch={batch_size}, 256 tokens each...")

    start = time.time()
    output = generator.generate(prompts, max_new_tokens=256, sampler=None)
    elapsed = time.time() - start

    # Count actual tokens generated
    total_tokens = 0
    for i, out in enumerate(output):
        # Each output is the completion text; estimate tokens
        # ExLlamaV3 generate returns strings, use tokenizer to count
        pass

    # Use a fixed count since we set max_new_tokens=256
    total_tokens = batch_size * 256
    agg_tps = total_tokens / elapsed

    print(f"\n{'='*60}")
    print(f"ExLlamaV3 gemma-4-12B-it EXL3 4.0bpw (GPU {gpu_id})")
    print(f"{'='*60}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Wall time:    {elapsed:.1f}s")
    print(f"  Aggregate:    {agg_tps:.1f} tok/s")
    print(f"  Per-slot:     {agg_tps/batch_size:.1f} tok/s")
    print(f"{'='*60}")

    # Check VRAM
    import subprocess
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    vram_lines = r.stdout.strip().split("\n")
    print(f"  VRAM used:    {vram_lines[gpu_id]}MB")

    return {"engine": "exllamav3", "model": "gemma-4-12B-it-exl3-4bpw",
            "gpu": gpu_id, "batch_size": batch_size,
            "total_tokens": total_tokens, "elapsed_s": round(elapsed, 1),
            "aggregate_tps": round(agg_tps, 1),
            "per_slot_tps": round(agg_tps/batch_size, 1)}

def bench_llama_bonsai(gpu_id, batch_size=8):
    """Benchmark PrismML llama.cpp with Bonsai at high parallelism."""
    binary = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
    model_path = "/home/files/llms/Ternary-Bonsai-27B-Q2_0.gguf"
    port = 18099

    # Each slot gets full ctx since Bonsai hybrid attention has tiny KV
    ctx_per_slot = 131072
    total_ctx = ctx_per_slot

    cmd = [
        binary, "--model", model_path, "--flash-attn", "on",
        "--batch-size", "2048", "--host", "127.0.0.1", "--port", str(port),
        "--parallel", str(batch_size), "--temp", "0.0", "-n", "4096",
        "--gpu-layers", "99", "--ctx-size", str(total_ctx), "--ubatch-size", "512",
        "--threads", "8", "--threads-batch", "8",
        "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
        # Enable continuous batching
        "--cont-batching",
    ]

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print(f"Starting Bonsai on GPU {gpu_id}, parallel={batch_size}, ctx={total_ctx//1024}K...")
    logf = open("/tmp/bonsai_batch_bench.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)

    for i in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                print("Server ready.")
                break
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open("/tmp/bonsai_batch_bench.log") as f:
                print(f"Server died: {f.read()[-500:]}")
            return None
    else:
        print("Server timeout!")
        proc.kill()
        return None

    # Warmup
    print("Warmup...")
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    payload = json.dumps({"messages": [{"role": "user", "content": "Hello"}],
                          "max_tokens": 8, "temperature": 0.0, "stream": False})
    conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    conn.getresponse().read()
    conn.close()

    prompt_text = ("Write a detailed essay about the history of computing, "
                   "from Babbage to modern GPUs. Include key milestones, "
                   "important figures, and technological breakthroughs.")

    results = [0] * batch_size

    def fire_request(idx):
        try:
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
            p = json.dumps({
                "messages": [{"role": "user", "content": prompt_text}],
                "max_tokens": 256, "temperature": 0.0, "stream": False,
            })
            c.request("POST", "/v1/chat/completions", p, {"Content-Type": "application/json"})
            resp = c.getresponse()
            data = json.loads(resp.read())
            results[idx] = data.get("usage", {}).get("completion_tokens", 256)
            c.close()
        except Exception as e:
            print(f"  Request {idx} failed: {e}")
            results[idx] = 0

    print(f"Running batch={batch_size}, 256 tokens each...")
    threads = []
    start = time.time()
    for i in range(batch_size):
        t = threading.Thread(target=fire_request, args=(i,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join(timeout=300)
    elapsed = time.time() - start

    total_tokens = sum(results)
    agg_tps = total_tokens / elapsed if elapsed > 0 else 0

    # Check VRAM
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    vram_lines = r.stdout.strip().split("\n")

    print(f"\n{'='*60}")
    print(f"llama.cpp (PrismML) Ternary-Bonsai-27B Q2_0 (GPU {gpu_id})")
    print(f"{'='*60}")
    print(f"  Batch size:   {batch_size}")
    print(f"  Total tokens: {total_tokens}")
    print(f"  Wall time:    {elapsed:.1f}s")
    print(f"  Aggregate:    {agg_tps:.1f} tok/s")
    print(f"  Per-slot:     {agg_tps/batch_size:.1f} tok/s")
    print(f"  VRAM used:    {vram_lines[gpu_id]}MB")
    print(f"{'='*60}")

    proc.terminate()
    try: proc.wait(timeout=10)
    except: proc.kill()
    logf.close()

    return {"engine": "llama.cpp-prismml", "model": "Ternary-Bonsai-27B-Q2_0",
            "gpu": gpu_id, "batch_size": batch_size,
            "total_tokens": total_tokens, "elapsed_s": round(elapsed, 1),
            "aggregate_tps": round(agg_tps, 1),
            "per_slot_tps": round(agg_tps/batch_size, 1)}

if __name__ == "__main__":
    results = []

    # Stop 3060 services to free VRAM
    subprocess.run(["sudo", "systemctl", "stop", "caimlas-ravenx", "caimlas-lfm"],
                   capture_output=True, timeout=15)
    time.sleep(3)

    # 1. ExLlamaV3 gemma-4-12B on 3060
    print("\n" + "#"*60)
    print("# BENCHMARK 1: ExLlamaV3 gemma-4-12B EXL3 on 3060")
    print("#"*60)
    try:
        r = bench_exl3_gemma("/home/files/llms/gemma-4-12B-it-exl3-4bpw", gpu_id=1, batch_size=8)
        if r: results.append(r)
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()

    # 2. Bonsai on V100 with parallel=8
    print("\n" + "#"*60)
    print("# BENCHMARK 2: llama.cpp Bonsai on V100 parallel=8")
    print("#"*60)
    try:
        # Stop V100 service
        subprocess.run(["sudo", "systemctl", "stop", "caimlas-coder"],
                       capture_output=True, timeout=15)
        time.sleep(3)
        r = bench_llama_bonsai(gpu_id=0, batch_size=8)
        if r: results.append(r)
    except Exception as e:
        import traceback
        print(f"FAILED: {e}")
        traceback.print_exc()

    # Restart services
    subprocess.run(["sudo", "systemctl", "start", "caimlas-ravenx", "caimlas-lfm", "caimlas-coder"],
                   capture_output=True, timeout=15)

    # Summary
    print("\n\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for r in results:
        print(f"  {r['engine']:<20} {r['model']:<35} batch={r['batch_size']} "
              f"agg={r['aggregate_tps']:.1f} tok/s  per-slot={r['per_slot_tps']:.1f} tok/s")
