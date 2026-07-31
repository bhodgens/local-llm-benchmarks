#!/usr/bin/env python3
"""
Master benchmark orchestrator. Runs all models sequentially, saves results to JSON.
Handles both V100 (dense) and 3060 (MoE with cpu-moe) configurations.
"""
import subprocess
import json
import time
import sys
import os

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
RESULTS_FILE = "/home/caimlas/llm-benchmarks/bench_results.json"

# --- Benchmark targets ---
# (model_filename, device, is_moe, has_mtp)
V100_MODELS = [
    ("Qwopus3.6-27B-Coder-MTP-Q5_K_S.gguf",     "CUDA0", False, True),   # existing
    ("Qwopus3.6-27B-v2-MTP-Q4_K_M.gguf",          "CUDA0", False, True),   # new
    ("Qwopus3.5-27B-v3-Q4_K_M.gguf",              "CUDA0", False, False),  # new, no MTP
    ("Qwopus3.6-27B-Coder-Compat-MTP-Q4_K_M.gguf","CUDA0", False, True),   # new
    ("Qwen3.6-27B-FableFusion-MTP-Q4_K_M.gguf",  "CUDA0", False, True),   # new - DavidAU Fable Fusion 711
]

M3060_MODELS = [
    ("Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf",       "CUDA1", True, False),  # existing
    ("unsloth-qwen3.6-35B-A3B-UD-Q4_K_M.gguf",                "CUDA1", True, False),  # existing
    ("Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf",               "CUDA1", True, True),   # new
    ("RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf", "CUDA1", True, True),  # new
]

PROMPT_EVAL_TEXT = """You are a helpful assistant. Please analyze the following text and provide a summary.

The quick brown fox jumps over the lazy dog. This sentence is a pangram, meaning it contains every letter of the English alphabet at least once. Pangrams have been used since the dawn of typography to display font samples and test keyboard layouts. The most famous pangram in English is undoubtedly "The quick brown fox jumps over the lazy dog," which has appeared in countless word processors, font preview windows, and typing tests around the world.

However, pangrams exist in many languages. In French, "Portez ce vieux whisky au juge blond qui fume" serves the same purpose. In Spanish, "El veloz murcielago hindu comia feliz cardillo y kiwi" contains every letter. German has "Zwolf Boxkampfer jagen Viktor quer uber den grossen Sylter Deich." Each of these sentences demonstrates the unique phonetic and orthographic characteristics of their respective languages.

The history of pangrams dates back to ancient times. Greek and Latin scholars created similar sentences for educational purposes. In the modern era, with the advent of digital typography and Unicode, pangrams have taken on new importance. They are used to test whether fonts support all necessary glyphs, whether keyboards are properly mapped, and whether text rendering engines handle all characters correctly.

Beyond pangrams, the printing press revolutionized information dissemination. Johannes Gutenberg's invention in the 15th century made books accessible to the masses, leading to increased literacy rates and the spread of knowledge. The Renaissance, the Scientific Revolution, and eventually the Enlightenment all owe much to the printing press. It democratized information in ways that were previously unimaginable.

Today, we stand at another inflection point with artificial intelligence. Large language models can generate text, translate languages, write code, and answer questions with remarkable accuracy. These models are trained on vast corpora of text data, learning patterns and relationships that allow them to produce human-like responses. The implications for education, research, and commerce are profound."""

DECODE_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list. Include docstring, type hints, and handle edge cases.",
    "Explain how merge sort works step by step. Include pseudocode and analyze the time complexity.",
]

def start_server(model_path, device, is_moe, mtp=False, n_max=5, port=18099):
    cmd = [BINARY,
        "--model", model_path,
        "--device", "CUDA0",  # Always CUDA0 since CUDA_VISIBLE_DEVICES already isolates
        "--flash-attn", "on",
        "--ctx-size", "8192",
        "--batch-size", "2048",
        "--ubatch-size", "512" if not is_moe else "256",
        "--threads", "4",
        "--threads-batch", "2",
        "--host", "127.0.0.1",
        "--port", str(port),
        "--parallel", "1",
        "--temp", "0.3",
        "--top-p", "0.9",
        "--top-k", "40",
        "--min-p", "0.05",
        "--repeat-penalty", "1.1",
        "-n", "512",
    ]

    if is_moe:
        cmd.extend(["--gpu-layers", "99", "--cpu-moe"])
    else:
        cmd.extend(["--gpu-layers", "99"])

    if mtp:
        cmd.extend(["--spec-type", "draft-mtp"])
        if n_max > 0:
            cmd.extend(["--spec-draft-n-max", str(n_max)])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0" if "CUDA0" in device else "1"

    print(f"  CMD: {' '.join(cmd[:8])}...")

    logfile = open(f"/tmp/bench_server_{port}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT, text=True)

    for i in range(90):
        time.sleep(2)
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            if json.loads(req.read()).get("status") == "ok":
                print(f"  Server ready after {i*2}s")
                return proc, logfile
        except:
            pass
        if proc.poll() is not None:
            logfile.close()
            with open(f"/tmp/bench_server_{port}.log") as f:
                output = f.read()[-1500:]
            print(f"  Server DIED. Last output:\n{output}")
            return None, None

    print("  Server timeout")
    proc.kill()
    logfile.close()
    return None, None

def stop_server(proc, logfile, port):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
    if logfile:
        logfile.close()
    time.sleep(3)

def benchmark(port, prompt, n_predict=256):
    import urllib.request
    payload = json.dumps({
        "prompt": prompt, "n_predict": n_predict,
        "temperature": 0.3, "top_p": 0.9, "top_k": 40,
        "min_p": 0.05, "repeat_penalty": 1.1, "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    t = data.get("timings", {})
    return {
        "prompt_tps": round(t.get("prompt_n",1) / (t.get("prompt_ms",1)/1000), 1),
        "prompt_tokens": t.get("prompt_n",0),
        "decode_tps": round(t.get("predicted_n",1) / (t.get("predicted_ms",1)/1000), 1),
        "decode_tokens": t.get("predicted_n",0),
    }

def get_mtp_stats(port, logfile_path):
    """Read MTP acceptance stats from server log."""
    stats = {}
    try:
        # Send a SIGUSR1 or just check the log after requests
        time.sleep(1)
        with open(logfile_path) as f:
            content = f.read()
        for line in content.split('\n'):
            ll = line.lower()
            if 'acc' in ll and ('token' in ll or 'draft' in ll):
                stats.setdefault('lines', []).append(line.strip())
            if 'spec' in ll and ('stat' in ll or 'accept' in ll or 'draft' in ll):
                stats.setdefault('lines', []).append(line.strip())
    except:
        pass
    return stats

def bench_model(model_path, device, is_moe, has_mtp, results, port=18099):
    label = os.path.basename(model_path)
    if not os.path.exists(model_path):
        print(f"  SKIP: {label} not found (still downloading?)")
        return

    configs = [("no-spec", False, 0)]
    if has_mtp:
        configs.append(("MTP-n3", True, 3))
        configs.append(("MTP-n5", True, 5))

    for config_name, mtp, n_max in configs:
        key = f"{label} [{config_name}]"
        if key in results and 'error' not in results[key]:
            print(f"  SKIP: {key} already benchmarked")
            continue
        print(f"\n{'='*60}")
        print(f"  Benchmark: {key}")
        print(f"{'='*60}")

        proc, logfile = start_server(model_path, device, is_moe, mtp=mtp, n_max=n_max, port=port)
        if proc is None:
            results[key] = {"error": "server failed"}
            continue

        try:
            # Warmup
            print("  Warmup...")
            benchmark(port, "Hello", n_predict=8)
            time.sleep(1)

            # Prompt eval
            print("  Prompt eval test...")
            pe = benchmark(port, PROMPT_EVAL_TEXT, n_predict=16)

            # Decode (2 runs)
            print("  Decode test 1...")
            d1 = benchmark(port, DECODE_PROMPTS[0], n_predict=256)
            print("  Decode test 2...")
            d2 = benchmark(port, DECODE_PROMPTS[1], n_predict=256)

            avg_decode = round((d1["decode_tps"] + d2["decode_tps"]) / 2, 1)

            results[key] = {
                "prompt_eval_tps": pe["prompt_tps"],
                "prompt_tokens": pe["prompt_tokens"],
                "decode_tps_1": d1["decode_tps"],
                "decode_tps_2": d2["decode_tps"],
                "decode_tps_avg": avg_decode,
                "decode_tokens_avg": (d1["decode_tokens"] + d2["decode_tokens"]) // 2,
            }

            if mtp:
                results[key]["mtp_log"] = get_mtp_stats(port, f"/tmp/bench_server_{port}.log")

            print(f"  -> prompt_eval: {pe['prompt_tps']} tok/s | "
                  f"decode_avg: {avg_decode} tok/s")

            # Save intermediate results
            with open(RESULTS_FILE, "w") as f:
                json.dump(results, f, indent=2)

        except Exception as e:
            results[key] = {"error": str(e)}
            print(f"  ERROR: {e}")
        finally:
            stop_server(proc, logfile, port)

def main():
    results = {}
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        print(f"Loaded {len(results)} existing results")

    # --- V100 benchmarks ---
    print("\n" + "="*60)
    print("V100 BENCHMARKS (CUDA0 - 27B dense models)")
    print("="*60)
    for fn, dev, moe, mtp in V100_MODELS:
        bench_model(os.path.join(LLMS_DIR, fn), dev, moe, mtp, results)

    # --- 3060 benchmarks ---
    print("\n" + "="*60)
    print("3060 BENCHMARKS (CUDA1 - 35B MoE models)")
    print("="*60)
    for fn, dev, moe, mtp in M3060_MODELS:
        bench_model(os.path.join(LLMS_DIR, fn), dev, moe, mtp, results)

    # Final save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {RESULTS_FILE}")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
