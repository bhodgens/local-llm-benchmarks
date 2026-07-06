#!/usr/bin/env python3
"""
Benchmark Qwopus3.6-35B-A3B-Coder-MTP with different cpu-moe settings.
No MTP, just no-spec decode speed at various configurations.
Also benchmarks LFM2.5-8B models.
"""
import subprocess, json, time, sys, os

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
PORT = 18088

PROMPT_EVAL = """You are a helpful assistant. Please analyze the following text and provide a summary.

The quick brown fox jumps over the lazy dog. This sentence is a pangram, meaning it contains every letter of the English alphabet at least once. Pangrams have been used since the dawn of typography to display font samples and test keyboard layouts. The most famous pangram in English is undoubtedly "The quick brown fox jumps over the lazy dog," which has appeared in countless word processors, font preview windows, and typing tests around the world.

However, pangrams exist in many languages. In French, "Portez ce vieux whisky au juge blond qui fume" serves the same purpose. In Spanish, "El veloz murcielago hindu comia feliz cardillo y kiwi" contains every letter. German has "Zwolf Boxkampfer jagen Viktor quer uber den grossen Sylter Deich." Each of these sentences demonstrates the unique phonetic and orthographic characteristics of their respective languages.

The history of pangrams dates back to ancient times. Greek and Latin scholars created similar sentences for educational purposes. In the modern era, with the advent of digital typography and Unicode, pangrams have taken on new importance. They are used to test whether fonts support all necessary glyphs, whether keyboards are properly mapped, and whether text rendering engines handle all characters correctly.

Beyond pangrams, the printing press revolutionized information dissemination. Johannes Gutenberg's invention in the 15th century made books accessible to the masses, leading to increased literacy rates and the spread of knowledge. The Renaissance, the Scientific Revolution, and eventually the Enlightenment all owe much to the printing press. It democratized information in ways that were previously unimaginable.

Today, we stand at another inflection point with artificial intelligence. Large language models can generate text, translate languages, write code, and answer questions with remarkable accuracy. These models are trained on vast corpora of text data, learning patterns and relationships that allow them to produce human-like responses. The implications for education, research, and commerce are profound."""

DECODE_PROMPTS = [
    "Here is a detailed implementation of merge sort in Python:\n\n```python\ndef merge_sort(arr: list) -> list:\n    \"\"\"Sort an array using the merge sort algorithm.\n    \n    Args:\n        arr: List of comparable elements to sort.\n    Returns:\n        A new sorted list.\n    \"\"\"\n    ",
    "Step-by-step explanation of quicksort:\n\n1. Choose a pivot element from the array\n2. ",
]

def start_server(model_path, extra_args, port=PORT):
    cmd = [BINARY,
        "--model", model_path,
        "--device", "CUDA0",
        "--flash-attn", "on",
        "--ctx-size", "8192",
        "--batch-size", "2048",
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
    cmd.extend(extra_args)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"

    logfile = open(f"/tmp/bench_cfg_{port}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logfile, stderr=subprocess.STDOUT, text=True)

    for i in range(90):
        time.sleep(2)
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            if json.loads(req.read()).get("status") == "ok":
                return proc, logfile
        except:
            pass
        if proc.poll() is not None:
            logfile.close()
            with open(f"/tmp/bench_cfg_{port}.log") as f:
                print(f"  FAILED: {f.read()[-500:]}")
            return None, None
    proc.kill()
    return None, None

def bench(port, prompt, n_predict=256):
    import urllib.request
    payload = json.dumps({"prompt": prompt, "n_predict": n_predict, "temperature": 0.3, "top_p": 0.9, "top_k": 40, "min_p": 0.05, "repeat_penalty": 1.1, "stream": False}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=payload, headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    t = data.get("timings", {})
    pred_n = t.get("predicted_n", 0)
    pred_ms = t.get("predicted_ms", 1)
    return {
        "prompt_tps": round(t.get("prompt_n",1) / (t.get("prompt_ms",1)/1000), 1),
        "decode_tps": round(pred_n / (pred_ms/1000), 1) if pred_ms > 1 and pred_n > 5 else 0,
        "decode_tokens": pred_n,
    }

def run_config(label, model_path, extra_args, results):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    proc, logfile = start_server(model_path, extra_args)
    if proc is None:
        results[label] = {"error": "server failed"}
        return

    try:
        bench(PORT, "Hello", 8)  # warmup
        time.sleep(0.5)
        pe = bench(PORT, PROMPT_EVAL, 16)
        d1 = bench(PORT, DECODE_PROMPTS[0], 256)
        d2 = bench(PORT, DECODE_PROMPTS[1], 256)

        valid = [d for d in [d1, d2] if d["decode_tps"] > 0]
        avg = round(sum(d["decode_tps"] for d in valid) / len(valid), 1) if valid else 0

        results[label] = {
            "prompt_eval_tps": pe["prompt_tps"],
            "decode_tps_avg": avg,
            "decode_tps_1": d1["decode_tps"],
            "decode_tps_2": d2["decode_tps"],
        }
        print(f"  prompt_eval={pe['prompt_tps']} tok/s  decode_avg={avg} tok/s")

        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
    except Exception as e:
        results[label] = {"error": str(e)}
        print(f"  ERROR: {e}")
    finally:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
        logfile.close()
        time.sleep(3)

RESULTS_FILE = "/home/caimlas/llm-benchmarks/bench_cfg_results.json"
results = {}
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        results = json.load(f)

MOE_MODEL = "/home/files/llms/Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf"

# --- 35B MoE cpu-moe parameter sweep ---
print("\n" + "="*60)
print("35B MoE CPU-MOE PARAMETER SWEEP (no-spec)")
print("="*60)

moe_configs = [
    # (label, extra_args)
    ("cpu-moe ngl=99 ubatch=256 thr=4",    ["--gpu-layers", "99", "--cpu-moe", "--ubatch-size", "256", "--threads", "4", "--threads-batch", "4"]),
    ("cpu-moe ngl=99 ubatch=512 thr=4",    ["--gpu-layers", "99", "--cpu-moe", "--ubatch-size", "512", "--threads", "4", "--threads-batch", "4"]),
    ("cpu-moe ngl=99 ubatch=128 thr=4",    ["--gpu-layers", "99", "--cpu-moe", "--ubatch-size", "128", "--threads", "4", "--threads-batch", "4"]),
    ("cpu-moe ngl=99 ubatch=256 thr=6",    ["--gpu-layers", "99", "--cpu-moe", "--ubatch-size", "256", "--threads", "6", "--threads-batch", "6"]),
    ("cpu-moe ngl=99 ubatch=256 thr=2",    ["--gpu-layers", "99", "--cpu-moe", "--ubatch-size", "256", "--threads", "2", "--threads-batch", "2"]),
    # Try without cpu-moe at lower ngl for comparison
    ("no-cpu-moe ngl=15 ubatch=256 thr=4",  ["--gpu-layers", "15", "--ubatch-size", "256", "--threads", "4", "--threads-batch", "4"]),
    ("no-cpu-moe ngl=10 ubatch=256 thr=4",  ["--gpu-layers", "10", "--ubatch-size", "256", "--threads", "4", "--threads-batch", "4"]),
]

for label, args in moe_configs:
    if label in results and "error" not in results[label]:
        print(f"  SKIP: {label} (already done)")
        continue
    run_config(label, MOE_MODEL, args, results)

# --- LFM2.5 models ---
print("\n" + "="*60)
print("LFM2.5-8B MODELS (3060, full GPU offload)")
print("="*60)

lfm_configs = [
    ("LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M",
     "/home/files/llms/LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"),
    ("LFM2.5-8B-A1B base Q4_K_M (existing)",
     "/home/files/llms/LFM2.5-8B-A1B-Q4_K_M.gguf"),
]

lfm_args = ["--gpu-layers", "99", "--ubatch-size", "512", "--threads", "4", "--threads-batch", "4"]

for label, model in lfm_configs:
    if label in results and "error" not in results[label]:
        print(f"  SKIP: {label}")
        continue
    if not os.path.exists(model):
        print(f"  SKIP: {model} not found yet")
        continue
    run_config(label, model, lfm_args, results)

# Final summary
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)
print(f"\n{'Config':<45} {'Decode':>8} {'Prompt':>8}")
print("-"*65)
for label in sorted(results.keys()):
    r = results[label]
    if "error" in r:
        print(f"  {label:<43} ERROR")
    else:
        print(f"  {label:<43} {r.get('decode_tps_avg','?'):>6}  {r.get('prompt_eval_tps','?'):>6}")
