#!/usr/bin/env python3
"""
Comprehensive benchmark for llama.cpp models.
Tests: no-spec baseline, MTP (where supported).
Measures: prompt eval speed (tok/s), decode speed (tok/s), MTP acceptance rate.

Usage: python3 bench_model.py --model PATH --port PORT --device CUDA0|CUDA1 [--mtp] [--n-max N]
"""

import argparse
import json
import subprocess
import time
import sys
import re
import os

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"

# Standard test prompts
PROMPT_EVAL_TEXT = """You are a helpful assistant. Please analyze the following text and provide a summary.

The quick brown fox jumps over the lazy dog. This sentence is a pangram, meaning it contains every letter of the English alphabet at least once. Pangrams have been used since the dawn of typography to display font samples and test keyboard layouts. The most famous pangram in English is undoubtedly "The quick brown fox jumps over the lazy dog," which has appeared in countless word processors, font preview windows, and typing tests around the world.

However, pangrams exist in many languages. In French, "Portez ce vieux whisky au juge blond qui fume" serves the same purpose. In Spanish, "El veloz murciélago hindú comía feliz cardillo y kiwi" contains every letter. German has "Zwölf Boxkämpfer jagen Viktor quer über den großen Sylter Deich." Each of these sentences demonstrates the unique phonetic and orthographic characteristics of their respective languages.

The history of pangrams dates back to ancient times. Greek and Latin scholars created similar sentences for educational purposes. In the modern era, with the advent of digital typography and Unicode, pangrams have taken on new importance. They are used to test whether fonts support all necessary glyphs, whether keyboards are properly mapped, and whether text rendering engines handle all characters correctly.

Beyond pangrams, the printing press revolutionized information dissemination. Johannes Gutenberg's invention in the 15th century made books accessible to the masses, leading to increased literacy rates and the spread of knowledge. The Renaissance, the Scientific Revolution, and eventually the Enlightenment all owe much to the printing press. It democratized information in ways that were previously unimaginable.

Today, we stand at another inflection point with artificial intelligence. Large language models can generate text, translate languages, write code, and answer questions with remarkable accuracy. These models are trained on vast corpora of text data, learning patterns and relationships that allow them to produce human-like responses. The implications for education, research, and commerce are profound.

As we look to the future, it is clear that technology will continue to evolve at an accelerating pace. The challenges we face - from climate change to social inequality - require innovative solutions that leverage the best of human creativity and machine intelligence. By understanding the tools at our disposal and using them wisely, we can build a better world for future generations."""

DECODE_PROMPT = "Write a Python function that implements binary search on a sorted list. Include docstring, type hints, and handle edge cases. Explain the time and space complexity."

def start_server(model_path, port, device, mtp=False, n_max=5, extra_args=None):
    """Start a llama-server instance and return the process."""
    cmd = [BINARY,
        "--model", model_path,
        "--device", device,
        "--gpu-layers", "99",
        "--flash-attn", "on",
        "--ctx-size", "8192",
        "--batch-size", "2048",
        "--ubatch-size", "512" if "CUDA0" in device else "256",
        "--threads", "4",
        "--threads-batch", "2",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--parallel", "1",
        "--temp", "0.3",
        "--top-p", "0.9",
        "--top-k", "40",
        "--min-p", "0.05",
        "--repeat-penalty", "1.1",
        "-n", "512",
    ]

    # Add cpu-moe for 3060 (CUDA1)
    if "CUDA1" in device:
        cmd.extend(["--gpu-layers", "15"])

    if mtp:
        cmd.extend(["--spec-type", "draft-mtp"])
        if n_max > 0:
            cmd.extend(["--spec-draft-n-max", str(n_max)])

    if extra_args:
        cmd.extend(extra_args)

    env = dict(os.environ)
    if "CUDA0" in device:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    else:
        env["CUDA_VISIBLE_DEVICES"] = "1"

    print(f"Starting server: {' '.join(cmd[:6])}... (port {port})")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Wait for server to be ready
    for i in range(120):
        time.sleep(2)
        try:
            import urllib.request
            req = urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
            resp = json.loads(req.read())
            if resp.get("status") == "ok":
                print(f"  Server ready after {i*2}s")
                return proc
        except:
            pass
        if proc.poll() is not None:
            # Process died
            output = proc.stdout.read()[-2000:] if proc.stdout else ""
            print(f"  Server DIED. Output:\n{output}")
            return None

    print("  Server failed to become ready in 240s")
    proc.kill()
    return None

def benchmark_decode(port, prompt, n_predict=256):
    """Run a decode benchmark and return (tokens_generated, time_seconds, tok_per_s)."""
    import urllib.request

    payload = json.dumps({
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.3,
        "top_p": 0.9,
        "top_k": 40,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
        "stream": False,
    }).encode()

    req = urllib.request.Request(f"http://localhost:{port}/completion", data=payload,
                                 headers={"Content-Type": "application/json"})

    start = time.time()
    resp = urllib.request.urlopen(req, timeout=120)
    data = json.loads(resp.read())
    elapsed = time.time() - start

    tokens_generated = data.get("timings", {}).get("predicted_n", 0)
    eval_time = data.get("timings", {}).get("predicted_ms", 0) / 1000.0
    prompt_tokens = data.get("timings", {}).get("prompt_n", 0)
    prompt_ms = data.get("timings", {}).get("prompt_ms", 0) / 1000.0

    decode_tps = tokens_generated / eval_time if eval_time > 0 else 0
    prompt_tps = prompt_tokens / prompt_ms if prompt_ms > 0 else 0

    return {
        "decode_tps": decode_tps,
        "decode_tokens": tokens_generated,
        "decode_time_s": eval_time,
        "prompt_tps": prompt_tps,
        "prompt_tokens": prompt_tokens,
        "prompt_time_s": prompt_ms,
    }

def get_acceptance_stats(proc_output):
    """Parse acceptance stats from server output."""
    # Look for patterns like: #acc tokens=X
    # Also look for draft stats
    stats = {}

    # Try to find acceptance rate
    for line in proc_output.split('\n'):
        if '#acc' in line or 'accept' in line.lower():
            stats['acceptance_line'] = line.strip()
        if 'draft' in line.lower() and 'gen' in line.lower():
            stats['draft_line'] = line.strip()

    return stats

def run_benchmark(model_path, port, device, mtp=False, n_max=5, extra_args=None):
    """Full benchmark suite for a model."""
    label = os.path.basename(model_path)
    config = f"MTP n={n_max}" if mtp else "no-spec"
    print(f"\n{'='*60}")
    print(f"Benchmark: {label} [{config}]")
    print(f"{'='*60}")

    proc = start_server(model_path, port, device, mtp=mtp, n_max=n_max, extra_args=extra_args)
    if proc is None:
        return {"model": label, "config": config, "error": "server failed to start"}

    try:
        # Warmup
        print("  Warmup...")
        benchmark_decode(port, "Hello", n_predict=8)

        # Prompt eval benchmark (large prompt, short generation)
        print("  Testing prompt eval...")
        prompt_result = benchmark_decode(port, PROMPT_EVAL_TEXT, n_predict=16)

        # Decode benchmark (short prompt, long generation)
        print("  Testing decode...")
        decode_result = benchmark_decode(port, DECODE_PROMPT, n_predict=256)

        # Second decode run for stability
        print("  Testing decode (run 2)...")
        decode_result2 = benchmark_decode(port, "Explain how merge sort works. Include pseudocode.", n_predict=256)

        # Average decode results
        avg_decode = (decode_result["decode_tps"] + decode_result2["decode_tps"]) / 2

        result = {
            "model": label,
            "config": config,
            "prompt_eval_tps": round(prompt_result["prompt_tps"], 1),
            "prompt_tokens": prompt_result["prompt_tokens"],
            "decode_tps_run1": round(decode_result["decode_tps"], 1),
            "decode_tps_run2": round(decode_result2["decode_tps"], 1),
            "decode_tps_avg": round(avg_decode, 1),
            "decode_tokens": decode_result["decode_tokens"],
        }

        print(f"\n  RESULTS: prompt_eval={result['prompt_eval_tps']} tok/s, "
              f"decode_avg={result['decode_tps_avg']} tok/s")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except:
            proc.kill()
        time.sleep(3)

    # Read server output for MTP stats
    try:
        remaining = proc.stdout.read() if proc.stdout else ""
        if mtp:
            stats = get_acceptance_stats(remaining)
            if stats:
                result["mtp_stats"] = stats
                print(f"  MTP stats: {stats}")
    except:
        pass

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--port", type=int, default=18099)
    parser.add_argument("--device", required=True)
    parser.add_argument("--mtp", action="store_true")
    parser.add_argument("--n-max", type=int, default=5)
    args = parser.parse_args()

    result = run_benchmark(args.model, args.port, args.device, mtp=args.mtp, n_max=args.n_max)
    print(json.dumps(result, indent=2))
