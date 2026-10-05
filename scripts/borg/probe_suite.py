#!/usr/bin/env python3
"""Phase 8 suite probe: local-llm-benchmarks protocol against a RUNNING server.
8K-class prompt eval + 2 decode prompts x 2 runs, sampled (temp 0.3) per the
suite's standard settings. Emits one JSON row. Usage:
  probe_suite.py --name qwen38-27b --engine luce_server --port 8902 --out bench_results.json
"""
import argparse, json, time, urllib.request

PROMPT_EVAL_TEXT = open("/root/bench/prompt_eval_text.txt").read().strip()
DECODE_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list. Include docstring, type hints, and handle edge cases.",
    "Explain how merge sort works step by step. Include pseudocode and analyze the time complexity.",
]

def completion(port, prompt, n_predict, reasoning_off=False):
    body = {"model": "luce", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict, "temperature": 0.3, "top_p": 0.9,
            "top_k": 40, "stream": False}
    if reasoning_off:
        body["reasoning_effort"] = "off"
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    t = d.get("timings") or d.get("usage", {}).get("timings", {})
    pps = t.get("predicted_per_second") or t.get("decode_tokens_per_sec") or 0
    pn = t.get("prompt_n") or t.get("prompt_tokens") or d.get("usage", {}).get("prompt_tokens", 0)
    pms = t.get("prompt_ms") or t.get("prefill_ms") or 0
    return {"decode_tps": round(pps, 2), "decode_tokens": t.get("predicted_n") or d.get("usage", {}).get("completion_tokens", 0),
            "prompt_tps": round(pn / pms * 1000, 2) if pms else 0}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--engine", default="luce_server")
    ap.add_argument("--port", type=int, default=8902)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--out", default="/root/bench/bench_results.json")
    args = ap.parse_args()
    reasoning_off = args.name == "qwen38-flash-next"

    row = {"model": args.name, "engine": args.engine,
           "protocol": "8K prompt-eval + 2 decode prompts x2, temp 0.3", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    pe1 = completion(args.port, PROMPT_EVAL_TEXT, 8, reasoning_off)
    pe2 = completion(args.port, PROMPT_EVAL_TEXT, 8, reasoning_off)
    row["prompt_tps"] = pe2["prompt_tps"]
    decodes = [completion(args.port, p, args.n, reasoning_off) for p in DECODE_PROMPTS for _ in range(2)]
    row["decode_runs"] = decodes
    row["decode_tps_avg"] = round(sum(d["decode_tps"] for d in decodes) / len(decodes), 2)
    print(json.dumps(row, indent=2))
    try:
        data = json.load(open(args.out))
    except Exception:
        data = {}
    data[args.name] = row
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
