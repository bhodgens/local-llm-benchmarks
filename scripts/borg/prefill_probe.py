#!/usr/bin/env python3
"""Cold-prefill probe for Strata: unique nonce text avoids prefix cache.
Usage: prefill_probe.py --port 8080 --tokens 8000 --label mmap
Prints JSON with prefill tok/s and decode tok/s.
"""
import argparse, json, time, urllib.request, random, string

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--tokens", type=int, default=8000)
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    # unique filler so nothing is prefix-cached
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(4, 9))) for _ in range(args.tokens)]
    text = "Summarize this log in one sentence: " + " ".join(words)
    payload = json.dumps({
        "model": "qwen3.8-flash-next-iq3_s",
        "messages": [{"role": "user", "content": text}],
        "max_tokens": 128, "temperature": 0.0,
        "reasoning_effort": "off", "stream": False,
    }).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    wall = time.time() - t0
    t = d.get("timings", {})
    pn = t.get("prompt_n", 0); pms = t.get("prompt_ms", 0)
    out = t.get("predicted_n", 0); dms = t.get("predicted_ms", 0)
    row = {"label": args.label, "prompt_tokens": pn,
           "prefill_tps": round(pn / pms * 1000, 1) if pms else None,
           "decode_tps": round(t.get("predicted_per_second", 0), 1),
           "decode_tokens": out, "wall_s": round(wall, 1)}
    print(json.dumps(row))

if __name__ == "__main__":
    main()
