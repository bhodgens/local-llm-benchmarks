#!/usr/bin/env python3
"""Generic cold-prefill/decode probe for any OpenAI-compatible server on borg.
Unique random text each run => no prefix cache. Usage:
  gen_probe.py --port 8902 --model luce --words 1200 --label ds4-2k [--out file.jsonl]
"""
import argparse, json, random, string, time, urllib.request

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--words", type=int, default=1200)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--reasoning-off", action="store_true")
    args = ap.parse_args()

    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(4, 9))) for _ in range(args.words)]
    text = ("Summarize the following text in one sentence, then write a short Python function.\n\n"
            + " ".join(words))
    body = {"model": args.model, "messages": [{"role": "user", "content": text}],
            "max_tokens": args.max_tokens, "temperature": 0.0, "stream": False}
    if args.reasoning_off:
        body["reasoning_effort"] = "off"
    req = urllib.request.Request(f"http://127.0.0.1:{args.port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    wall = time.time() - t0
    u = d.get("usage", {})
    t = d.get("timings") or u.get("timings", {})
    pn = t.get("prompt_n") or u.get("prompt_tokens", 0)
    pms = t.get("prompt_ms") or t.get("prefill_ms", 0)
    dn = t.get("predicted_n") or u.get("completion_tokens", 0)
    dms = t.get("predicted_ms") or t.get("decode_ms", 0)
    row = {"label": args.label, "prompt_tokens": pn,
           "prefill_tps": round(pn / pms * 1000, 1) if pms else None,
           "decode_tps": round(dn / dms * 1000, 1) if dms else None,
           "decode_tokens": dn, "wall_s": round(wall, 1)}
    print(json.dumps(row))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

if __name__ == "__main__":
    main()
