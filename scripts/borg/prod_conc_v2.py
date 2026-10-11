#!/usr/bin/env python3
"""Concurrent smoke v2: BTL-4 via superfluid :8453 (4 parallel) while flashnext
(Strata :8080, R9700) serves its own concurrent wave — but flashnext currently
down; restart it first, keep BTL4/superfluid up."""
import json, subprocess, time, urllib.request, threading

def wait_up(url, want, tries=100):
    for _ in range(tries):
        time.sleep(5)
        try:
            if want in urllib.request.urlopen(url, timeout=3).read():
                return True
        except Exception:
            pass
    return False

# restart flashnext (it was down)
log = open("/root/bench/conc-fn-server.log", "w")
p = subprocess.Popen(["/root/strata/run-iq3_s.sh"], stdout=log, stderr=subprocess.STDOUT,
                     stdin=subprocess.DEVNULL, start_new_session=True)
print("flashnext starting...", flush=True)

def hit(port, model, label, out, max_tokens=256):
    body = {"model": model, "messages": [{"role": "user",
            "content": f"Write a detailed technical plan, section {label}. Be thorough."}],
            "max_tokens": max_tokens, "temperature": 0.7}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    wall = time.time() - t0
    ct = (d.get("usage") or {}).get("completion_tokens")
    out[label] = {"tok": ct, "wall": round(wall,1), "client_tps": round(ct/wall,1)}

def wave(port, model, tag, n=4):
    out = {}
    ts = [threading.Thread(target=hit, args=(port, model, f"{tag}-{i}", out)) for i in range(1, n+1)]
    t0 = time.time()
    for t in ts: t.start()
    for t in ts: t.join()
    wall = time.time() - t0
    tot = sum(v["tok"] for v in out.values())
    return out, round(tot/wall, 1), round(wall, 1)

print("waiting for flashnext...", flush=True)
if not wait_up("http://127.0.0.1:8080/v1/models", b"flash"):
    print("FLASHNEXT FAILED"); sys.exit(1)
time.sleep(20)
print("both up - firing concurrent waves", flush=True)

results = {}
th1 = threading.Thread(target=lambda: results.update(fn=wave(8080, "qwen3.8-flash-next-iq3_s", "fn", 4)))
th2 = threading.Thread(target=lambda: results.update(btl=wave(8453, "BTL-4-Compact-IQ2_XXS", "btl", 4)))
th1.start(); th2.start(); th1.join(); th2.join()

for k in ("fn", "btl"):
    out, agg, wall = results[k]
    print(f"{k}: aggregate {agg} tok/s over {wall}s", flush=True)
    print(json.dumps(out, indent=1), flush=True)

json.dump(results, open("/root/bench/results/prod-concurrency-smoke.json", "w"), indent=1)
print("CONCURRENCY_DONE", flush=True)
