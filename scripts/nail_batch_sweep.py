#!/usr/bin/env python3
"""Batching sweep for Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL on the 3060 (prod tuning).

For each --parallel level: start llama-server (cpu-moe), run:
  - 1 concurrent request  (single-stream decode t/s)
  - P concurrent requests (aggregate t/s + latency spread)
Realistic reviewer load: ~1.3k-token prompt, 384 gen tokens, temp 0.
Ladder stops at first startup failure (likely KV OOM). Stdlib only.
"""
import subprocess, json, time, os, sys, threading, http.client, urllib.request, statistics

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
MODEL = "/home/files/llms/Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
PORT = 18099
LOGS = "/tmp/coding-bench/logs"
OUT = os.path.join(LOGS, f"nail_batch_sweep{'_nc' + os.environ.get('NCMOE', '') if os.environ.get('NCMOE') else ''}.json")
THREADS_CANDIDATES = [int(x) for x in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["6"])]
PARALLEL_CANDIDATES = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["1", "2", "4", "6", "8"])]

CODE_FILLER = (
    "Review the following Python module for bugs, race conditions, and style issues. "
    "Be concise but cover every function.\n\n"
    "```python\n"
    + "\n".join(
        f"def handler_{i}(payload):\n"
        "    cache = {}\n"
        "    for key, value in sorted(payload.items()):\n"
        "        if key not in cache:\n"
        "            cache[key] = [value]\n"
        "        else:\n"
        "            cache[key].append(value)\n"
        "    return {k: sum(v) / len(v) for k, v in cache.items()}\n"
        for i in range(18)
    )
    + "\n```\nWrite the review now."
)


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def start_server(threads, parallel):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    ncmoe = os.environ.get("NCMOE")
    logf = open(os.path.join(LOGS, f"nail_sweep_t{threads}_p{parallel}{'_nc' + ncmoe if ncmoe else ''}.log"), "w")
    cmd = [BINARY, "--model", MODEL, "--flash-attn", "on", "--gpu-layers", "99"]
    # NCMOE set: first N layers' MoE experts on CPU, REST on GPU (partial residency).
    # NCMOE unset: --cpu-moe = ALL experts on CPU (baseline).
    cmd += ["--n-cpu-moe", ncmoe] if ncmoe else ["--cpu-moe"]
    cmd += ["--ctx-size", "262144", "--batch-size", "2048", "--ubatch-size", "512",
            "--threads", str(threads), "--threads-batch", str(threads),
            "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--reasoning", "off"]
    if os.environ.get("NO_MMAP"):
        cmd += ["--no-mmap"]
    cmd += ["--host", "127.0.0.1", "--port", str(PORT), "--parallel", str(parallel),
           "--cont-batching", "--kv-unified", "--temp", "0.0", "-n", "4096"]
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(150):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            tail = open(os.path.join(LOGS, f"nail_sweep_t{threads}_p{parallel}{'_nc' + ncmoe if ncmoe else ''}.log")).read()[-300:]
            return None, None, f"died: {tail}"
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
            if "Nail" not in props:
                proc.kill(); logf.close()
                return None, None, "wrong model on port"
            return proc, logf, None
        except RuntimeError:
            raise
        except Exception:
            pass
    proc.kill(); logf.close()
    return None, None, "timeout"


def stop_server(proc, logf):
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()
    if logf:
        logf.close()
    time.sleep(4)


def one_request(idx, results, max_tokens=384):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
    payload = json.dumps({"messages": [{"role": "user", "content": CODE_FILLER}],
                          "max_tokens": max_tokens, "temperature": 0.0, "stream": False})
    t0 = time.time()
    try:
        conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
        body = json.loads(conn.getresponse().read())
        dt = time.time() - t0
        toks = (body.get("usage") or {}).get("completion_tokens") or 0
        if not toks:
            results[idx] = {"error": str(body)[:200], "latency": dt}
        else:
            results[idx] = {"toks": toks, "latency": dt, "tps": toks / dt}
    except Exception as e:
        results[idx] = {"error": str(e)[:200], "latency": time.time() - t0}
    finally:
        conn.close()


def run_burst(n, max_tokens=384):
    results = {}
    threads = [threading.Thread(target=one_request, args=(i, results, max_tokens)) for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    ok = [r for r in results.values() if "toks" in r]
    if not ok:
        return {"error": "all requests failed", "detail": str(list(results.values())[:2])}
    total_toks = sum(r["toks"] for r in ok)
    lat = sorted(r["latency"] for r in ok)
    return {"n": n, "ok": len(ok), "wall_s": round(wall, 1),
            "agg_tps": round(total_toks / wall, 1),
            "lat_mean": round(statistics.mean(lat), 1),
            "lat_p95": round(lat[max(0, int(len(lat) * 0.95) - 1)], 1),
            "single_tps_median": round(statistics.median([r["tps"] for r in ok]), 2) if len(ok) > 1 else round(ok[0]["tps"], 2)}


def main():
    all_results = {}
    for threads in THREADS_CANDIDATES:
        for par in PARALLEL_CANDIDATES:
            log(f"=== threads={threads} parallel={par} ===")
            proc, logf, err = start_server(threads, par)
            if err:
                log(f"  startup FAILED: {err}")
                all_results[f"t{threads}_p{par}"] = {"error": err}
                if "died" in err or "timeout" in err:
                    log("  ladder stop (likely KV OOM)")
                    break
                continue
            try:
                single = run_burst(1)
                log(f"  single: {single}")
                row = {"single": single}
                if par > 1:
                    burst = run_burst(par)
                    log(f"  burst x{par}: {burst}")
                    row["burst"] = burst
                all_results[f"t{threads}_p{par}"] = row
            finally:
                stop_server(proc, logf)
        json.dump(all_results, open(OUT, "w"), indent=1)
    json.dump(all_results, open(OUT, "w"), indent=1)
    log(f"saved {OUT}")
    print(json.dumps(all_results, indent=1))


if __name__ == "__main__":
    main()
