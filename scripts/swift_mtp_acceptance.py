#!/usr/bin/env python3
"""
Swift-Qwen3.8-27B Q4_K_M MTP acceptance test (V100, port 18096).

Question: did ukisai's LoRA keep the Qwen3.8 trunk's MTP head calibrated?
Baseline: stock Qwen3.8-27B Q4_K_M measured 67.4% acc @ n=3 -> 35.6 t/s (+13%).

Protocol per references/mtp-spec-decode.md: for each of no-spec / MTP-n3 /
MTP-n5: start server (8K ctx), warmup, 2x timed 256-token completions at
temp 0.3, parse `draft acceptance = X (N accepted / M generated)` from the
server log. Verify the draft context actually engaged.

Writes bench_results.json keys "<file> [cfg]" + progress.json mtp_acceptance.
Carnice stopped once at top, restarted in a single top-level finally.
"""
import subprocess, json, time, os, re, urllib.request

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
MODEL = "/var/tmp/llms/Swift-Qwen3.8-27B-Q4_K_M.gguf"
LABEL = os.path.basename(MODEL)
PORT = 18096
LOGS = "/tmp/coding-bench/logs"
BENCH_RESULTS = "/home/caimlas/llm-benchmarks/bench_results.json"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
NAME = "Swift-Qwen3.8-27B Q4_K_M"
V100_PROD = "caimlas-carnice"

DECODE_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list. Include docstring, type hints, and handle edge cases.",
    "Explain how merge sort works step by step. Include pseudocode and analyze the time complexity.",
]

CONFIGS = [
    ("no-spec", []),
    ("MTP-n3", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]),
    ("MTP-n5", ["--spec-type", "draft-mtp", "--spec-draft-n-max", "5"]),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def start_server(extra, tag):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logname = os.path.join(LOGS, f"swift_mtp_{tag}_server.log")
    logf = open(logname, "w")
    cmd = [BINARY, "--model", MODEL, "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(PORT),
           "--gpu-layers", "99", "--ctx-size", "8192",
           "--batch-size", "2048", "--ubatch-size", "512",
           "--threads", "8", "--threads-batch", "8",
           "--parallel", "1", "--temp", "0.0", "-n", "4096", "--jinja"] + extra
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(240):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"{tag} died: " + open(logname).read()[-300:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
            if "Swift-Qwen3.8-27B-Q4_K_M" not in props:
                proc.kill(); logf.close()
                raise RuntimeError("wrong model on port")
            return proc, logf, logname
        except RuntimeError:
            raise
        except Exception:
            pass
    proc.kill(); logf.close()
    raise RuntimeError(f"{tag} timeout")


def stop(proc, logf):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    logf.close()
    time.sleep(3)


def timed_completion(prompt, max_tokens=256, temperature=0.3):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=180)
    payload = json.dumps({"messages": [{"role": "user", "content": prompt}],
                          "max_tokens": max_tokens, "temperature": temperature, "stream": False})
    t0 = time.time()
    conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    d = json.loads(conn.getresponse().read())
    dt = time.time() - t0
    conn.close()
    toks = (d.get("usage") or {}).get("completion_tokens")
    if not toks:
        raise RuntimeError("no completion_tokens")
    return toks / dt, toks


def parse_acceptance(logname):
    accs = []
    engaged = False
    for line in open(logname, errors="replace"):
        ll = line.lower()
        if "creating mtp draft context" in ll or "common_speculative_init_result" in ll and "mtp" in ll:
            engaged = True
        m = re.search(r"acceptance = ([\d.]+)", line)
        if m:
            accs.append(float(m.group(1)))
    return engaged, accs


def main():
    assert os.path.exists(MODEL), "GGUF missing"
    results = json.load(open(BENCH_RESULTS)) if os.path.exists(BENCH_RESULTS) else {}
    log("stopping caimlas-carnice for clean V100...")
    subprocess.run(["sudo", "-n", "systemctl", "stop", V100_PROD], capture_output=True)
    time.sleep(5)
    summary = {}
    try:
        for cfg_name, extra in CONFIGS:
            key = f"{LABEL} [{cfg_name}]"
            log(f"config: {cfg_name}")
            proc, logf, logname = start_server(extra, cfg_name.replace("-", "_"))
            try:
                timed_completion("Write a short hello world function.", max_tokens=16)
                runs = []
                for p in DECODE_PROMPTS:
                    tps, toks = timed_completion(p)
                    runs.append(round(tps, 1))
                    log(f"  decode: {tps:.1f} tok/s ({toks} tok)")
                avg = round(sum(runs) / len(runs), 1)
            except Exception as e:
                log(f"  probe failed: {e}")
                runs, avg = [], None
            engaged, accs = parse_acceptance(logname)
            if cfg_name != "no-spec" and not engaged:
                log("  WARNING: draft context line NOT found - head did not engage")
            entry = {"decode_tps_runs": runs, "decode_tps_avg": avg,
                     "acceptance_rates": accs,
                     "acceptance_avg": round(sum(accs) / len(accs), 3) if accs else None,
                     "draft_engaged": engaged,
                     "mtp_log_lines": [l.strip() for l in open(logname, errors="replace")
                                       if "acceptance" in l.lower()][-6:]}
            results[key] = entry
            summary[cfg_name] = {"decode_tps": avg, "acceptance_avg": entry["acceptance_avg"],
                                 "acceptance_rates": accs}
            with open(BENCH_RESULTS, "w") as f:
                json.dump(results, f, indent=2)
            stop(proc, logf)
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", V100_PROD], capture_output=True)
        time.sleep(15)

    # progress.json: attach mtp_acceptance to the Swift entry
    progress = json.load(open(PROGRESS_FILE))
    e = next((m for m in progress["models"] if m["name"] == NAME), None)
    if e:
        e["mtp_acceptance"] = summary
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp, PROGRESS_FILE)

    log("SUMMARY " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
