#!/usr/bin/env python3
"""MiniCPM5-2B Q8_0 + DSpark Q8_0 draft head: speed probe on both GPUs,
identical serving config to the plain baselines (131K ctx, f16 KV).
Baselines: 3060 112.66 t/s, V100 166.04 t/s.
Stops caimlas-qwythos once for the V100 lane; restarts at the end."""
import subprocess, json, time, os, urllib.request

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
PORT = 18096
MAIN = os.path.join(LLMS, "MiniCPM5-2B-Q8_0.gguf")
DRAFT = os.path.join(LLMS, "MiniCPM5-2B-DSpark-Q8_0.gguf")


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def probe(gpu, draft, tag):
    extra = ["--spec-draft-model", DRAFT, "--spec-draft-n-max", "4"] if draft else []
    cmd = [BINARY, "--model", MAIN, "--flash-attn", "on", "--host", "127.0.0.1",
           "--port", str(PORT), "--gpu-layers", "99", "--ctx-size", "131072",
           "-np", "1", "--batch-size", "2048", "--ubatch-size", "1024",
           "--cache-type-k", "f16", "--cache-type-v", "f16",
           "--temp", "1.0", "--top-p", "0.95", "--jinja",
           "--threads", "8", "--threads-batch", "8", "-fit", "off"] + extra
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    logf = open(os.path.join(LOGS, f"minicpm_dspark_{tag}_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    try:
        for _ in range(240):
            time.sleep(2)
            if proc.poll() is not None:
                raise RuntimeError("server died: " + open(logf.name).read()[-300:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                break
            except Exception:
                pass
        else:
            raise RuntimeError("timeout")
        props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
        assert "MiniCPM5-2B-Q8_0" in props, "wrong model on port"
        import http.client
        best = 0.0
        for _ in range(3):
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
            payload = json.dumps({"messages": [{"role": "user", "content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens": 256, "temperature": 0.0, "stream": False})
            t0 = time.time()
            conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
            d = json.loads(conn.getresponse().read())
            dt = time.time() - t0
            toks = (d.get("usage") or {}).get("completion_tokens")
            conn.close()
            if not toks:
                raise RuntimeError("no completion_tokens")
            best = max(best, toks / dt)
        return round(best, 2)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        logf.close()
        time.sleep(3)


def main():
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(5)
    results = {}
    try:
        for gpu, tag, base in [(0, "v100", 166.04), (1, "3060", 112.66)]:
            t0 = time.time()
            tps = probe(gpu, True, tag)
            results[tag] = {"dspark_decode_tps": tps, "plain_baseline": base,
                            "delta_pct": round(100 * (tps - base) / base, 1)}
            log(f"{tag}: dspark {tps} t/s vs plain {base} ({results[tag]['delta_pct']:+.1f}%)")
        # save into progress.json
        p = json.load(open(PROGRESS_FILE))
        for m in p["models"]:
            if m["name"] == "Ternary-Bonsai-27B Q2_0 (dspark)":
                pass  # untouched
        for name_key, gpu_tag in [("MiniCPM5-2B Q8_0 (V100)", "v100"), ("MiniCPM5-2B Q8_0 (3060)", "3060")]:
            for m in p["models"]:
                if m["name"] == name_key:
                    m["dspark_draft"] = results[gpu_tag]
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(p, f, indent=2)
        os.replace(tmp, PROGRESS_FILE)
        log("progress.json updated")
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "caimlas-qwythos"], capture_output=True)
        time.sleep(15)
    log(json.dumps(results))


if __name__ == "__main__":
    main()
