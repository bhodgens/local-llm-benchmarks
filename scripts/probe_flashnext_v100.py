#!/usr/bin/env python3
"""
Qwen3.8 Flash Next GSQ-RCO Q2_0 (66.4 GB, qwen4exp arch) - V100 speed probe
(+ sanity gate; full LCB/tau2 lane only if decode is workable).

Fit: weights 66.4 GB vs V100 32 GB + ~37 GB RAM. Config ladder (first that
loads wins):
  A. --cpu-moe (experts on CPU, dense+attention+n-gram table on GPU)
     GPU side ~= n-gram table 26.8 GB + dense/attn few GB ~ 30 GB. CPU side
     ~= 34 GB experts (mmap; only touched pages resident).
  B. partial --n-cpu-moe N (first N layers' experts on CPU)
  C. mmap everything, low -ngl

Engine: ~/git/llama.cpp-xing4 fork (qwen4exp support verified in libllama.so,
matches GGUF header keys: ple.*, attention.indexer.*, expert_count 512).

Prod: caimlas-btl4-v100 + caimlas-lfm25-v100 stopped ONCE at start, restarted
in the single top-level finally (both live on the V100).

House protocol: 8K ctx, 256-token decode, temp 0, fa on, best of 2.
"""
import subprocess, json, time, os, urllib.request, http.client, sys

XING4 = "/home/caimlas/git/llama.cpp-xing4/build/bin/llama-server"
MODEL_DIR = "/home/files/llms/flashnext-q20"
MODEL = f"{MODEL_DIR}/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
LOGS = "/tmp/coding-bench/logs"
OUT = "/tmp/coding-bench/flashnext_v100_probe.json"
PORT = 18096
V100_PRODS = ["caimlas-btl4-v100", "caimlas-lfm25-v100"]

os.makedirs(LOGS, exist_ok=True)

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def sudo(*args):
    return subprocess.run(["sudo", "-n"] + list(args), capture_output=True, text=True)

def serve(extra, tag, ctx=8192, threads=8, timeout_s=1800):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(LOGS, f"flashnext_{tag}_server.log"), "w")
    cmd = [XING4, "--model", MODEL, "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(PORT),
           "--ctx-size", str(ctx),
           "--batch-size", "2048", "--ubatch-size", "512",
           "--threads", str(threads), "--threads-batch", str(threads),
           "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
           "--parallel", "1", "--temp", "0.0", "-n", "4096",
           "--jinja"] + list(extra)
    log(f"  serve: {' '.join(extra)}")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(5)
        if proc.poll() is not None:
            tail = open(logf.name).read()[-600:]
            logf.close()
            raise RuntimeError(f"{tag} died:\n{tail}")
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
            if "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0" not in props:
                proc.kill(); logf.close()
                raise RuntimeError(f"{tag} identity mismatch")
            log(f"  {tag} up (V100:{PORT})")
            return proc, logf
        except RuntimeError:
            raise
        except Exception:
            pass
    proc.kill(); logf.close()
    raise RuntimeError(f"{tag} timeout after {timeout_s}s")

def stop(proc, logf):
    if not proc:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except Exception:
        proc.kill()
    if logf:
        logf.close()
    time.sleep(5)

def speed_probe():
    best = 0.0
    last = None
    for _ in range(2):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=900)
            payload = {"messages": [{"role": "user", "content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens": 256, "temperature": 0.0, "stream": False,
                "chat_template_kwargs": {"enable_thinking": False}}
            t0 = time.time()
            conn.request("POST", "/v1/chat/completions",
                         json.dumps(payload), {"Content-Type": "application/json"})
            d = json.loads(conn.getresponse().read())
            dt = time.time() - t0
            toks = (d.get("usage") or {}).get("completion_tokens")
            conn.close()
            if toks:
                best = max(best, toks / dt)
        except Exception as e:
            last = str(e)
    return {"decode_tps": round(best, 2)} if best else {"error": last or "no tokens"}

def smoke_quality():
    """Coherence check: greedy short answer must be sane prose."""
    try:
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
        conn.request("POST", "/v1/chat/completions", json.dumps(
            {"messages": [{"role": "user", "content": "In one sentence, what is the capital of France?"}],
             "max_tokens": 64, "temperature": 0.0, "stream": False,
             "chat_template_kwargs": {"enable_thinking": False}}),
            {"Content-Type": "application/json"})
        d = json.loads(conn.getresponse().read())
        conn.close()
        txt = (d.get("choices") or [{}])[0].get("message", {}).get("content", "")
        return {"reply": txt[:300]}
    except Exception as e:
        return {"error": str(e)[:200]}

def main():
    result: dict = {"model": "Qwen3.8-Flash-Next GSQ-RCO Q2_0", "gpu": "V100",
              "engine": XING4, "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    stopped = []
    try:
        for svc in V100_PRODS:
            sudo("systemctl", "stop", svc)
            stopped.append(svc)
        time.sleep(10)
        subprocess.run(["pkill", "-f", "BTL-4"], capture_output=True)
        subprocess.run(["pkill", "-f", "LFM2.5"], capture_output=True)
        time.sleep(3)
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader", "-i", "0"],
                             capture_output=True, text=True).stdout.strip()
        log(f"V100 compute apps after stop: {out or 'NONE'}")

        # config ladder
        ladder = [
            ("cpu-moe", ["--cpu-moe", "--gpu-layers", "99"]),
            ("cpu-moe-ngl48", ["--cpu-moe", "--gpu-layers", "48"]),
            ("n-cpu-moe-48", ["--n-cpu-moe", "48", "--gpu-layers", "99"]),
        ]
        for tag, extra in ladder:
            try:
                proc, logf = serve(extra, tag, timeout_s=2400)
                try:
                    result["config"] = tag
                    result["serve_flags"] = " ".join(extra)
                    result["speed"] = speed_probe()
                    result["smoke"] = smoke_quality()
                    log(f"  {tag}: {result['speed']} | smoke: {str(result['smoke'])[:120]}")
                finally:
                    stop(proc, logf)
                break
            except Exception as e:
                log(f"  {tag}: FAILED {str(e)[:300]}")
                result.setdefault("failed_configs", []).append(
                    {"config": tag, "error": str(e)[:400]})
    finally:
        for svc in stopped:
            sudo("systemctl", "start", svc)
        time.sleep(5)
        for svc in stopped:
            st = sudo("systemctl", "is-active", svc).stdout.strip()
            log(f"prod {svc}: {st}")

    with open(OUT + ".tmp", "w") as f:
        json.dump(result, f, indent=2)
    os.replace(OUT + ".tmp", OUT)
    log(f"RESULT: {json.dumps(result)[:500]}")
    log("PROBE DONE")

if __name__ == "__main__":
    main()
