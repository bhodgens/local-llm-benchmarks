#!/usr/bin/env python3
"""Flash Next V100 probe, arm 2/3: try to place the 26.8GB n-gram table on GPU.
Arm A2: no cpu-moe, -fit on (auto-fit places what fits in VRAM)
Arm A3: n-cpu-moe 24, -fit on (half experts on GPU)
"""
import subprocess, json, time, os, urllib.request, http.client

XING4 = "/home/caimlas/git/llama.cpp-xing4/build/bin/llama-server"
MODEL = "/home/files/llms/flashnext-q20/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
LOGS = "/tmp/coding-bench/logs"
OUT = "/tmp/coding-bench/flashnext_v100_probe4.json"
PORT = 18096
V100_PRODS = ["caimlas-btl4-v100", "caimlas-lfm25-v100"]

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def sudo(*args):
    return subprocess.run(["sudo", "-n"] + list(args), capture_output=True, text=True)

def serve(extra, tag, timeout_s=2400):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(LOGS, f"flashnext_{tag}_server.log"), "w")
    cmd = [XING4, "--model", MODEL, "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(PORT),
           "--ctx-size", "8192", "--batch-size", "2048", "--ubatch-size", "512",
           "--threads", "8", "--threads-batch", "8",
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
            log(f"  {tag} up")
            return proc, logf
        except RuntimeError:
            raise
        except Exception:
            pass
    proc.kill(); logf.close()
    raise RuntimeError(f"{tag} timeout")

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
            conn.request("POST", "/v1/chat/completions", json.dumps(payload),
                         {"Content-Type": "application/json"})
            d = json.loads(conn.getresponse().read())
            dt = time.time() - t0
            toks = (d.get("usage") or {}).get("completion_tokens")
            conn.close()
            if toks:
                best = max(best, toks / dt)
        except Exception as e:
            last = str(e)
    return {"decode_tps": round(best, 2)} if best else {"error": last or "no tokens"}

def main():
    result: dict = {"model": "Qwen3.8-Flash-Next GSQ-RCO Q2_0", "gpu": "V100", "engine": XING4,
                    "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
    stopped = []
    try:
        for svc in V100_PRODS:
            sudo("systemctl", "stop", svc)
            stopped.append(svc)
        time.sleep(10)
        subprocess.run(["pkill", "-f", "BTL-4"], capture_output=True)
        subprocess.run(["pkill", "-f", "LFM2.5"], capture_output=True)
        time.sleep(3)

        arms = [
            ("ncpumoe8", ["--n-cpu-moe", "8", "-fit", "on", "--gpu-layers", "99"]),
            ("ncpumoe8-mtp", ["--n-cpu-moe", "8", "-fit", "on", "--gpu-layers", "99",
                              "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]),
        ]
        for tag, extra in arms:
            try:
                proc, logf = serve(extra, tag)
                try:
                    vram = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                           "--format=csv,noheader", "-i", "0"],
                                          capture_output=True, text=True).stdout.strip()
                    sp = speed_probe()
                    result[tag] = {"flags": " ".join(extra), "speed": sp, "vram": vram}
                    log(f"  {tag}: {sp} | vram: {vram}")
                finally:
                    stop(proc, logf)
            except Exception as e:
                log(f"  {tag}: FAILED {str(e)[:300]}")
                result[tag] = {"flags": " ".join(extra), "error": str(e)[:400]}
    finally:
        for svc in stopped:
            sudo("systemctl", "start", svc)
        time.sleep(5)
        for svc in stopped:
            log(f"prod {svc}: {sudo('systemctl', 'is-active', svc).stdout.strip()}")
    with open(OUT + ".tmp", "w") as f:
        json.dump(result, f, indent=2)
    os.replace(OUT + ".tmp", OUT)
    log(f"RESULT: {json.dumps(result)[:600]}")

if __name__ == "__main__":
    main()
