#!/usr/bin/env python3
"""
3060 tok/s probes for the last-run (bench-0918 + think-rerun) models.

Models from the 2026-10-06 batch + 2026-10-08/09 think reruns that FIT the
3060 (12 GB) but were benched on V100 only:
  1. Ternary-Bonsai-2-27B PQ2_0        (6.71 GB, PrismML prebuilt)
  2. Bonsai-2-27B CRACK PQ2_0          (6.71 GB, PrismML prebuilt)
  3. Bonsai-2-27B PTQ1_0               (5.54 GB, sudoingx fork)
  4. Bonsai-2-27B PTQ1_0 + MTP n1      (6.53 GB, sudoingx fork)

Does NOT fit / not probed: BTL-4 Q4_K_M (~16 GB), Muse-Glimmer-30B Q4_K_XL
(~17 GB), Xing4.0 (20.1 GB; 3060 cpu-moe probe already exists 2026-10-06,
17.23 t/s — merged separately), gemma-4-12B QAT (already has a 3060 number),
K2-Horizon-Uno (row INVALID, 3060 speed 53.93 already recorded).

Prod handling: caimlas-gemma4-3060 stopped ONCE at start, restarted in a
single top-level finally. Probe protocol = house standard: 8K ctx, 256-token
decode, temp 0, flash-attn on, best of 2.

Results -> /tmp/coding-bench/probes_3060_bench0918.json (merged into
progress.json separately), then report.html regenerated.
"""
import subprocess, json, time, os, urllib.request, http.client, sys

STAGE = "/var/tmp/llms/bench-0918"
PRISMML = f"{STAGE}/prismml-bin/llama-prism-b10685-7dffb15/llama-server"
SUDOINGX = "/home/caimlas/git/llama.cpp-sudoingx/build/bin/llama-server"
LOGS = "/tmp/coding-bench/logs/bench0918"
OUT = "/tmp/coding-bench/probes_3060_bench0918.json"
PORT = 18097
G30_PROD = "caimlas-gemma4-3060"

os.makedirs(LOGS, exist_ok=True)
RESULTS = {}

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def save():
    with open(OUT + ".tmp", "w") as f:
        json.dump(RESULTS, f, indent=2)
    os.replace(OUT + ".tmp", OUT)

def sudo(*args):
    return subprocess.run(["sudo", "-n"] + list(args), capture_output=True, text=True)

def serve(binary, model, extra, tag, ctx=8192, threads=6):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "1"
    logf = open(os.path.join(LOGS, f"probe3060_{tag}_server.log"), "w")
    cmd = [binary, "--model", model, "--flash-attn", "on",
           "--host", "127.0.0.1", "--port", str(PORT),
           "--gpu-layers", "99", "--ctx-size", str(ctx),
           "--batch-size", "2048", "--ubatch-size", "512",
           "--threads", str(threads), "--threads-batch", str(threads),
           "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
           "--parallel", "1", "--temp", "0.0", "-n", "4096",
           "--jinja"] + list(extra)
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(300):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"{tag} died: " + open(logf.name).read()[-400:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
            key = os.path.basename(model).split("-0000")[0]
            if key not in props:
                proc.kill(); logf.close()
                raise RuntimeError(f"{tag} identity mismatch: props lacks {key}")
            log(f"  {tag} up (3060:{PORT})")
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
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    if logf:
        logf.close()
    time.sleep(3)

def speed_probe(thinking_off=True):
    best = 0.0
    last = None
    for _ in range(2):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=600)
            payload = {"messages": [{"role": "user", "content":
                "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
                "max_tokens": 256, "temperature": 0.0, "stream": False}
            if thinking_off:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
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


def main():
    lanes = [
        ("Ternary-Bonsai-2-27B PQ2_0 (3060)", PRISMML,
         f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf", ["-fit", "off"]),
        ("Bonsai-2-27B Ternary CRACK PQ2_0 (3060)", PRISMML,
         f"{STAGE}/crack/Bonsai-2-27B-PQ2_0-CRACK.gguf", ["-fit", "off"]),
        ("Bonsai-2-27B PTQ1_0 (3060)", SUDOINGX,
         f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf", []),
        ("Bonsai-2-27B PTQ1_0 + MTP n1 (3060)", SUDOINGX,
         f"{STAGE}/bonsai2/Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf",
         ["--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]),
    ]
    stopped = False
    try:
        sudo("systemctl", "stop", G30_PROD)
        time.sleep(8)
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader", "-i", "1"],
                             capture_output=True, text=True).stdout.strip()
        log(f"3060 compute apps after stop: {out or 'NONE'}")
        stopped = True
        for name, binary, model, extra in lanes:
            tag = name.replace(" ", "_").replace("+", "plus")
            RESULTS[name] = {"engine": binary, "gpu": "3060",
                             "file": model,
                             "ts": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())}
            try:
                proc, logf = serve(binary, model, extra, tag)
                try:
                    RESULTS[name]["speed"] = speed_probe()
                    log(f"  {name}: {RESULTS[name]['speed']}")
                finally:
                    stop(proc, logf)
            except Exception as e:
                RESULTS[name]["error"] = str(e)[:400]
                log(f"  {name}: ERROR {str(e)[:200]}")
            save()
    finally:
        if stopped:
            sudo("systemctl", "start", G30_PROD)
            time.sleep(5)
            st = sudo("systemctl", "is-active", G30_PROD).stdout.strip()
            log(f"prod {G30_PROD} restored: {st}")
    log("ALL PROBES DONE")

if __name__ == "__main__":
    main()
