#!/usr/bin/env python3
"""Flash Next LCB-leg-only rerun (alias was unregistered on first attempt;
speed/sanity/tau2 already complete and merged). Same serve config, same merge."""
import subprocess, json, time, os, re, urllib.request, glob

XING4 = "/home/caimlas/git/llama.cpp-xing4/build/bin/llama-server"
MODEL = "/home/files/llms/flashnext-q20/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf"
LOGS = "/tmp/coding-bench/logs/flashnext_lane"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
ENTRY_NAME = "Qwen3.8-Flash-Next GSQ-RCO Q2_0 (V100)"
PORT = 18096
V100_PRODS = ["caimlas-btl4-v100", "caimlas-lfm25-v100"]
BEST_EXTRA = ["--n-cpu-moe", "8", "-fit", "on", "--gpu-layers", "99"]

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def sudo(*args):
    return subprocess.run(["sudo", "-n"] + list(args), capture_output=True, text=True)

def merge_lcb(lcb):
    prog = json.load(open(PROGRESS_FILE))
    for m in prog["models"]:
        if m["name"] == ENTRY_NAME:
            if lcb.get("pass_at_1") is not None:
                m["livecodebench"] = {"pass_at_1": lcb["pass_at_1"],
                                      "wall_time_s": lcb.get("wall_time_s"),
                                      "empty": lcb.get("empty"), "of": lcb.get("of")}
            else:
                m.setdefault("failures", []).append(
                    {"benchmark": "livecodebench", "attempt": 3,
                     "error": str(lcb.get("error", "no score"))[:400],
                     "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            break
    with open(PROGRESS_FILE + ".tmp", "w") as f:
        json.dump(prog, f, indent=1)
    os.replace(PROGRESS_FILE + ".tmp", PROGRESS_FILE)

def main():
    stopped = []
    try:
        for svc in V100_PRODS:
            sudo("systemctl", "stop", svc)
            stopped.append(svc)
        time.sleep(10)
        subprocess.run(["pkill", "-f", "BTL-4"], capture_output=True)
        subprocess.run(["pkill", "-f", "LFM2.5"], capture_output=True)
        time.sleep(3)

        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        logf = open(os.path.join(LOGS, "lcb_rerun2_server.log"), "w")
        cmd = [XING4, "--model", MODEL, "--flash-attn", "on",
               "--host", "127.0.0.1", "--port", str(PORT),
               "--ctx-size", "32768", "--batch-size", "2048", "--ubatch-size", "512",
               "--threads", "8", "--threads-batch", "8",
               "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
               "--parallel", "1", "--temp", "0.0", "-n", "4096",
               "--jinja"] + BEST_EXTRA
        proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        try:
            for _ in range(480):
                time.sleep(5)
                if proc.poll() is not None:
                    raise RuntimeError("server died: " + open(logf.name).read()[-500:])
                try:
                    urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                    props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
                    assert "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0" in props, "identity mismatch"
                    log("lcb server up")
                    break
                except AssertionError:
                    raise
                except Exception:
                    pass
            else:
                raise RuntimeError("server timeout")

            lenv = dict(os.environ)
            lenv.update({"OPENAI_KEY": "none",
                         "OPENAI_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
                         "HF_ALLOW_CODE_EVAL": "1",
                         "LCB_DISABLE_THINKING": "1"})
            out_dir = os.path.join(LCB_DIR, "output", ENTRY_NAME)
            if os.path.isdir(out_dir):
                subprocess.run(["rm", "-rf", out_dir])
            t0 = time.time()
            with open(os.path.join(LOGS, "lcb_local_flashnext_rerun2.log"), "w") as lf:
                subprocess.run([BENCH_PY, "-m", "lcb_runner.runner.main",
                                "--model", "local/flashnext-qwen38-q20",
                                "--scenario", "codegeneration",
                                "--release_version", "release_latest", "--n", "1",
                                "--temperature", "0.0", "--max_tokens", "4096",
                                "--num_problems", "75", "--openai_timeout", "900",
                                "--evaluate"],
                               stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR, env=lenv,
                               timeout=24 * 3600)
            wall = round(time.time() - t0, 1)
            pass1 = None
            for f in glob.glob(os.path.join(out_dir, "*_eval.json")):
                if f.endswith("_all.json"):
                    continue
                try:
                    data = json.load(open(f))
                    if isinstance(data, list) and data:
                        pass1 = data[0].get("pass@1")
                except Exception:
                    pass
            if pass1 is None:
                txt = open(os.path.join(LOGS, "lcb_local_flashnext_rerun2.log")).read().strip()
                m = re.search(r"^([\d.]+)$", txt.splitlines()[-1].strip()) if txt else None
                if m:
                    pass1 = float(m.group(1))
            n = empty = 0
            for f in glob.glob(os.path.join(out_dir, "Scenario.codegeneration_1_*.json")):
                if f.endswith("_all.json") or f.endswith("_eval.json"):
                    continue
                try:
                    for g in json.load(open(f)):
                        n += 1
                        if not (g.get("output_list") or [""])[0].strip():
                            empty += 1
                except Exception:
                    pass
            result = {"pass_at_1": pass1, "empty": empty, "of": n, "wall_time_s": wall}
            log(f"lcb rerun: {result}")
            merge_lcb(result)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except Exception:
                proc.kill()
            logf.close()
    finally:
        for svc in stopped:
            sudo("systemctl", "start", svc)
        time.sleep(5)
        for svc in stopped:
            log(f"prod {svc}: {sudo('systemctl','is-active',svc).stdout.strip()}")
    try:
        subprocess.run([BENCH_PY, "/home/caimlas/llm-benchmarks/scripts/generate_report.py"],
                       capture_output=True, text=True, timeout=300)
        log("report regenerated")
    except Exception as e:
        log(f"report regen failed: {e}")
    log("LCB RERUN DONE")

if __name__ == "__main__":
    main()
