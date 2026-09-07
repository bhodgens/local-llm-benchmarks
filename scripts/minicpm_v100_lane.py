#!/usr/bin/env python3
"""MiniCPM5-2B on V100 (GPU0) - same full lane as the 3060 run: user-supplied
serving config, speed probe, sanity:25, LCB 75 thinking-off, tau2 (LFM user
sim on 3060:8081). Records 'MiniCPM5-2B Q8_0 (V100)'.
Requires caimlas-qwythos stopped (handled: stop at start, restart at end)."""
import subprocess, json, time, os, re, urllib.request

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
GATE = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"
NAME = "MiniCPM5-2B Q8_0 (V100)"
PORT = 18096
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wait_health(proc, port, logf, tag):
    for _ in range(240):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"{tag} died: " + open(logf.name).read()[-300:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3)
            return
        except Exception:
            pass
    proc.kill(); logf.close()
    raise RuntimeError(f"{tag} timeout")


def serve(extra, tag):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(LOGS, f"minicpmv100_{tag}_server.log"), "w")
    proc = subprocess.Popen([BINARY, "--model", os.path.join(LLMS, "MiniCPM5-2B-Q8_0.gguf"),
                             "--flash-attn", "on", "--host", "127.0.0.1",
                             "--port", str(PORT)] + extra,
                            env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    wait_health(proc, PORT, logf, tag)
    props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
    assert "MiniCPM5-2B-Q8_0" in props, "wrong model on port"
    log(f"  {tag} up, identity OK")
    return proc, logf


def stop(proc, logf):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    logf.close()
    time.sleep(3)


def speed_probe():
    import http.client
    best = 0.0
    for _ in range(2):
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
            return {"error": "no completion_tokens"}
        best = max(best, toks / dt)
    return {"decode_tps": round(best, 2)}


CFG = ["--gpu-layers", "99", "--ctx-size", "131072", "-np", "1",
       "--batch-size", "2048", "--ubatch-size", "1024",
       "--cache-type-k", "f16", "--cache-type-v", "f16",
       "--temp", "1.0", "--top-p", "0.95", "--jinja",
       "--threads", "8", "--threads-batch", "8"]


def main():
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(5)
    try:
        progress = json.load(open(PROGRESS_FILE))
        proc, logf = serve(CFG, "main")
        sp = speed_probe()
        log(f"  speed: {sp}")
        stop(proc, logf)

        e = next((m for m in progress["models"] if m["name"] == NAME), None)
        if e is None:
            e = {"name": NAME, "file": "MiniCPM5-2B-Q8_0.gguf", "category": "2B-dense",
                 "gpu": "V100", "engine": "llama.cpp", "thinking": False, "template": "stock"}
            progress["models"].append(e)
        e["decode_tps"] = sp.get("decode_tps")
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp, PROGRESS_FILE)

        glog = os.path.join(LOGS, "minicpmv100_gate.log")
        with open(glog, "w") as lf:
            subprocess.run([BENCH_PY, GATE, "--model-file", "MiniCPM5-2B-Q8_0.gguf",
                            "--name", NAME, "--gpu", "0", "--benchmarks", "sanity:25"],
                           stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
        frac = re.findall(r"(\d+)\s*/\s*25", open(glog).read())
        san = {"sanity_passed": int(frac[-1]), "sanity_pct": round(100 * int(frac[-1]) / 25, 1)} if frac else {"error": "no score"}
        log(f"  sanity: {san}")

        proc, logf = serve(CFG, "lcb")
        usim_logf = open(os.path.join(LOGS, "minicpmv100_usersim.log"), "w")
        usim_env = dict(os.environ)
        usim_env["CUDA_VISIBLE_DEVICES"] = "1"
        usim = subprocess.Popen([BINARY, "--model", os.path.join(LLMS, USER_SIM_FILE),
                                 "--flash-attn", "on", "--host", "127.0.0.1", "--port", "8081",
                                 "--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
                                 "--threads", "4", "--threads-batch", "4",
                                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                                 "--parallel", "2", "--temp", "0.0", "-n", "4096"],
                                env=usim_env, stdout=usim_logf, stderr=subprocess.STDOUT, text=True)
        wait_health(usim, 8081, usim_logf, "usersim")
        log("  user sim up on GPU1:8081")
        try:
            # LCB
            lenv = dict(os.environ)
            lenv.update({"OPENAI_KEY": "none", "OPENAI_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
                         "HF_ALLOW_CODE_EVAL": "1", "LCB_DISABLE_THINKING": "1"})
            out_dir = os.path.join(LCB_DIR, "output", NAME)
            import glob
            t0 = time.time()
            with open(os.path.join(LOGS, "minicpmv100_lcb.log"), "w") as lf:
                subprocess.run([BENCH_PY, "-m", "lcb_runner.runner.main",
                                "--model", "local/minicpm5-2b-q8", "--scenario", "codegeneration",
                                "--release_version", "release_latest", "--n", "1",
                                "--temperature", "0.0", "--max_tokens", "4096",
                                "--num_problems", "75", "--openai_timeout", "300", "--evaluate"],
                               stdout=lf, stderr=subprocess.STDOUT, cwd=LCB_DIR, env=lenv, timeout=4 * 3600)
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
            n = empty = 0
            for f in glob.glob(os.path.join(out_dir, "Scenario.codegeneration_1_*.json")):
                if f.endswith("_all.json") or f.endswith("_eval.json"):
                    continue
                for g in json.load(open(f)):
                    n += 1
                    if not (g.get("output_list") or [""])[0].strip():
                        empty += 1
            log(f"  LCB: pass@1={pass1} ({wall/60:.0f} min) empty={empty}/{n}")
            stop(proc, logf)

            # tau2 (agent still needed! serve again)
            proc, logf = serve(CFG, "tau2")
            safe = re.sub(r"[/()\[\]]", "", NAME.replace(" ", "_"))
            cmd = ["uv", "run", "tau2", "run", "--domain", "airline",
                   "--agent-llm", f"openai/{safe}",
                   "--agent-llm-args", json.dumps({"api_key": "none",
                       "api_base": f"http://127.0.0.1:{PORT}/v1", "temperature": 0.0}),
                   "--user-llm", f"openai/{USER_SIM_NAME}",
                   "--user-llm-args", json.dumps({"api_key": "none",
                       "api_base": "http://127.0.0.1:8081/v1"}),
                   "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
                   "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
                   "--seed", "42", "--save-to", f"tau2_{safe}"]
            tenv = dict(os.environ)
            tenv["OPENAI_API_KEY"] = "none"
            t0 = time.time()
            with open(os.path.join(LOGS, f"minicpmv100_tau2.log"), "w") as lf:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=86400,
                               env=tenv, cwd=TAU2_DIR)
            content = open(os.path.join(LOGS, "minicpmv100_tau2.log")).read()
            m = re.search(r"Average Reward\s+([\d.]+)", content)
            m2 = re.search(r"Task Pass Rate\s+([\d.]+)", content)
            tau = {"reward": float(m.group(1)) if m else None,
                   "task_pass_rate": float(m2.group(1)) if m2 else None,
                   "wall_time_s": round(time.time() - t0, 1)}
            log(f"  tau2: reward={tau['reward']} ({(time.time()-t0)/60:.0f} min)")
        finally:
            stop(proc, logf)
            usim.terminate()
            try:
                usim.wait(timeout=10)
            except Exception:
                usim.kill()
            usim_logf.close()

        progress = json.load(open(PROGRESS_FILE))
        e = next(m for m in progress["models"] if m["name"] == NAME)
        e["benchkit"] = {"sanity": san}
        e["livecodebench"] = {"pass_at_1": pass1, "empty": empty, "of": n, "wall_time_s": wall}
        if pass1 is not None:
            e["tau2"] = tau
        with open(PROGRESS_FILE + ".tmp", "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(PROGRESS_FILE + ".tmp", PROGRESS_FILE)
        log("MiniCPM V100 lane COMPLETE")
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "caimlas-qwythos"], capture_output=True)
        time.sleep(15)


if __name__ == "__main__":
    main()
