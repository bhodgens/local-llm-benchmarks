#!/usr/bin/env python3
"""Full lane for Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic Q4_K_M (V100):
speed probe -> sanity:25 -> LCB 75 (thinking off) -> tau2 (thinking off,
LFM user sim on 3060). All thinking-family fixes applied:
- LCB: LCB_DISABLE_THINKING=1 (qwen38 allowlisted in oai_runner.py)
- tau2: --chat-template-kwargs {"enable_thinking": false} on the agent server
(empty-turn fix; all-15 denominator from results.json)
V100 exclusive: caimlas-qwythos stopped for the duration."""
import subprocess, json, time, os, re, urllib.request, glob

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
GATE = "/home/caimlas/llm-benchmarks/scripts/run_benchkit_gate.py"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
MODEL = os.path.join(LLMS, "Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic.Q4_K_M.gguf")
NAME = "Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic Q4_K_M"
USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"
PORT = 18096
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def serve(extra, tag):
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"
    logf = open(os.path.join(LOGS, f"oblit_{tag}_server.log"), "w")
    proc = subprocess.Popen([BINARY, "--model", MODEL, "--flash-attn", "on",
                             "--host", "127.0.0.1", "--port", str(PORT),
                             "--gpu-layers", "99", "--ctx-size", "32768",
                             "--batch-size", "2048", "--ubatch-size", "512",
                             "--threads", "8", "--threads-batch", "8",
                             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                             "--parallel", "2", "--temp", "0.0", "-n", "4096",
                             "--jinja"] + extra,
                            env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for _ in range(300):
        time.sleep(2)
        if proc.poll() is not None:
            logf.close()
            raise RuntimeError(f"{tag} died: " + open(logf.name).read()[-300:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
            if "OBLITERATED-Mythos-Class-Agentic" not in props:
                proc.kill(); logf.close()
                raise RuntimeError("wrong model on port")
            log(f"  {tag} up, identity OK")
            return proc, logf
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


def speed_probe():
    import http.client
    best = 0.0
    for _ in range(2):
        conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=300)
        payload = json.dumps({"messages": [{"role": "user", "content":
            "Write a detailed essay about the history of computing, from Babbage to modern GPUs."}],
            "max_tokens": 256, "temperature": 0.0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}})
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


def start_usersim():
    uenv = dict(os.environ)
    uenv["CUDA_VISIBLE_DEVICES"] = "1"
    ulogf = open(os.path.join(LOGS, "oblit_usersim.log"), "w")
    usim = subprocess.Popen([BINARY, "--model", os.path.join(LLMS, USER_SIM_FILE),
                             "--flash-attn", "on", "--host", "127.0.0.1", "--port", "8081",
                             "--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
                             "--threads", "4", "--threads-batch", "4",
                             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                             "--parallel", "2", "--temp", "0.0", "-n", "4096"],
                            env=uenv, stdout=ulogf, stderr=subprocess.STDOUT, text=True)
    for _ in range(150):
        time.sleep(2)
        if usim.poll() is not None:
            raise RuntimeError("user sim died")
        try:
            urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=3)
            log("  user sim up (3060:8081)")
            return usim, ulogf
        except Exception:
            pass
    usim.kill(); ulogf.close()
    raise RuntimeError("user sim timeout")


def main():
    assert os.path.exists(MODEL), "GGUF missing"
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(5)
    try:
        # 1. speed probe
        proc, logf = serve(["--chat-template-kwargs", '{"enable_thinking": false}'], "speed")
        sp = speed_probe()
        log(f"speed: {sp}")
        stop(proc, logf)

        progress = json.load(open(PROGRESS_FILE))
        e = next((m for m in progress["models"] if m["name"] == NAME), None)
        if e is None:
            e = {"name": NAME, "file": os.path.basename(MODEL), "category": "27B-dense",
                 "gpu": "V100", "engine": "llama.cpp", "thinking": False, "template": "stock"}
            progress["models"].append(e)
        e["decode_tps"] = sp.get("decode_tps")
        e["quant_source"] = "mradermacher GGUF (medismera safetensors upstream)"
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp, PROGRESS_FILE)

        # 2. sanity gate
        glog = os.path.join(LOGS, "oblit_gate.log")
        with open(glog, "w") as lf:
            subprocess.run([BENCH_PY, GATE, "--model-file", os.path.basename(MODEL),
                            "--name", NAME, "--gpu", "0", "--benchmarks", "sanity:25"],
                           stdout=lf, stderr=subprocess.STDOUT, timeout=3600)
        frac = re.findall(r"(\d+)\s*/\s*25", open(glog).read())
        san = {"sanity_passed": int(frac[-1]), "sanity_pct": round(100 * int(frac[-1]) / 25, 1)} if frac else {"error": "no score"}
        log(f"sanity: {san}")

        # 3. LCB (thinking off via allowlist - qwen38 token)
        proc, logf = serve([], "lcb")
        usim, ulogf = start_usersim()
        try:
            lenv = dict(os.environ)
            lenv.update({"OPENAI_KEY": "none", "OPENAI_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
                         "HF_ALLOW_CODE_EVAL": "1", "LCB_DISABLE_THINKING": "1"})
            out_name = NAME
            out_dir = os.path.join(LCB_DIR, "output", out_name)
            t0 = time.time()
            with open(os.path.join(LOGS, "oblit_lcb.log"), "w") as lf:
                subprocess.run([BENCH_PY, "-m", "lcb_runner.runner.main",
                                "--model", "local/qwen38-obliterated-mythos", "--scenario", "codegeneration",
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
            log(f"LCB: pass@1={pass1} ({wall/60:.0f} min) empty={empty}/{n}")
            stop(proc, logf)

            # 4. tau2 (thinking off via server kwarg = empty-turn fix)
            proc, logf = serve(["--chat-template-kwargs", '{"enable_thinking": false}'], "tau2")
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
            with open(os.path.join(LOGS, "oblit_tau2.log"), "w") as lf:
                subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=86400,
                               env=tenv, cwd=TAU2_DIR)
            tau_elapsed = round(time.time() - t0, 1)
            sdir = os.path.join(TAU2_DIR, "data", "simulations", f"tau2_{safe}", "results.json")
            d = json.load(open(sdir))
            per_task = [(s.get("reward_info") or {}).get("reward") for s in d["simulations"]]
            scored = [r for r in per_task if r is not None]
            tau = {"reward": sum(r or 0 for r in per_task) / len(per_task),
                   "task_pass_rate": None, "wall_time_s": tau_elapsed,
                   "user_sim": USER_SIM_NAME, "canonical": True,
                   "sims_scored": f"{len(scored)}/15",
                   "passes": sum(1 for r in scored if r == 1.0),
                   "infra_errors": sum(1 for s in d["simulations"] if s.get("termination_reason") == "infrastructure_error"),
                   "serving": "enable_thinking:false via chat-template-kwargs"}
            log(f"tau2: all-15={tau['reward']:.4f} passes={tau['passes']}/15 "
                f"infra={tau['infra_errors']} ({tau_elapsed/60:.0f} min)")
        finally:
            stop(proc, logf)
            usim.terminate()
            try:
                usim.wait(timeout=10)
            except Exception:
                usim.kill()
            ulogf.close()

        progress = json.load(open(PROGRESS_FILE))
        e = next(m for m in progress["models"] if m["name"] == NAME)
        e["benchkit"] = {"sanity": san}
        e["livecodebench"] = {"pass_at_1": pass1, "empty": empty, "of": n, "wall_time_s": wall}
        e["tau2"] = tau
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(progress, f, indent=2)
        os.replace(tmp, PROGRESS_FILE)
        log("progress.json updated - LANE COMPLETE")
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "caimlas-qwythos"], capture_output=True)
        time.sleep(15)
    log("DONE")


if __name__ == "__main__":
    main()
