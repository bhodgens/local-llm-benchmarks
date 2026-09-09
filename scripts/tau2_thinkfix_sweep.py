#!/usr/bin/env python3
"""tau2 sweep with enable_thinking:false passed to the agent server.

Validates the empty-turn fix on thinking-family agents and re-scores the
worst-affected rows. Models (all had infra_error sims):
  1. Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M  (7 infra, validation target)
  2. Qwen3.6-27B-FableFusion-MTP Q4_K_M  (5 infra)
  3. Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M (4 infra)
  4. Ternary-Bonsai-27B Q2_0 (dspark)    (4 infra; PrismML + -fit off)

Mechanics: agent served with --jinja + --chat-template-kwargs
'{"enable_thinking": false}'. Bonsai dspark keeps -fit off (chat-template-kwargs
supported on PrismML too; Bonsai verified thinking-suppressible earlier).
User sim LFM2.5 on 3060; airline/15/seed42; all-15 denominator from results.json.
caimlas-qwythos stopped for the duration, restored at the end.
"""
import subprocess, json, time, os, re, urllib.request

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
PORT = 18096
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

USER_SIM_FILE = "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
USER_SIM_NAME = "LFM2.5-8B-A1B-Clean-RealWorld-v2"

MODELS = [
    # (entry_name, gguf_path, server_binary, extra_args, tag)
    ("Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M",
     os.path.join(LLMS, "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf"), BINARY, [], "qwopus35b"),
    ("Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M",
     os.path.join(LLMS, "Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf"), BINARY, [], "heretic35b"),
    ("Ternary-Bonsai-27B Q2_0 (dspark)",
     os.path.join(LLMS, "Ternary-Bonsai-27B-Q2_0.gguf"), PRISMML,
     ["-fit", "off",
      "--spec-draft-model", os.path.join(LLMS, "Ternary-Bonsai-27B-dspark-Q4_1.gguf"),
      "--spec-draft-n-max", "4",
      "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"], "bonsai"),
]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def wait_health(proc, logf, tag):
    for _ in range(300):
        time.sleep(2)
        if proc.poll() is not None:
            raise RuntimeError(f"{tag} died: " + open(logf.name).read()[-300:])
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            return
        except Exception:
            pass
    proc.kill(); logf.close()
    raise RuntimeError(f"{tag} timeout")


def main():
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(5)
    summary = []
    try:
        uenv = dict(os.environ)
        uenv["CUDA_VISIBLE_DEVICES"] = "1"
        ulogf = open(os.path.join(LOGS, "thinkfix_usersim.log"), "w")
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
                break
            except Exception:
                pass
        else:
            usim.kill(); raise RuntimeError("user sim timeout")
        log("user sim up (3060:8081)")

        for name, model, binary, extra, tag in MODELS:
            if not os.path.exists(model):
                log(f"SKIP {name}: file missing")
                continue
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = "0"
            logf = open(os.path.join(LOGS, f"thinkfix_{tag}_server.log"), "w")
            proc = subprocess.Popen([binary, "--model", model,
                                     "--flash-attn", "on", "--host", "127.0.0.1", "--port", str(PORT),
                                     "--gpu-layers", "99", "--ctx-size", "32768", "--ubatch-size", "512",
                                     "--threads", "8", "--threads-batch", "8",
                                     "--parallel", "2", "--temp", "0.0", "-n", "4096",
                                     "--jinja",
                                     "--chat-template-kwargs", '{"enable_thinking": false}'] + extra,
                                    env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
            try:
                wait_health(proc, logf, tag)
                # per-model identity check
                props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
                base = os.path.basename(model).replace(".gguf", "")
                key = "Ternary-Bonsai-27B-Q2_0" if "bonsai" == tag else base
                assert key in props or base.split("-MTP")[0] in props, f"wrong model on port: {props[:120]}"
                log(f"{name}: server up, identity OK, thinking OFF")

                safe = re.sub(r"[/()\[\]]", "", name.replace(" ", "_"))
                cmd = ["uv", "run", "tau2", "run", "--domain", "airline",
                       "--agent-llm", f"openai/{safe}",
                       "--agent-llm-args", json.dumps({"api_key": "none",
                           "api_base": f"http://127.0.0.1:{PORT}/v1", "temperature": 0.0}),
                       "--user-llm", f"openai/{USER_SIM_NAME}",
                       "--user-llm-args", json.dumps({"api_key": "none",
                           "api_base": "http://127.0.0.1:8081/v1"}),
                       "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
                       "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
                       "--seed", "42", "--save-to", f"tau2_{safe}_thinkfix"]
                tenv = dict(os.environ)
                tenv["OPENAI_API_KEY"] = "none"
                t0 = time.time()
                with open(os.path.join(LOGS, f"thinkfix_{tag}.log"), "w") as lf:
                    subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=86400,
                                   env=tenv, cwd=TAU2_DIR)
                elapsed = round(time.time() - t0, 1)

                sdir = os.path.join(TAU2_DIR, "data", "simulations", f"tau2_{safe}_thinkfix", "results.json")
                d = json.load(open(sdir))
                per_task = [(s.get("reward_info") or {}).get("reward") for s in d["simulations"]]
                scored = [r for r in per_task if r is not None]
                final = sum(r or 0 for r in per_task) / len(per_task)
                passes = sum(1 for r in scored if r == 1.0)
                infra = sum(1 for s in d["simulations"] if s.get("termination_reason") == "infrastructure_error")
                log(f"{name}: all-15={final:.4f} passes={passes}/15 scored={len(scored)} "
                    f"infra_errors={infra} ({elapsed/60:.0f} min)")
                summary.append((name, final, passes, infra, scored))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except Exception:
                    proc.kill()
                logf.close()
                time.sleep(3)

        usim.terminate()
        try:
            usim.wait(timeout=10)
        except Exception:
            usim.kill()
        ulogf.close()

        # record all results
        p = json.load(open(PROGRESS_FILE))
        for name, final, passes, infra, scored in summary:
            e = next((mm for mm in p["models"] if mm["name"] == name), None)
            if e is None:
                log(f"no progress entry for {name}; skipping record")
                continue
            old = (e.get("tau2") or {}).get("reward")
            e["tau2"] = {"reward": final, "task_pass_rate": None,
                         "user_sim": USER_SIM_NAME, "canonical": True,
                         "sims_scored": f"{len(scored)}/15", "passes": passes,
                         "infra_errors": infra,
                         "serving": "enable_thinking:false via chat-template-kwargs"}
            e.setdefault("failures", []).append({
                "benchmark": "tau2 (thinking-off rerun)",
                "error": (f"Served with enable_thinking:false (empty-turn fix). "
                          f"Result {final:.4f} (passes={passes}/15, infra_errors={infra}). "
                          f"Superseded {old}. Raw: {scored}"),
                "timestamp": TS})
            log(f"recorded {name}: {old} -> {final}")
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(p, f, indent=2)
        os.replace(tmp, PROGRESS_FILE)
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "caimlas-qwythos"], capture_output=True)
        time.sleep(15)

    log("=== SWEEP SUMMARY ===")
    for name, final, passes, infra, scored in summary:
        print(f"  {name:<46} all-15={final:.4f} passes={passes} infra={infra} scored={len(scored)}")


if __name__ == "__main__":
    main()
