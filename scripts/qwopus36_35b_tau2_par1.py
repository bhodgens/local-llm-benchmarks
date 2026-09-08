#!/usr/bin/env python3
"""Qwopus3.6-35B-Coder-MTP tau2, attempt 3: --parallel 1.
Both prior runs (MTP and no-MTP) had infra errors under --parallel 2.
Suspect: per-slot state (SSM hybrid layers) wedges under concurrency 2 on
this model. parallel 1 + tau2 max-concurrency 1 = serialized, slow but safe.
Passes as argv so the same harness logic is reused."""
import subprocess, json, time, os, re, urllib.request, sys

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
PORT = 18096
NAME = "Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M"
MODEL = os.path.join(LLMS, "Qwopus3.6-35B-A3B-Coder-MTP-Q4_K_M.gguf")
TAG = "par1"
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(5)
    try:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        logf = open(os.path.join(LOGS, f"qwopus36_35b_tau2_{TAG}_server.log"), "w")
        proc = subprocess.Popen([BINARY, "--model", MODEL,
                                 "--flash-attn", "on", "--host", "127.0.0.1", "--port", str(PORT),
                                 "--gpu-layers", "99", "--ctx-size", "65536", "--ubatch-size", "512",
                                 "--threads", "8", "--threads-batch", "8",
                                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                                 "--parallel", "1", "--cont-batching", "--temp", "0.0",
                                 "-n", "4096", "--jinja"],
                                env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        up = False
        for _ in range(300):
            time.sleep(2)
            if proc.poll() is not None:
                raise RuntimeError("agent died: " + open(logf.name).read()[-300:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                up = True
                break
            except Exception:
                pass
        if not up:
            proc.kill()
            raise RuntimeError("agent timeout")
        props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
        assert "Qwopus3.6-35B-A3B-Coder-MTP" in props, "wrong model"
        log("agent up (V100:18096, parallel 1, no MTP)")

        uenv = dict(os.environ)
        uenv["CUDA_VISIBLE_DEVICES"] = "1"
        ulogf = open(os.path.join(LOGS, f"qwopus36_35b_tau2_{TAG}_usersim.log"), "w")
        usim = subprocess.Popen([BINARY, "--model", os.path.join(LLMS, "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"),
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
            usim.kill()
            raise RuntimeError("user sim timeout")
        log("user sim up (3060:8081)")

        safe = re.sub(r"[/()\[\]]", "", NAME.replace(" ", "_"))
        cmd = ["uv", "run", "tau2", "run", "--domain", "airline",
               "--agent-llm", f"openai/{safe}",
               "--agent-llm-args", json.dumps({"api_key": "none",
                   "api_base": f"http://127.0.0.1:{PORT}/v1", "temperature": 0.0}),
               "--user-llm", "openai/LFM2.5-8B-A1B-Clean-RealWorld-v2",
               "--user-llm-args", json.dumps({"api_key": "none",
                   "api_base": "http://127.0.0.1:8081/v1"}),
               "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "1",
               "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
               "--seed", "42", "--save-to", f"tau2_{safe}_{TAG}"]
        tenv = dict(os.environ)
        tenv["OPENAI_API_KEY"] = "none"
        t0 = time.time()
        with open(os.path.join(LOGS, f"qwopus36_35b_tau2_{TAG}.log"), "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=86400,
                           env=tenv, cwd=TAU2_DIR)
        elapsed = round(time.time() - t0, 1)

        sdir = os.path.join(TAU2_DIR, "data", "simulations", f"tau2_{safe}_{TAG}", "results.json")
        d = json.load(open(sdir))
        per_task = [(s.get("reward_info") or {}).get("reward") for s in d["simulations"]]
        scored = [r for r in per_task if r is not None]
        final = sum(r or 0 for r in per_task) / len(per_task)
        passes = sum(1 for r in scored if r == 1.0)
        infra = sum(1 for s in d["simulations"] if s.get("termination_reason") == "infrastructure_error")
        log(f"tau2 (parallel1): all-15={final:.4f} passes={passes}/15 scored={len(scored)} "
            f"infra_errors={infra} ({elapsed/60:.0f} min)")

        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        logf.close()
        usim.terminate()
        try:
            usim.wait(timeout=10)
        except Exception:
            usim.kill()
        ulogf.close()

        p = json.load(open(PROGRESS_FILE))
        e = next(mm for mm in p["models"] if mm["name"] == NAME)
        old = (e.get("tau2") or {}).get("reward")
        e["tau2"] = {"reward": final, "task_pass_rate": None, "wall_time_s": elapsed,
                     "exit_code": 0, "user_sim": "LFM2.5-8B-A1B-Clean-RealWorld-v2",
                     "canonical": True, "sims_scored": f"{len(scored)}/15",
                     "passes": passes, "infra_errors": infra,
                     "serving": "parallel 1, max-concurrency 1, no MTP, --jinja default"}
        if infra > 0:
            e["failures"].append({
                "benchmark": "tau2 root-cause analysis (2026-09-08)",
                "error": ("infra_error sims are NOT server crashes: tau2 raises "
                          "'UserMessage/AssistantMessage must have either content or tool_calls' "
                          "when a turn comes back EMPTY (4 attempts, then task = infra_error, "
                          "reward None). Split across par1 run: 3x agent-empty (think-tag model "
                          "emitted think-only/empty under --jinja default), 4x user-sim-empty "
                          "(LFM interaction). MTP/parallel ruled out: infra errors occur with "
                          "MTP+par2 (4), no-MTP+par2 (5), no-MTP+par1 (7) - highest WITHOUT MTP. "
                          "Fix for future runs: pass chat_template_kwargs enable_thinking:false "
                          "to the agent server (template supports it: pre-fills <think></think>)."),
                "timestamp": TS})
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(p, f, indent=2)
        os.replace(tmp, PROGRESS_FILE)
        log(f"progress.json: tau2 {old} -> {final}")
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "caimlas-qwythos"], capture_output=True)
        time.sleep(15)
    log("DONE")


if __name__ == "__main__":
    main()
