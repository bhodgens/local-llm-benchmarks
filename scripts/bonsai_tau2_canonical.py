#!/usr/bin/env python3
"""Canonical tau2 rerun for Ternary-Bonsai-27B Q2_0 (dspark):
LFM2.5-Clean user sim (same as all other rows), airline/15/seed42/conc2/30,
all 15 tasks scored (errored sims = 0). Replaces the 0.800 (10-task
scored-only denominator, non-canonical user sim) on the progress entry.
Serving: PrismML fork + dspark draft n=4 + -fit off, exclusive V100."""
import subprocess, json, time, os, re, urllib.request

PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
LLMS = "/home/files/llms"
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
PORT = 18096
NAME = "Ternary-Bonsai-27B Q2_0 (dspark)"
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    subprocess.run(["sudo", "-n", "systemctl", "stop", "caimlas-qwythos"], capture_output=True)
    time.sleep(5)
    try:
        # agent server: Bonsai dspark on V100
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0"
        logf = open(os.path.join(LOGS, "bonsai_tau2_canonical_server.log"), "w")
        proc = subprocess.Popen([PRISMML, "--model", os.path.join(LLMS, "Ternary-Bonsai-27B-Q2_0.gguf"),
                                 "--flash-attn", "on", "--host", "127.0.0.1", "--port", str(PORT),
                                 "--gpu-layers", "99", "--ctx-size", "32768", "--ubatch-size", "512",
                                 "--threads", "8", "--threads-batch", "8",
                                 "--cache-type-k", "q4_0", "--cache-type-v", "q4_0",
                                 "-fit", "off", "--parallel", "2", "--temp", "0.0", "-n", "4096",
                                 "--jinja",
                                 "--spec-draft-model", os.path.join(LLMS, "Ternary-Bonsai-27B-dspark-Q4_1.gguf"),
                                 "--spec-draft-n-max", "4"],
                                env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
        up = False
        for _ in range(240):
            time.sleep(2)
            if proc.poll() is not None:
                raise RuntimeError("agent server died: " + open(logf.name).read()[-300:])
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
                up = True
                break
            except Exception:
                pass
        if not up:
            proc.kill()
            raise RuntimeError("agent server timeout")
        props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
        assert "Ternary-Bonsai-27B-Q2_0" in props, "wrong model on port"
        log("agent server up (V100:18096, dspark n=4, -fit off), identity OK")

        # user sim: LFM2.5-Clean on 3060:8081 (canonical)
        uenv = dict(os.environ)
        uenv["CUDA_VISIBLE_DEVICES"] = "1"
        ulogf = open(os.path.join(LOGS, "bonsai_tau2_canonical_usersim.log"), "w")
        usim = subprocess.Popen(["/home/caimlas/git/llama.cpp/build/bin/llama-server",
                                 "--model", os.path.join(LLMS, "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"),
                                 "--flash-attn", "on", "--host", "127.0.0.1", "--port", "8081",
                                 "--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
                                 "--threads", "4", "--threads-batch", "4",
                                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                                 "--parallel", "2", "--temp", "0.0", "-n", "4096"],
                                env=uenv, stdout=ulogf, stderr=subprocess.STDOUT, text=True)
        for _ in range(150):
            time.sleep(2)
            if usim.poll() is not None:
                raise RuntimeError("user sim died: " + open(ulogf.name).read()[-300:])
            try:
                urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=3)
                break
            except Exception:
                pass
        else:
            usim.kill()
            raise RuntimeError("user sim timeout")
        log("user sim up (3060:8081, LFM2.5-Clean) - canonical")

        safe = re.sub(r"[/()\[\]]", "", NAME.replace(" ", "_"))
        cmd = ["uv", "run", "tau2", "run", "--domain", "airline",
               "--agent-llm", f"openai/{safe}",
               "--agent-llm-args", json.dumps({"api_key": "none",
                   "api_base": f"http://127.0.0.1:{PORT}/v1", "temperature": 0.0}),
               "--user-llm", "openai/LFM2.5-8B-A1B-Clean-RealWorld-v2",
               "--user-llm-args", json.dumps({"api_key": "none",
                   "api_base": "http://127.0.0.1:8081/v1"}),
               "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
               "--max-steps", "30", "--max-errors", "5", "--timeout", "300",
               "--seed", "42", "--save-to", f"tau2_{safe}_lfm_canonical"]
        tenv = dict(os.environ)
        tenv["OPENAI_API_KEY"] = "none"
        t0 = time.time()
        with open(os.path.join(LOGS, "bonsai_tau2_canonical_run.log"), "w") as lf:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=86400,
                           env=tenv, cwd=TAU2_DIR)
        elapsed = round(time.time() - t0, 1)
        content = open(os.path.join(LOGS, "bonsai_tau2_canonical_run.log")).read()
        m = re.search(r"Average Reward\s+([\d.]+)", content)
        reward = float(m.group(1)) if m else None

        # authoritative per-task read from the simulation file
        sdir = os.path.join(TAU2_DIR, "data", "simulations", f"tau2_{safe}_lfm_canonical", "results.json")
        per_task = []
        if os.path.exists(sdir):
            d = json.load(open(sdir))
            per_task = [(s.get("reward_info") or {}).get("reward") for s in d["simulations"]]
        scored = [r for r in per_task if r is not None]
        log(f"tau2: reward={reward} ({elapsed/60:.0f} min)")
        log(f"  sims={len(per_task)} scored={len(scored)} passes={sum(1 for r in scored if r == 1.0)}")

        # sanity: canonical runs must have all 15 scored
        if len(per_task) == 15 and len(scored) == 15:
            final = sum(scored) / 15.0
            log(f"  canonical all-15 mean = {final:.4f}")
        else:
            final = reward
            log(f"  WARNING: {15 - len(scored)} sims unscored; reporting harness mean {reward}")

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
                     "canonical": True, "sims_scored": f"{len(scored)}/15"}
        e.setdefault("failures", []).append({
            "benchmark": "tau2 (canonical rerun)",
            "error": (f"Rerun with canonical protocol (LFM2.5 user sim, seed 42, all 15 scored): "
                      f"{final:.4f}. Superseded values: 0.800 (10-task scored-only denominator, "
                      f"Qwythos user sim) and 0.5333 (8/15 recount of that same run). "
                      f"Raw per-task rewards: {per_task}"),
            "timestamp": TS})
        tmp = PROGRESS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(p, f, indent=2)
        os.replace(tmp, PROGRESS_FILE)
        log(f"progress.json updated: tau2 {old} -> {final} (canonical)")
    finally:
        subprocess.run(["sudo", "-n", "systemctl", "start", "caimlas-qwythos"], capture_output=True)
        time.sleep(15)
    log("DONE")


if __name__ == "__main__":
    main()
