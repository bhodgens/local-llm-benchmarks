#!/usr/bin/env python3
"""tau2 think-ON rerun for LFM3 base + uncensored. Hypothesis: enable_thinking:false
on a reasoning-family model causes empty assistant turns -> 0.0 tau2.
Rerun with --jinja but WITHOUT --chat-template-kwargs. Prod stop/restore in finally.
Scores: reward_all15 (raw) + reward_valid (excludes empty_response/infra_error sims)."""
import json, os, re, shutil, subprocess, sys, time, glob

BIN = "/var/tmp/llms/llama-lfm3-build/build/bin/llama-server"
SIM_FILE = "/home/files/llms/LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
INTAKE = "/home/caimlas/llm-benchmarks/results/intake_20261105-061928.json"

MODELS = [
    {"key": "LFM3-1.7B Q4_K_M (3060)", "name": "LFM3-1.7B Q4_K_M (3060)",
     "file": "/var/tmp/llms/lfm3-base/LFM3-1.7B-Q4_K_M.gguf", "port": 18096},
    {"key": "LFM3-1.7B-Uncensored AD-Q6_K (3060)", "name": "LFM3-1.7B-Uncensored AD-Q6_K (3060)",
     "file": "/var/tmp/llms/lfm3-uncensored/LFM3-1.7B-Uncensored-AD-Q6_K.gguf", "port": 18096},
]

def sh(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def wait_health(port, tries=90):
    for _ in range(tries):
        time.sleep(5)
        r = sh(f"curl -s -o /dev/null -w '%{{http_code}}' http://127.0.0.1:{port}/health", timeout=15)
        if r.stdout.strip() == "200": return True
    return False

def chat_test(port, model):
    # NOTE: thinking-capable models (--jinja) can burn the entire budget on
    # <think> tokens, leaving content empty — a healthy response still has
    # choices[]. Pass on any well-formed completion response.
    r = sh(f"""curl -s http://127.0.0.1:{port}/v1/chat/completions -H 'Content-Type: application/json' -d '{{"model": "{model}", "messages": [{{"role":"user","content":"Say OK"}}], "max_tokens": 64}}'""", timeout=90)
    try:
        d = json.loads(r.stdout)
        return bool(d.get("choices"))
    except Exception:
        return False

def wait200(url, tries=60):
    for _ in range(tries):
        time.sleep(5)
        r = sh(f"curl -s -o /dev/null -w '%{{http_code}}' {url}", timeout=15)
        if r.stdout.strip() == "200": return True
    return False

OUT = {}
try:
    print("== stopping prod ==", flush=True)
    sh("sudo -n systemctl stop caimlas-gemma-4-26b-v100")
    sh("sudo -n systemctl stop caimlas-bonsai-3060")
    time.sleep(5)

    for mi, m in enumerate(MODELS):
        print(f"=== RERUN thinkON: {m['name']} ===", flush=True)
        sim_gpu = "0"          # sim on V100
        agent_gpu = "1"        # agent on 3060
        sim_port, ag_port = 8081, m["port"]
        sim_name = "LFM2.5-8B-A1B-Clean-RealWorld-v2"

        # start sim
        sim_cmd = [BIN, "-m", SIM_FILE, "--port", str(sim_port),
                   "-c", "32768", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1",
                   "--host", "127.0.0.1"]
        sim_env = dict(os.environ, CUDA_VISIBLE_DEVICES=sim_gpu)
        sim_p = subprocess.Popen(sim_cmd, env=sim_env, stdout=open(f"/tmp/rerun_sim_{mi}.log","w"),
                                 stderr=subprocess.STDOUT)
        if not wait_health(sim_port):
            OUT[m['name']] = {"error": "sim failed health"}; continue
        if not chat_test(sim_port, sim_name):
            OUT[m['name']] = {"error": "sim chat test empty"}; sim_p.kill(); continue

        # start agent — NO chat-template-kwargs => thinking ON
        ag_cmd = [BIN, "-m", m['file'], "--port", str(ag_port),
                  "-c", "65536", "-fa", "1", "--jinja", "-sm", "none", "--parallel", "1",
                  "--host", "127.0.0.1"]
        ag_env = dict(os.environ, CUDA_VISIBLE_DEVICES=agent_gpu)
        ag_p = subprocess.Popen(ag_cmd, env=ag_env, stdout=open(f"/tmp/rerun_ag_{mi}.log","w"),
                                stderr=subprocess.STDOUT)
        if not wait_health(ag_port):
            OUT[m['name']] = {"error": "agent failed health"}; sim_p.kill(); continue
        if not chat_test(ag_port, "lfm3"):
            OUT[m['name']] = {"error": "agent chat test empty (possible always-empty bug!)"}
            ag_p.kill(); sim_p.kill(); continue

        # query the agent's actual model id (GGUF internal name) instead of guessing
        mr = sh(f"curl -s http://127.0.0.1:{ag_port}/v1/models", timeout=20)
        try:
            ag_model_id = json.loads(mr.stdout)["data"][0]["id"]
        except Exception:
            ag_model_id = "lfm3"
        safe = re.sub(r'[^A-Za-z0-9]+', '_', m["name"]).strip('_') + "_thinkON"
        save_dir = os.path.join(TAU2_DIR, "data", "simulations", f"tau2_{safe}")
        if os.path.isdir(save_dir): shutil.rmtree(save_dir)
        cmd = ["uv", "run", "tau2", "run", "--domain", "airline",
               "--agent-llm", f"openai/{ag_model_id}",
               "--agent-llm-args", json.dumps({"api_key": "none", "api_base": f"http://127.0.0.1:{ag_port}/v1", "temperature": 0.0}),
               "--user-llm", f"openai/{sim_name}",
               "--user-llm-args", json.dumps({"api_key": "none", "api_base": f"http://localhost:{sim_port}/v1"}),
               "--num-tasks", "15", "--num-trials", "1", "--max-concurrency", "2",
               "--max-steps", "30", "--max-errors", "5", "--timeout", "300", "--seed", "42",
               "--save-to", save_dir]
        env = dict(os.environ); env["OPENAI_API_KEY"] = "none"
        logf = f"/tmp/tau2_{safe}.log"
        t0 = time.time()
        with open(logf, "w") as lf, open(os.devnull) as dn:
            subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=TAU2_DIR, env=env,
                           timeout=24*3600, stdin=dn)
        wall = round(time.time()-t0, 1)

        # parse
        rj = os.path.join(save_dir, "results.json")
        rec = {"wall_s": wall, "save_dir": save_dir, "log": logf}
        if os.path.exists(rj):
            data = json.load(open(rj))
            sims = data.get("simulations", [])
            terms = {}
            for s in sims:
                terms[s.get("termination_reason") or "?"] = terms.get(s.get("termination_reason") or "?", 0) + 1
            vals = [(s.get("reward_info") or {}).get("reward") for s in sims]
            scored = [v for v in vals if v is not None]
            excluded = sum(1 for s in sims if (s.get("termination_reason") in ("empty_response", "infrastructure_error")))
            valid_vals, valid_sims = [], []
            for s, v in zip(sims, vals):
                if s.get("termination_reason") not in ("empty_response", "infrastructure_error"):
                    valid_sims.append(s); 
                    if v is not None: valid_vals.append(v)
            rec.update({
                "reward_all15": round(sum(scored)/15.0, 4) if len(scored) == 15 else None,
                "reward_valid": round(sum(valid_vals)/len(valid_vals), 4) if valid_vals else None,
                "n_excluded": excluded, "n_scored": len(scored),
                "termination": terms, "sims": len(sims),
            })
        else:
            rec["error"] = "no results.json"
        # empty-msg diagnostic
        lg = open(logf).read()
        rec["empty_msg_count"] = lg.count("must have either content or tool_calls")
        OUT[m["name"]] = rec
        print(f"  -> {json.dumps(rec)}", flush=True)

        ag_p.kill(); sim_p.kill(); time.sleep(3)

finally:
    print("== restoring prod ==", flush=True)
    sh("sudo -n systemctl start caimlas-gemma-4-26b-v100")
    sh("sudo -n systemctl start caimlas-bonsai-3060")
    w1 = wait200("http://127.0.0.1:8082/v1/models")
    w2 = wait200("http://127.0.0.1:8083/v1/models")
    print(f"prod 200: 8082={w1} 8083={w2}", flush=True)

    # merge into intake json
    intake = json.load(open(INTAKE))
    for name, rec in OUT.items():
        if name in intake:
            intake[name]["tau2_rerun_thinkON"] = rec
    json.dump(intake, open(INTAKE, "w"), indent=1, default=str)
    json.dump(OUT, open("/home/caimlas/llm-benchmarks/results/tau2_rerun_20261105.json", "w"), indent=1, default=str)
    print("saved rerun results", flush=True)
