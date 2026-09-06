#!/usr/bin/env python3
"""
Re-benchmark 7 models with the peculiar-ragdoll Qwen-Sharp chat template (v22.1).
Template applied at runtime via --chat-template-file (no GGUF rewrite).
Runs on V100: tok/s probe -> LCB (75) -> tau2 (15, LFM user sim on :8082).
Models: Bonsai-27B(dspark/PrismML), Qwythos-27B-MTP, Qwen3.8-27B-Q4_K_M-MTP,
Nail-35B-A3B(MoE, cpu-moe), ThinkingCap-27B, BTL-3-Full-27B, RavenX-35B-A3B(MoE).

Usage:
    /home/caimlas/bench-venv/bin/python scripts/run_sharp_template.py
    (start LFM user sim on 3060:8082 first)
"""
import subprocess, json, time, os, urllib.request, shutil, re
from datetime import datetime, timezone

BINARY_UPSTREAM = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
BINARY_PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18099
USER_PORT = 8082
SCRATCH = "/tmp/coding-bench"
RESULTS = os.path.join(SCRATCH, "results")
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TAU2_DIR = "/home/caimlas/git/tau2-bench"
TEMPLATE = "/home/caimlas/llm-benchmarks/templates/sharp_chat_template.jinja"

V100_SERVICE = "caimlas-bonsai"

MODELS = [
    {
        "name": "Ternary-Bonsai-27B Q2_0 (dspark) [Sharp]",
        "file": "Ternary-Bonsai-27B-Q2_0.gguf",
        "draft_file": "Ternary-Bonsai-27B-dspark-Q4_1.gguf",
        "lcb_model": "local/ternary-bonsai-27b",
        "binary": BINARY_PRISMML,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q4_0", "--cache-type-v", "q4_0"],
        "category": "27B Ternary",
        "thinking": True,
        "template": TEMPLATE,
    },
    {
        "name": "Qwythos-27B-MTP Q4_K_M [Sharp]",
        "file": "Qwythos-27B-MTP-Q4_K_M.gguf",
        "lcb_model": "local/qwythos-27b-mtp",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                 "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        "category": "27B Dense",
        "thinking": True,
        "template": TEMPLATE,
    },
    {
        "name": "Qwen3.8-27B Q4_K_M MTP [Sharp]",
        "file": "Qwen3.8-27B-Q4_K_M.gguf",
        "lcb_model": "local/qwen38-27b-q4km-mtp",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0",
                 "--spec-type", "draft-mtp", "--spec-draft-n-max", "3"],
        "category": "27B Dense",
        "thinking": True,
        "template": TEMPLATE,
    },
    {
        "name": "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL [Sharp]",
        "file": "Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "lcb_model": "local/nail-35b",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "MoE 35B",
        "thinking": True,
        "template": TEMPLATE,
    },
    {
        "name": "ThinkingCap-Qwen3.6-27B Q4_K_M [Sharp]",
        "file": "ThinkingCap-Qwen3.6-27B-Q4_K_M.gguf",
        "lcb_model": "local/thinkingcap-qwen36-27b",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "131072", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "27B Dense",
        "thinking": True,
        "template": TEMPLATE,
    },
    {
        "name": "BTL-3 Full Q4_K_M [Sharp]",
        "file": "BTL-3-merged-Q4_K_M.gguf",
        "lcb_model": "local/btl3-full-q4km",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "27B Dense",
        "thinking": True,
        "template": TEMPLATE,
    },
    {
        "name": "Ornith-1.5-35B-A3B Q4_K_M",
        "file": "Ornith-1.5-35B-Q4_K_M.gguf",
        "lcb_model": "local/ornith-15-35b",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "MoE 35B",
        "thinking": True,
        "template": None,  # stock embedded template
    },
    {
        "name": "RavenX-OpenFable-Holo3 Q4_K_M [Sharp]",
        "file": "RavenX-OpenFable-Qwopus-Coder-Holo3-Qwen3.6-35B-A3B-MTP-Q4_K_M.gguf",
        "lcb_model": "local/ravenx-holo3",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "MoE 35B",
        "thinking": True,
        "template": TEMPLATE,
    },
    {
        "name": "Ornith-1.5-35B-A3B Q4_K_M [Sharp]",
        "file": "Ornith-1.5-35B-Q4_K_M.gguf",
        "lcb_model": "local/ornith-15-35b",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--cpu-moe", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "MoE 35B",
        "thinking": True,
        "template": TEMPLATE,
    },
    {
        "name": "Carnice-V3 Q4_K_M",
        "file": "Carnice-V3-Q4_K_M.gguf",
        "lcb_model": "local/carnice-v3",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "27B Dense",
        "thinking": True,
        "template": None,  # stock embedded template
    },
    {
        "name": "Carnice-V3 Q4_K_M [Sharp]",
        "file": "Carnice-V3-Q4_K_M.gguf",
        "lcb_model": "local/carnice-v3",
        "binary": BINARY_UPSTREAM,
        "args": ["--gpu-layers", "99", "--ctx-size", "262144", "--ubatch-size", "512",
                 "--threads", "8", "--threads-batch", "8",
                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"],
        "category": "27B Dense",
        "thinking": True,
        "template": TEMPLATE,
    },
]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def stop_v100():
    subprocess.run(["sudo", "systemctl", "stop", V100_SERVICE], capture_output=True, timeout=15)
    time.sleep(3)

def start_v100():
    subprocess.run(["sudo", "systemctl", "start", V100_SERVICE], capture_output=True, timeout=15)
    for _ in range(60):
        time.sleep(3)
        try:
            r = urllib.request.urlopen("http://localhost:8081/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return
        except:
            pass

def start_model(model):
    binary = model.get("binary", BINARY_UPSTREAM)
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [binary, "--model", path, "--flash-attn", "on",
           "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
           "--parallel", "2", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model["args"])
    if model.get("draft_file"):
        draft_path = os.path.join(LLMS_DIR, model["draft_file"])
        cmd.extend(["--spec-draft-model", draft_path, "--spec-draft-n-max", "4"])
    # Sharp template applied at runtime
    if model.get("template"):
        cmd.extend(["--chat-template-file", model["template"]])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = "0"  # V100
    safe = model["name"].replace(" ", "_").replace("[", "").replace("]", "")
    logf = open(os.path.join(LOGS, f"{safe}_sharp_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                # verify template actually applied
                try:
                    props = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=3).read())
                    tmpl = props.get("chat_template", "")
                    if "Answer directly, after thinking" not in tmpl:
                        print(f"  WARNING: Sharp template NOT active on server!")
                    else:
                        print(f"  Sharp template verified active ({len(tmpl)} chars)")
                except Exception:
                    pass
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{safe}_sharp_server.log")) as f:
                err = f.read()[-500:]
            return None, None, f"Server died: {err}"
    proc.kill()
    return None, None, "Timeout"

def kill_model(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def timed_completion(prompt, max_tokens=256):
    import http.client
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=120)
    payload = json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": 0.3, "stream": False,
    })
    start = time.time()
    conn.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = json.loads(resp.read())
    elapsed = time.time() - start
    usage = data.get("usage", {})
    toks = usage.get("completion_tokens", max_tokens)
    return toks / elapsed if elapsed > 0 else 0, toks

def probe_tps(model):
    try:
        timed_completion("Write a short hello world function.", max_tokens=8)
        tps, toks = timed_completion(
            "Write a detailed essay about the history of computing, from Babbage to modern GPUs.",
            max_tokens=256)
        print(f"  tok/s: {tps:.1f} ({toks} tokens)")
        return round(tps, 1)
    except Exception as e:
        print(f"  TPS probe failed: {e}")
        return None

def run_livecodebench(model_name, lcb_model):
    safe = model_name.replace(" ", "_").replace("[", "").replace("]", "")
    logpath = os.path.join(LOGS, f"{safe}_sharp_lcb.log")

    for dirname in [lcb_model.replace("/", "_"), model_name, model_name.replace(" [Sharp]", "")]:
        d = os.path.join(LCB_DIR, "output", dirname)
        if os.path.exists(d):
            shutil.rmtree(d)

    cmd = [
        BENCH_PY, "-m", "lcb_runner.runner.main",
        "--model", lcb_model,
        "--scenario", "codegeneration",
        "--release_version", "release_latest",
        "--n", "1",
        "--temperature", "0.0",
        "--max_tokens", "4096",
        "--num_problems", "75",
        "--openai_timeout", "300",
        "--evaluate",
    ]

    env = dict(os.environ)
    env["OPENAI_KEY"] = "none"
    env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{PORT}/v1"
    env["HF_ALLOW_CODE_EVAL"] = "1"
    env["LCB_DISABLE_THINKING"] = "1"

    print(f"  Running LiveCodeBench (75 problems, thinking off)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=36000, env=env, cwd=LCB_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"pass_at_1": None, "wall_time_s": 36000, "error": "timeout"}
    except Exception as e:
        return {"pass_at_1": None, "wall_time_s": time.time() - start, "error": str(e)}

    pass1 = None
    # candidate output dirs: model key, full name, name minus Sharp suffix, sanitized variants
    base = model_name.replace(" [Sharp]", "")
    candidates = [lcb_model.replace("/", "_"), model_name, base,
                  base.replace(" ", "_"), base.replace(" ", "_").replace("(", "").replace(")", "")]
    # also glob anything matching the first two words of the base name
    import glob as _g
    first_words = "-".join(base.split()[:2]).replace("(", "").replace(")", "")
    candidates += [os.path.basename(d) for d in _g.glob(os.path.join(LCB_DIR, "output", f"*{first_words}*"))]
    seen = set()
    for dirname in [c for c in candidates if c and c not in seen and not seen.add(c)]:
        output_dir = os.path.join(LCB_DIR, "output", dirname)
        for root, dirs, files in os.walk(output_dir):
            for f in files:
                if f.endswith("_eval.json"):
                    with open(os.path.join(root, f)) as rf:
                        try:
                            data = json.load(rf)
                            if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], dict):
                                pass1 = data[0].get("pass@1")
                        except:
                            pass
        if pass1 is not None:
            break

    print(f"  LCB pass@1: {pass1}  ({elapsed/60:.0f} min)")
    return {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1), "exit_code": result.returncode}

def run_tau2(model_name, model_file, max_steps=30):
    safe = model_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    logpath = os.path.join(LOGS, f"{safe}_sharp_tau2.log")
    save_dir = f"tau2_sharp_{safe}"

    agent_model = f"openai/{safe}"
    user_model = "openai/LFM2.5-8B-A1B-Clean-RealWorld-v2"

    cmd = [
        "uv", "run", "tau2", "run",
        "--domain", "airline",
        "--agent-llm", agent_model,
        "--agent-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://127.0.0.1:{PORT}/v1",
            "temperature": 0.0,
        }),
        "--user-llm", user_model,
        "--user-llm-args", json.dumps({
            "api_key": "none",
            "api_base": f"http://localhost:{USER_PORT}/v1",
        }),
        "--num-tasks", "15",
        "--num-trials", "1",
        "--max-concurrency", "2",
        "--max-steps", str(max_steps),
        "--max-errors", "5",
        "--timeout", "300",
        "--seed", "42",
        "--save-to", save_dir,
    ]

    env = dict(os.environ)
    env["OPENAI_API_KEY"] = "none"

    print(f"  Running tau2-bench (airline, 15 tasks)...")
    start = time.time()
    try:
        with open(logpath, "w") as lf:
            result = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                    timeout=86400, env=env, cwd=TAU2_DIR)
        elapsed = time.time() - start
    except subprocess.TimeoutExpired:
        return {"reward": None, "wall_time_s": 86400, "error": "timeout"}
    except Exception as e:
        return {"reward": None, "wall_time_s": time.time() - start, "error": str(e)}

    reward = None
    task_pass = None
    with open(logpath) as f:
        content = f.read()
    reward_match = re.search(r'Average Reward\s+([\d.]+)', content)
    if reward_match:
        reward = float(reward_match.group(1))
    pass_match = re.search(r'Task Pass Rate\s+([\d.]+)', content)
    task_pass = float(pass_match.group(1)) if pass_match else None

    print(f"  tau2 reward: {reward}  ({elapsed/60:.0f} min)")
    return {
        "reward": reward,
        "task_pass_rate": task_pass,
        "wall_time_s": round(elapsed, 1),
        "exit_code": result.returncode,
    }

def main():
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    progress = load_progress()

    done_names = set()
    for m in progress["models"]:
        if m.get("livecodebench", {}).get("pass_at_1") is not None and m.get("tau2", {}).get("reward") is not None:
            done_names.add(m["name"])

    for model in MODELS:
        path = os.path.join(LLMS_DIR, model["file"])
        if not os.path.exists(path):
            print(f"SKIP: {model['file']} not on disk")
            continue
        if model["name"] in done_names:
            print(f"SKIP: {model['name']} (already done)")
            continue

        print(f"\n{'='*70}")
        print(f"  {model['name']}")
        print(f"{'='*70}")

        mr = None
        for m in progress["models"]:
            if m["name"] == model["name"]:
                mr = m
                break
        if mr is None:
            mr = {"name": model["name"], "file": model["file"],
                  "category": model.get("category", ""), "gpu": "V100",
                  "template": "sharp"}
            progress["models"].append(mr)

        try:
            r = urllib.request.urlopen(f"http://localhost:{USER_PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") != "ok":
                print(f"  WARNING: User simulator on :{USER_PORT} not healthy!")
        except:
            print(f"  WARNING: User simulator on :{USER_PORT} not reachable! (tau2 will fail)")

        stop_v100()

        proc, logf, err = start_model(model)
        if err:
            print(f"  FATAL: Server failed to start: {err[:200]}")
            if "failures" not in mr:
                mr["failures"] = []
            mr["failures"].append({
                "benchmark": "all",
                "error": str(err)[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            mr["lcb_error"] = err
            save_progress(progress)
            start_v100()
            continue

        mr["template"] = "sharp" if model.get("template") else "stock"
        mr["thinking"] = model.get("thinking", True)
        mr["dspark"] = bool(model.get("draft_file"))
        mr["mtp_enabled"] = "--spec-type" in " ".join(model["args"])

        try:
            tps = probe_tps(model)
            mr["decode_tps"] = tps
            save_progress(progress)

            lcb = run_livecodebench(model["name"], model["lcb_model"])
            mr["livecodebench"] = lcb
            save_progress(progress)

            tau2 = run_tau2(model["name"], model["file"], max_steps=30)
            mr["tau2"] = tau2
            save_progress(progress)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            if "failures" not in mr:
                mr["failures"] = []
            mr["failures"].append({
                "benchmark": "unknown",
                "error": str(e)[:500],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        finally:
            kill_model(proc, logf)
            save_progress(progress)

        start_v100()

    print(f"\nSHARP TEMPLATE BENCHMARKS COMPLETE")

if __name__ == "__main__":
    main()
