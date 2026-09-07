#!/usr/bin/env python3
"""
LCB empty-content remediation reruns (queue Tiers 1/2/4 + LFM Tier 3).

Per model: preserve broken artifacts as <dir>.thinking-artifact-broken, serve
on a DEDICATED port, verify /props identity, run LCB 75 with thinking off,
audit empty-rate, update progress.json (never overwrite a score with None),
record invalidation/repair notes in failures. Sequential, V100 lane; 3060
only if V100 can't fit the file.

Models whose GGUFs were deleted (Qwen3.8 Q4_K_S, DSV4-Flash, R1-8B q4/q8,
Nanbeige) are NOT rerunnable: handled separately with invalidation notes only.
"""
import subprocess, json, time, os, urllib.request, shutil, re, sys, glob

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18096  # dedicated; other harnesses use 18099
LOGS = "/tmp/coding-bench/logs"
PROGRESS_FILE = "/tmp/coding-bench/progress.json"
BENCH_PY = "/home/caimlas/bench-venv/bin/python"
LCB_DIR = "/home/caimlas/git/LiveCodeBench"
TS = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())

# name -> (file, lcb_model_id, out_dir_name, gpu, extra_server_args)
# gpu: 0=V100, 1=3060
MODELS = {
    # Tier 1: Qwen3.8-27B family (thinking-suspect, files on disk)
    "Qwen3.8-27B Q4_K_M":                     ("Qwen3.8-27B-Q4_K_M.gguf", "local/qwen38-27b-q4km", "Qwen3.8-27B Q4_K_M", 0, []),
    "Qwen3.8-27B Q4_K_M MTP":                 ("Qwen3.8-27B-Q4_K_M.gguf", "local/qwen38-27b-q4km-mtp", "Qwen3.8-27B Q4_K_M MTP", 0,
                                               ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]),
    "Qwen3.8-27B Heretic Q4_K_M":             ("Qwen3.8-27B-Heretic-Q4_K_M.gguf", "local/qwen38-27b-heretic-mtp", "Qwen3.8-27B Heretic Q4_K_M MTP", 0, []),
    "Qwen3.8-27B Uncensored Q4_K_M MTP":      ("Qwen3.8-27B-Uncensored-Q4_K_M.gguf", "local/qwen38-27b-uncens-mtp", "Qwen3.8-27B Uncensored Q4_K_M MTP", 0,
                                               ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]),
    "Qwen3.8-27B AEON Ultimate Uncensored Q4_K_M MTP": ("Qwen3.8-27B-AEON-Ultimate-Uncensored-Q4_K_M.gguf", "local/qwen38-27b-aeon-mtp", "Qwen3.8-27B AEON Ultimate Uncensored Q4_K_M MTP", 0,
                                               ["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"]),
    "Qwen3.8-27B UD-IQ3_S (GGUF, ~3.5bpw)":   ("Qwen3.8-27B-UD-IQ3_S.gguf", "local/qwen38-27b-ud-iq3s", "Qwen3.8-27B UD-IQ3_S", 0, []),
    "Qwen3.8-27B UD-Q4_K_S (GGUF, ~4.5bpw)":  ("Qwen3.8-27B-UD-Q4_K_S.gguf", "local/qwen38-27b-ud-q4ks", "Qwen3.8-27B UD-Q4_K_S", 0, []),
    # Tier 2/4: other thinking-suspect with file on disk
    "Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M": ("Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M.gguf", "local/qwen35-abliterated", "Qwen3.6-35B-A3B-Abliterated-Heretic-Q4_K_M", 0, []),
    "Carnice-V3 Q4_K_M":                      ("Carnice-V3-Q4_K_M.gguf", "local/carnice-v3", "Carnice-V3 Q4_K_M", 0, []),
    "Muse-Glimmer-30B-UD-Q4_K_XL":            ("Muse-Glimmer-30B-UD-Q4_K_XL.gguf", "local/muse-glimmer-30b", "Muse-Glimmer-30B-UD-Q4_K_XL", 0, []),
    # Tier 3: LFM (unknown cause, diagnose via empty-rate after rerun)
    "LFM2.5-8B-A1B base Q4_K_M":              ("LFM2.5-8B-A1B-Q4_K_M.gguf", "local/lfm-base", "LFM2.5-8B-A1B-Q4_K_M", 0, []),
    "LFM2.5-8B-A1B Q6_K":                     ("LFM2.5-8B-A1B-Q6_K.gguf", "local/lfm-q6", "LFM2.5-8B-A1B-Q6_K", 0, []),
    "LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M": ("LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M.gguf", "local/lfm-coder-v2", "LFM2.5-8B-A1B-Clean-RealWorld-v2-Q4_K_M", 0, []),
}

BASE_ARGS = ["--gpu-layers", "99", "--ctx-size", "32768", "--ubatch-size", "512",
             "--threads", "8", "--threads-batch", "8",
             "--cache-type-k", "q8_0", "--cache-type-v", "q8_0", "--jinja",
             "--parallel", "2", "--temp", "0.0", "-n", "4096", "--flash-attn", "on",
             "--batch-size", "2048", "--host", "127.0.0.1"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_progress():
    with open(PROGRESS_FILE) as f:
        return json.load(f)


def save_progress(p):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(p, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


def note(progress, name, benchmark, error):
    for m in progress["models"]:
        if m["name"] == name:
            m.setdefault("failures", []).append(
                {"benchmark": benchmark, "attempt": "rerun", "error": error[:500], "timestamp": TS})
            return


def audit_empty(out_dir):
    n = empty = 0
    for gf in glob.glob(os.path.join(out_dir, "Scenario.codegeneration_1_*.json")):
        if gf.endswith("_all.json") or gf.endswith("_eval.json"):
            continue
        for g in json.load(open(gf)):
            n += 1
            if not (g.get("output_list") or [""])[0].strip():
                empty += 1
    return empty, n


def run_one(name, cfg, progress):
    fname, lcb_model, out_name, gpu, extra = cfg
    out_dir = os.path.join(LCB_DIR, "output", out_name)
    if progress["models"] and next((m for m in progress["models"] if m["name"] == name), None) is None:
        log(f"  SKIP {name}: no progress entry")
        return
    if not os.path.exists(os.path.join(LLMS_DIR, fname)):
        log(f"  SKIP {name}: file missing")
        return

    # preserve broken artifacts (never clobber existing preservation with empty dir)
    if os.path.exists(out_dir) and any(os.scandir(out_dir)):
        broken = out_dir + ".empty-content-broken"
        if os.path.exists(broken):
            shutil.rmtree(broken)
        os.rename(out_dir, broken)
    elif os.path.exists(out_dir):
        os.rmdir(out_dir)

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    srv_log = open(os.path.join(LOGS, f"rerun_{re.sub(r'[^a-zA-Z0-9]+','_',name)[:40]}_server.log"), "w")
    proc = subprocess.Popen([BINARY, "-m", os.path.join(LLMS_DIR, fname)] + BASE_ARGS + extra +
                            ["--port", str(PORT)],
                            env=env, stdout=srv_log, stderr=subprocess.STDOUT, text=True)
    up = False
    for _ in range(240):
        time.sleep(2)
        if proc.poll() is not None:
            log(f"  FATAL {name}: server died: " + open(srv_log.name).read()[-250:])
            note(progress, name, "lcb rerun (server died)", open(srv_log.name).read()[-400:])
            save_progress(progress)
            srv_log.close()
            return
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            up = True
            break
        except Exception:
            pass
    if not up:
        proc.kill()
        log(f"  FATAL {name}: server timeout")
        note(progress, name, "lcb rerun (server timeout)", "health check timeout")
        save_progress(progress)
        srv_log.close()
        return

    # identity check
    try:
        props = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5).read().decode()
        if fname.replace(".gguf", "") not in props:
            log(f"  FATAL {name}: wrong model on port: {props[:150]}")
            note(progress, name, "lcb rerun (wrong server)", f"props mismatch: {props[:200]}")
            save_progress(progress)
            proc.kill()
            srv_log.close()
            return
        log(f"  identity verified: {name}")
    except Exception as e:
        log(f"  FATAL {name}: props check failed: {e}")
        note(progress, name, "lcb rerun (props check failed)", str(e))
        save_progress(progress)
        proc.kill()
        srv_log.close()
        return

    ok = False
    try:
        lcmd = [BENCH_PY, "-m", "lcb_runner.runner.main",
                "--model", lcb_model,
                "--scenario", "codegeneration", "--release_version", "release_latest",
                "--n", "1", "--temperature", "0.0", "--max_tokens", "4096",
                "--num_problems", "75", "--openai_timeout", "300", "--evaluate"]
        lenv = dict(os.environ)
        lenv.update({"OPENAI_KEY": "none",
                     "OPENAI_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
                     "HF_ALLOW_CODE_EVAL": "1",
                     "LCB_DISABLE_THINKING": "1"})
        llog = os.path.join(LOGS, f"rerun_{re.sub(r'[^a-zA-Z0-9]+','_',name)[:40]}_lcb.log")
        t0 = time.time()
        with open(llog, "w") as lf:
            result = subprocess.run(lcmd, stdout=lf, stderr=subprocess.STDOUT,
                                    cwd=LCB_DIR, env=lenv, timeout=4 * 3600)
        elapsed = time.time() - t0

        pass1 = None
        for root, dirs, files in os.walk(out_dir):
            for f in files:
                if f.endswith("_eval.json") and not f.endswith("_all.json"):
                    try:
                        data = json.load(open(os.path.join(root, f)))
                        if isinstance(data, list) and data and isinstance(data[0], dict):
                            pass1 = data[0].get("pass@1")
                    except Exception:
                        pass
        empty, n = audit_empty(out_dir)
        log(f"  RESULT {name}: pass@1={pass1} ({elapsed/60:.0f} min) empty={empty}/{n}")

        entry = next(m for m in progress["models"] if m["name"] == name)
        old = (entry.get("livecodebench") or {}).get("pass_at_1")
        if pass1 is None:
            note(progress, name, "lcb rerun (no score)",
                 f"rerun produced no score; entry unchanged (old {old}). exit={result.returncode}")
        else:
            entry["livecodebench"] = {"pass_at_1": pass1, "wall_time_s": round(elapsed, 1),
                                      "exit_code": result.returncode}
            note(progress, name, "livecodebench (invalidated run)",
                 f"Original LCB {old} recorded with {empty}-suspect empty-content rate; rerun with "
                 f"thinking suppression verified: {pass1} (empty {empty}/{n}). Broken artifacts: "
                 f"{out_name}.empty-content-broken")
            ok = True
    except subprocess.TimeoutExpired:
        note(progress, name, "lcb rerun (timeout)", "LCB 4h timeout")
    except Exception as e:
        note(progress, name, "lcb rerun (error)", str(e))
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:
            proc.kill()
        srv_log.close()
        save_progress(progress)
    return ok


def main():
    only = sys.argv[1:] if len(sys.argv) > 1 else None
    progress = load_progress()
    log(f"=== LCB remediation reruns ({len(only) if only else len(MODELS)} models) ===")
    for name, cfg in MODELS.items():
        if only and name not in only:
            continue
        log(f"--- {name} ---")
        run_one(name, cfg, progress)
    log("=== ALL RERUNS DONE ===")
    for name in (only or MODELS):
        e = next((m for m in progress["models"] if m["name"] == name), None)
        if e:
            l = (e.get("livecodebench") or {}).get("pass_at_1")
            print(f"  {name:<50} LCB={l}")


if __name__ == "__main__":
    main()
