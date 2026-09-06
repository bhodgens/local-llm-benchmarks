#!/usr/bin/env python3
"""
Stage-0 benchmark gate using BenchKit: sanity slice + stability check.

For each model: start llama-server (same config as main benchmarks) ->
run BenchKit sanity:25 -> parse score -> optionally run a second slice with
choice-order perturbation for variance signal. Writes results into
progress.json under 'benchkit' key for the report.

Designed to be called from orchestrators before committing to LCB/tau2 hours,
or standalone for a quick vetting pass.

Usage:
    /home/caimlas/bench-venv/bin/python scripts/run_benchkit_gate.py \
        [--model-file FILE.gguf --name "Name" --gpu 0 --mtp]
    # or edit MODELS below
"""
import subprocess, json, time, os, sys, urllib.request, re
from datetime import datetime, timezone

BINARY = "/home/caimlas/git/llama.cpp/build/bin/llama-server"
BINARY_PRISMML = "/home/caimlas/git/llama.cpp-prismml/build/bin/llama-server"
LLMS_DIR = "/home/files/llms"
PORT = 18052
SCRATCH = "/tmp/coding-bench"
LOGS = os.path.join(SCRATCH, "logs")
PROGRESS_FILE = os.path.join(SCRATCH, "progress.json")
BENCHKIT_DIR = "/home/caimlas/git/BenchKit"
RESULTS_ROOT = os.path.join(BENCHKIT_DIR, "results")

MODELS = [
    # edit as needed; the CLI path (--model-file) appends/overrides
]

DEFAULT_ARGS = ["--gpu-layers", "99", "--ctx-size", "65536", "--ubatch-size", "512",
                "--threads", "8", "--threads-batch", "8",
                "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"start_time": datetime.now(timezone.utc).isoformat(), "models": []}

def save_progress(p):
    with open(PROGRESS_FILE, "w") as f:
        json.dump(p, f, indent=2)

def start_server(model):
    binary = model.get("binary", BINARY)
    path = os.path.join(LLMS_DIR, model["file"])
    cmd = [binary, "--model", path, "--flash-attn", "on",
           "--batch-size", "2048", "--host", "0.0.0.0", "--port", str(PORT),
           "--parallel", "1", "--temp", "0.0", "-n", "4096"]
    cmd.extend(model.get("args", DEFAULT_ARGS))
    if model.get("mtp"):
        cmd.extend(["--spec-type", "draft-mtp", "--spec-draft-n-max", "3"])

    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(model.get("gpu", 0))
    safe = model["name"].replace(" ", "_")
    logf = open(os.path.join(LOGS, f"{safe}_benchkit_server.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    for i in range(180):
        time.sleep(2)
        try:
            r = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3)
            if json.loads(r.read()).get("status") == "ok":
                return proc, logf, None
        except:
            pass
        if proc.poll() is not None:
            logf.close()
            with open(os.path.join(LOGS, f"{safe}_benchkit_server.log")) as f:
                err = f.read()[-400:]
            return None, None, f"Server died: {err}"
    proc.kill()
    return None, None, "Timeout"

def kill_server(proc, logf):
    if proc:
        proc.terminate()
        try: proc.wait(timeout=10)
        except: proc.kill()
    if logf: logf.close()
    time.sleep(3)

def run_benchkit(benchmarks, model_served_name):
    """Run BenchKit headless. Returns (score_pct, passed, total, wall_s, results_dir)."""
    env = dict(os.environ)
    env["BENCHKIT_PROVIDER"] = "openai"
    env["BENCHKIT_HOST"] = f"http://127.0.0.1:{PORT}/v1"
    cmd = ["uv", "run", "benchkit", "--headless",
           "--models", model_served_name,
           "--benchmarks", benchmarks]
    start = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200,
                       cwd=BENCHKIT_DIR, env=env)
    wall = time.time() - start
    out = r.stdout + r.stderr

    # Parse summary line: "✓ sanity · MODEL · Direct  100.0%  PASS 5  FAIL 0 ..."
    score = passed = total = None
    m = re.search(r'(\d+(?:\.\d+)?)%\s+PASS (\d+)\s+FAIL \d+\s+ERR \d+\s+LOOP \d+\s+(\d+) tasks', out)
    if m:
        score = float(m.group(1))
        passed = int(m.group(2))
        total = int(m.group(3))

    # find latest results dir
    results_dir = None
    if os.path.isdir(RESULTS_ROOT):
        dirs = sorted(d for d in os.listdir(RESULTS_ROOT) if os.path.isdir(os.path.join(RESULTS_ROOT, d)))
        if dirs:
            results_dir = os.path.join(RESULTS_ROOT, dirs[-1])

    return score, passed, total, wall, results_dir, out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file")
    ap.add_argument("--name")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--mtp", action="store_true")
    ap.add_argument("--benchmarks", default="sanity:25")
    ap.add_argument("--stability", action="store_true",
                    help="run a second slice with choice-order perturbation")
    args = ap.parse_args()

    models = list(MODELS)
    if args.model_file:
        models.append({"name": args.name or os.path.basename(args.model_file),
                       "file": args.model_file, "gpu": args.gpu, "mtp": args.mtp})

    if not models:
        print("No models specified (edit MODELS or pass --model-file)")
        sys.exit(1)

    os.makedirs(LOGS, exist_ok=True)
    progress = load_progress()

    for model in models:
        name = model["name"]
        print(f"\n=== BenchKit gate: {name} ===")
        proc, logf, err = start_server(model)
        if err:
            print(f"  FATAL: {err[:200]}")
            continue

        # llama-server reports the full model path as the served model id
        served = os.path.join(LLMS_DIR, model["file"])
        try:
            score, passed, total, wall, rdir, out = run_benchkit(args.benchmarks, served)
            print(f"  {args.benchmarks}: {score}% ({passed}/{total}) in {wall/60:.0f} min")

            mr = None
            for m in progress["models"]:
                if m["name"] == name:
                    mr = m
                    break
            if mr is None:
                mr = {"name": name, "file": model["file"], "gpu": f"GPU{model.get('gpu',0)}"}
                progress["models"].append(mr)
            mr.setdefault("benchkit", {})[args.benchmarks.split(":")[0]] = {
                "score_pct": score, "passed": passed, "total": total,
                "wall_time_s": round(wall, 1), "results_dir": rdir}
            save_progress(progress)

            if args.stability:
                s2, p2, t2, w2, rd2, _ = run_benchkit(
                    args.benchmarks + " --perturbation choice-order", served)
                print(f"  stability (choice-order): {s2}% ({p2}/{t2})")
                mr["benchkit"]["stability_choice_order"] = {
                    "score_pct": s2, "passed": p2, "total": t2,
                    "wall_time_s": round(w2, 1), "results_dir": rd2}
                save_progress(progress)
        finally:
            kill_server(proc, logf)

    print("\nBENCHKIT GATE COMPLETE")

if __name__ == "__main__":
    main()
