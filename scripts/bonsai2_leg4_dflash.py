#!/usr/bin/env python3
"""Leg 4: DFlash-2 ft5 drafter probe on V100 (prism HEAD 73bfe6c).

Serve PTQ1_0 target + DFlash2-ft5-Q8_0 drafter via --spec-type draft-dflash
(also try draft-dspark, and a renamed copy containing 'dspark-dflash' if the
loader rejects the name). Measure load, draft acceptance, decode t/s vs no-draft.
GPU lock held via fcntl on /tmp/gpu0.lock for the whole phase (prod stop->start).
"""
import fcntl, json, os, subprocess, sys, time, urllib.request

BIN = "/tmp/prism-head/build/bin"
MODELS = "/var/tmp/llms/bench-0918/bonsai2"
DRAFT_DIR = "/home/files/llms/bonsai2-drafter"
TARGET = f"{MODELS}/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
DRAFT = f"{DRAFT_DIR}/Ternary-Bonsai-2-27B-DFlash2-ft5-Q8_0.gguf"
RENAME_DIR = "/var/tmp/llms/bench-0918/bonsai2"  # put renamed copy next to target? keep in draft dir
RENAMED = f"{DRAFT_DIR}/dspark-dflash-Ternary-Bonsai-2-27B-DFlash2-ft5-Q8_0.gguf"
PORT = 18096
OUT = {}

PROMPT = ("Write a Python module with four functions: (1) merge_sorted_lists(a, b) merging two "
          "sorted lists into one sorted list in linear time; (2) quicksort(arr) with a random "
          "pivot and a three-way partition, handling duplicates efficiently; (3) binary_search(arr, x) "
          "returning the leftmost index of x or -1; (4) run_tests() with at least eight assert-based "
          "test cases covering edge cases like empty lists, single elements, duplicates, and "
          "reverse-sorted input. Include a docstring for each function explaining its complexity, "
          "and a __main__ guard that runs the tests.")

def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)

def stop_prod():
    sh("sudo -n systemctl stop caimlas-minicpm-v100")
    out = None
    for _ in range(30):
            time.sleep(2)
            out = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0").stdout.strip()
            if not out:
                print("[prod] V100 clear")
                return
    print(f"[prod] WARN: V100 still busy: {out}")

def start_prod():
    sh("sudo -n systemctl start caimlas-minicpm-v100")
    print("[prod] restart issued")

def wait_health(port, tries=90):
    for _ in range(tries):
        try:
            with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
                if json.load(r).get("status") == "ok":
                    return True
        except Exception:
            time.sleep(5)
    return False

def chat(port, max_tokens=512):
    body = json.dumps({"messages": [{"role": "user", "content": PROMPT}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    return d, time.time() - t0

def serve_once(tag, draft_path, spec_type, log_path):
    """Start server with given drafter/spec-type; return dict of results or None."""
    srv = (f"{BIN}/llama-server -m {TARGET} --port {PORT} -ngl 99 -ngld 999 -c 32768 "
           f"-fa 1 -ctk q8_0 -ctv q8_0 -ub 1024 --jinja -sm none --main-gpu 0 "
           f"-md {draft_path} --spec-type {spec_type} "
           f"--spec-draft-type-k q8_0 --spec-draft-type-v q8_0") if draft_path else (
           f"{BIN}/llama-server -m {TARGET} --port {PORT} -ngl 99 -c 32768 "
           f"-fa 1 -ctk q8_0 -ctv q8_0 -ub 1024 --jinja -sm none --main-gpu 0")
    env = dict(os.environ); env["CUDA_VISIBLE_DEVICES"] = "0"
    print(f"[{tag}] starting: {spec_type or 'no-draft'}")
    logf = open(log_path, "w")
    p = subprocess.Popen(srv.split(), stdout=logf, stderr=subprocess.STDOUT, env=env)
    try:
        if not wait_health(PORT):
            logf.flush()
            print(f"[{tag}] server FAILED to come up; log tail:")
            print(open(log_path).read()[-3000:])
            return {"loaded": False}
        logf.flush()
        log = open(log_path).read()
        # drafter type evidence from log
        ev = [l for l in log.splitlines() if any(k in l.lower() for k in
              ("dflash", "dspark", "speculative", "draft"))][-25:]
        d, wall = chat(PORT)
        res = {
            "loaded": True,
            "spec_type_arg": spec_type,
            "log_evidence": "\n".join(evs for evs in ev) if ev else "(no spec lines)",
            "timings": d.get("timings"),
            "usage": d.get("usage"),
            "wall_s": round(wall, 2),
            "content_head": (d["choices"][0]["message"].get("content") or "")[:150],
        }
        t = d.get("timings") or {}
        if t.get("draft_n"):
            res["acceptance"] = round(t["draft_n_accepted"] / t["draft_n"], 4)
            res["draft_n"] = t["draft_n"]; res["draft_n_accepted"] = t["draft_n_accepted"]
        print(json.dumps({k: v for k, v in res.items() if k != "log_evidence"}, indent=1))
        return res
    finally:
        p.terminate()
        try: p.wait(timeout=15)
        except Exception: p.kill()
        time.sleep(5)

def main():
    lock = open("/tmp/gpu0.lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    print("[lock] gpu0 acquired")
    try:
        stop_prod()
        # A: baseline no-draft on identical server config
        OUT["baseline_nodraft"] = serve_once("base", None, None, "/tmp/leg4_base.log")
        # B: spec-type draft-dflash, original name
        OUT["dflash_named"] = serve_once("dflash", DRAFT, "draft-dflash", "/tmp/leg4_dflash.log")
        # C: spec-type draft-dspark, original name (loader should warn + auto-pick)
        if OUT["dflash_named"].get("loaded"):
            OUT["dspark_named"] = serve_once("dspark", DRAFT, "draft-dspark", "/tmp/leg4_dspark.log")
        # D: renamed copy containing 'dspark-dflash' if B failed to load
        if not OUT["dflash_named"].get("loaded"):
            if not os.path.exists(RENAMED):
                sh(f"cp {DRAFT} {RENAMED}")
            OUT["renamed_dsparkdflash"] = serve_once("renamed", RENAMED, "draft-dflash", "/tmp/leg4_renamed.log")
    finally:
        start_prod()
        fcntl.flock(lock, fcntl.LOCK_UN)
        print("[lock] gpu0 released")
        with open("/tmp/leg4_results.json", "w") as f:
            json.dump(OUT, f, indent=1)
        print("[done] -> /tmp/leg4_results.json")

if __name__ == "__main__":
    main()
