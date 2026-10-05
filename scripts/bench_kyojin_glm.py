#!/usr/bin/env python3
"""Ready-to-run benchmark for the Kyojin (exllamav3 / ROCm gfx1151) GLM-5.3-Flash-EXL3 server.

WHAT IT DOES (only after yamz-labs/GLM-5.3-Flash-EXL3-Yamz finishes downloading):
  1. Starts Kyojin's tools/glm/serve.py on the Strix Halo iGPU (gfx1151, hip:1 -> cuda:0),
     port 8000, -c 131072, --num-draft 2.
  2. Waits for health (GET /v1/models).
  3. Runs one prompt-eval (prefill) probe over UNIQUE random text (no prefix-cache reuse),
     then two code-generation prompts at 256 output tokens each (temperature 0, greedy).
  4. Measures CLIENT-SIDE wall clock for every request and reports prompt tokens and output
     tokens separately. Also records the server's own `timings` block when it is present
     (Kyojin's serve.py does emit one: prompt_ms / predicted_ms / *_per_second).
  5. Writes JSON to /root/bench/results/kyojin-glm.json and stops the server.

MODELED ON: /root/bench/gen_probe.py (same probe prompt shape / measurement style).

USAGE (weights must be complete):
    /root/kyojin/.venv/bin/python /root/bench/bench_kyojin_glm.py
  or, since only the stdlib is needed:
    python3 /root/bench/bench_kyojin_glm.py

Useful flags: --model, --port, --ctx, --num-draft, --probe-words, --code-tokens,
              --reuse-server (bench an already-running server; do not start/stop one),
              --out, --server-log, --wait-load.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import shlex
import signal
import string
import subprocess
import sys
import time
import urllib.error
import urllib.request

REPO = "/root/kyojin"
DEFAULT_MODEL = "/root/models/glm-exl3-yamz"   # the in-progress HF download target
DEFAULT_MODEL_ID = "glm-5.3-exl3"              # serve.py default --model-id
DEFAULT_OUT = "/root/bench/results/kyojin-glm.json"
DEFAULT_SERVER_LOG = "/root/bench/kyojin-serve.log"
# borg has TWO GPUs and the repo assumes a Strix-Halo-only box:
#   torch device 0 = Radeon AI PRO R9700 (gfx1201), device 1 = Radeon 8060S (gfx1151, iGPU).
# The exllamav3_ext was built for gfx1151 only, so the iGPU must be cuda:0 => HIP_VISIBLE_DEVICES=1.
# env.sh's default LD_PRELOAD (/opt/rocm HSA 1.18) is too old for the 7.13 nightly torch and makes
# torch import die with "undefined symbol: hsa_ext_image_create_v2"; the matching HSA ships in
# _rocm_sdk_devel, and the env check passes with it.
DEFAULT_HIP_DEVICE = "1"
DEFAULT_HSA_LIB = ("/root/kyojin/.venv/lib/python3.12/site-packages/"
                   "_rocm_sdk_devel/lib/libhsa-runtime64.so.1")


# ----------------------------------------------------------------------------- helpers
def http_json(url: str, body: dict | None = None, timeout: float = 3600.0):
    """Return (parsed_json, wall_seconds). body=None => GET."""
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    wall = time.perf_counter() - t0
    return json.loads(raw), wall


def server_healthy(port: int) -> bool:
    try:
        http_json(f"http://127.0.0.1:{port}/v1/models", timeout=5.0)
        return True
    except Exception:
        return False


def check_weights(model_dir: str) -> tuple[bool, str]:
    """Best-effort completeness check: config + template + 12 shards present."""
    if not os.path.isdir(model_dir):
        return False, f"model dir does not exist: {model_dir}"
    have_cfg = os.path.isfile(os.path.join(model_dir, "config.json"))
    tmpl = os.path.isfile(os.path.join(model_dir, "chat_template.jinja"))
    shards = sorted(glob.glob(os.path.join(model_dir, "model-*-of-*.safetensors")))
    total = sum(os.path.getsize(p) for p in shards)
    ok = have_cfg and tmpl and len(shards) >= 12
    msg = (f"config={have_cfg} template={tmpl} shards={len(shards)} "
           f"shard_bytes={total/2**30:.1f} GiB")
    return ok, msg


# ----------------------------------------------------------------------------- prompts
def make_probe_prompt(words: int) -> str:
    """Unique random text every run => genuinely cold prefill (no prefix cache)."""
    toks = ["".join(random.choices(string.ascii_lowercase, k=random.randint(4, 9)))
            for _ in range(words)]
    return ("Summarize the following text in one sentence, then write a short Python function.\n\n"
            + " ".join(toks))


CODE_PROMPTS = [
    ("code-fib-memo",
     "Write a Python function `fib_memo(n)` that returns the n-th Fibonacci number using "
     "memoization, with a docstring and a short usage example. Output only the code."),
    ("code-lru-cache",
     "Implement a thread-safe LRU cache class in Python with `get` and `put` methods, "
     "capacity-bounded, using an OrderedDict internally. Output only the code."),
]


def run_chat(port: int, model_id: str, prompt: str, max_tokens: int, temperature: float) -> dict:
    body = {"model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False}
    d, wall = http_json(f"http://127.0.0.1:{port}/v1/chat/completions", body, timeout=7200.0)
    usage = d.get("usage", {}) or {}
    tm = d.get("timings", {}) or {}
    prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")
    text = ""
    try:
        text = d["choices"][0]["message"].get("content") or ""
    except Exception:
        pass
    row = {
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "wall_s": round(wall, 3),
        # client-side wall-clock rates (the numbers we trust)
        "wall_output_tps": round(output_tokens / wall, 2) if output_tokens and wall else None,
        "wall_total_tps": round((prompt_tokens + output_tokens) / wall, 2)
                          if (prompt_tokens is not None and output_tokens is not None and wall) else None,
        # server-reported split (present in Kyojin serve.py responses)
        "server_cache_n": tm.get("cache_n"),
        "server_prompt_ms": tm.get("prompt_ms"),
        "server_predicted_ms": tm.get("predicted_ms"),
        "server_prompt_per_second": tm.get("prompt_per_second"),
        "server_predicted_per_second": tm.get("predicted_per_second"),
    }
    row["text_preview"] = (text[:200] + ("..." if len(text) > 200 else ""))
    return row


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--ctx", "-c", type=int, default=131072)
    ap.add_argument("--num-draft", type=int, default=2)
    ap.add_argument("--probe-words", type=int, default=1200)
    ap.add_argument("--code-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--wait-load", type=float, default=2400.0,
                    help="seconds to wait for the server to load the model and answer health")
    ap.add_argument("--reuse-server", action="store_true",
                    help="do not start/stop a server; bench whatever is on --port")
    ap.add_argument("--keep-server", action="store_true", help="do not stop the server at the end")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--server-log", default=DEFAULT_SERVER_LOG)
    ap.add_argument("--hip-device", default=DEFAULT_HIP_DEVICE,
                    help="HIP_VISIBLE_DEVICES value; borg's gfx1151 Strix Halo is torch index 1")
    ap.add_argument("--hsa-lib", default=DEFAULT_HSA_LIB,
                    help="libhsa-runtime64.so.1 to LD_PRELOAD (must match the nightly torch wheel)")
    ap.add_argument("--skip-preflight", action="store_true",
                    help="skip the env.sh --check GPU preflight")
    ap.add_argument("--no-seed-dense-tune", action="store_true",
                    help="do not seed the dense-GEMM tune cache from tools/lanes/assets/tune_seed.txt")
    args = ap.parse_args()

    results: dict = {
        "meta": {
            "host": os.uname().nodename,
            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "model": args.model,
            "model_id": args.model_id,
            "port": args.port,
            "ctx": args.ctx,
            "num_draft": args.num_draft,
            "probe_words": args.probe_words,
            "code_tokens": args.code_tokens,
            "temperature": args.temperature,
            "python": sys.version.split()[0],
        },
        "runs": [],
        "errors": [],
    }

    proc = None
    logf = None
    try:
        reuse = args.reuse_server or server_healthy(args.port)
        if reuse:
            print(f"[bench] reusing running server on port {args.port}", flush=True)
        else:
            ok, msg = check_weights(args.model)
            print(f"[bench] weights check: {msg}", flush=True)
            if not ok:
                results["errors"].append(f"weights incomplete: {msg}")
                print(f"[bench] ABORT — {msg}", file=sys.stderr, flush=True)
                _write(args.out, results)
                return 2

            # Runtime env: source the repo's Strix Halo env (sets LD_PRELOAD libhsa + PYTHONPATH).
            # Trap 5 in env.sh: the build-time ROCm SDK paths must NOT be exported at runtime,
            # so scrub them; instead pin the iGPU and the matching HSA lib explicitly.
            env = dict(os.environ)
            for k in ("EXL3_ROCM_SDK", "EXL3_ROCM_DEV_INCLUDE", "ROCM_HOME", "ROCM_PATH",
                      "HIP_PATH", "HIPCXX", "HIP_DEVICE_LIB_PATH", "HIPCC_COMPILE_FLAGS_APPEND",
                      "PYTHONPATH", "HSA_OVERRIDE_GFX_VERSION"):
                env.pop(k, None)
            env["HIP_VISIBLE_DEVICES"] = args.hip_device      # gfx1151 iGPU -> cuda:0
            if args.hsa_lib and os.path.exists(args.hsa_lib):
                env["EXL3_HSA_LIB"] = args.hsa_lib            # overrides env.sh's /opt/rocm pick
            env["PYTHONNOUSERSITE"] = "1"
            results["meta"]["hip_visible_devices"] = args.hip_device
            results["meta"]["hsa_lib"] = env.get("EXL3_HSA_LIB")

            # Seed the dense-GEMM tune cache (the repo's serving lane does this) so first prefills
            # don't pay ~40 s/row-class rocBLAS screening. If the seed's tag doesn't match the
            # running lib the engine just re-tunes, so this is safe either way.
            if not args.no_seed_dense_tune:
                tune = os.path.expanduser("~/.cache/exllamav3/dense_gemm_tune.txt")
                seed = os.path.join(REPO, "tools/lanes/assets/tune_seed.txt")
                if not os.path.exists(tune) and os.path.exists(seed):
                    os.makedirs(os.path.dirname(tune), exist_ok=True)
                    import shutil as _sh
                    _sh.copyfile(seed, tune)
                if os.path.exists(tune):
                    env["EXL3_DENSE_GEMM_TUNE_FILE"] = tune
            results["meta"]["dense_gemm_tune_file"] = env.get("EXL3_DENSE_GEMM_TUNE_FILE")

            # Preflight: the repo's own env.sh --check (torch HIP + gfx1151 matmul + exllamav3_ext).
            if not args.skip_preflight:
                chk = subprocess.run(
                    ["bash", "-lc",
                     f"cd {shlex.quote(REPO)} && bash tools/strix_halo/env.sh --check"],
                    env=env, cwd=REPO, capture_output=True, text=True, timeout=900)
                out = (chk.stdout or "") + (chk.stderr or "")
                results["meta"]["env_check"] = out.strip()
                print("[bench] env.sh --check:\n" + out.strip(), flush=True)
                if chk.returncode != 0:
                    results["errors"].append(f"env.sh --check failed rc={chk.returncode}")
                    print("[bench] ABORT — GPU preflight failed", file=sys.stderr, flush=True)
                    _write(args.out, results)
                    return 4

            launch = (
                f"cd {shlex.quote(REPO)} && source tools/strix_halo/env.sh && "
                f"exec {shlex.quote(REPO)}/.venv/bin/python tools/glm/serve.py "
                f"--model {shlex.quote(args.model)} --model-id {shlex.quote(args.model_id)} "
                f"--host 127.0.0.1 --port {args.port} -c {args.ctx} --num-draft {args.num_draft}"
            )
            logf = open(args.server_log, "w")
            print(f"[bench] starting server: {launch}", flush=True)
            proc = subprocess.Popen(["bash", "-lc", launch], env=env, cwd=REPO,
                                    stdout=logf, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            results["meta"]["server_pid"] = proc.pid
            results["meta"]["server_log"] = args.server_log

            # Wait for health.
            t0 = time.perf_counter()
            ready = False
            while time.perf_counter() - t0 < args.wait_load:
                if proc.poll() is not None:
                    results["errors"].append(f"server exited early rc={proc.returncode}")
                    print(f"[bench] server exited early rc={proc.returncode}; see {args.server_log}",
                          file=sys.stderr, flush=True)
                    break
                if server_healthy(args.port):
                    ready = True
                    break
                time.sleep(5)
            results["meta"]["load_s"] = round(time.perf_counter() - t0, 1)
            if not ready:
                if not results["errors"]:
                    results["errors"].append("server did not become healthy in time")
                print("[bench] server never became healthy", file=sys.stderr, flush=True)
                _write(args.out, results)
                return 3
            print(f"[bench] server healthy after {results['meta']['load_s']:.0f}s", flush=True)

        # ---- 1) prompt-eval (prefill) probe ------------------------------------
        probe = make_probe_prompt(args.probe_words)
        print("[bench] running prompt-eval probe ...", flush=True)
        row = run_chat(args.port, args.model_id, probe, max_tokens=32, temperature=0.0)
        row["label"] = "prompt-eval-probe"
        row["kind"] = "prompt_eval"
        results["runs"].append(row)
        print(f"[bench]   probe: prompt_tokens={row['prompt_tokens']} "
              f"server_prompt_tps={row['server_prompt_per_second']} wall_s={row['wall_s']}", flush=True)

        # ---- 2) two code-generation prompts, 256 output tokens each ------------
        for label, prompt in CODE_PROMPTS:
            print(f"[bench] running {label} ...", flush=True)
            row = run_chat(args.port, args.model_id, prompt, max_tokens=args.code_tokens,
                           temperature=args.temperature)
            row["label"] = label
            row["kind"] = "code_gen"
            results["runs"].append(row)
            print(f"[bench]   {label}: prompt_tokens={row['prompt_tokens']} "
                  f"output_tokens={row['output_tokens']} wall_output_tps={row['wall_output_tps']} "
                  f"server_decode_tps={row['server_predicted_per_second']}", flush=True)

        # ---- summary -----------------------------------------------------------
        code = [r for r in results["runs"] if r["kind"] == "code_gen"]
        wall_code = [r["wall_output_tps"] for r in code if r["wall_output_tps"]]
        srv_code = [r["server_predicted_per_second"] for r in code
                    if r["server_predicted_per_second"]]
        probe_rows = [r for r in results["runs"] if r["kind"] == "prompt_eval"]
        results["summary"] = {
            "probe_prompt_tokens": probe_rows[0]["prompt_tokens"] if probe_rows else None,
            "probe_server_prefill_tps": probe_rows[0]["server_prompt_per_second"] if probe_rows else None,
            "code_mean_wall_output_tps": round(sum(wall_code) / len(wall_code), 2) if wall_code else None,
            "code_mean_server_decode_tps": round(sum(srv_code) / len(srv_code), 2) if srv_code else None,
            "code_tokens_each": args.code_tokens,
        }
        results["meta"]["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write(args.out, results)
        print(f"[bench] wrote {args.out}", flush=True)
        print("[bench] summary: " + json.dumps(results["summary"]), flush=True)
        return 0

    finally:
        if proc is not None and proc.poll() is None and not args.keep_server:
            print("[bench] stopping server ...", flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                try:
                    proc.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if logf is not None:
            logf.close()


def _write(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
