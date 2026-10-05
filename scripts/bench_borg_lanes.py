#!/usr/bin/env python3
"""local-llm-benchmarks lanes for the borg host (bhodgens/local-llm-benchmarks).

One registry covers every engine we run on borg, so results are like-for-like:
same prompts, same sampling, same 2-run-per-prompt protocol, one row per model
with the engine/server named.

Columns recorded per model
  engine            server/engine that served the model (its own row)
  prefill_cold_tps  prompt processing, COLD (no prefix cache), short prompt
  prefill_long_tps  prompt processing, COLD, ~2.5K-token prompt (the fair one)
  decode_sampled    the harness default: temperature 0.3
  spec_sampled      did the engine run speculative decode at temp 0.3?
  decode_greedy     production path: temperature 0 (this is where drafters fire)
  spec_greedy / accept_greedy   drafter state and acceptance on the greedy path
  boot_s            model load time

Timing sources differ per engine and are normalised here:
  luce_server     usage.timings.decode_tokens_per_sec / prefill_ms / accept_rate / spec_decode_ran
  llama.cpp-based timings.predicted_per_second / prompt_per_second (no spec/accept fields)
  Kyojin/Strata   timings.predicted_per_second / prompt_per_second

Usage:
  bench_borg_lanes.py --models all
  bench_borg_lanes.py --models qwen38-27b,laguna-xs21
  bench_borg_lanes.py --table            # just print the table from the JSON
"""
import argparse, json, os, signal, subprocess, sys, time, urllib.request

RES = "/root/bench/results"
OUT = "/root/bench/bench_results_borg.json"

# model -> (launcher, port, gpu, engine label, api model id, launcher args)
MODELS = {
    "qwen38-27b":        ("/root/bench/serve_qwen3827b.sh",     8901, "R9700",            "luce_server (Lucebox HIP)",        "luce", ["f16", "16"]),
    "qwen38-27b-vision": ("/root/bench/serve_qwen38vision.sh",  8902, "R9700",            "luce_server (Lucebox HIP)",        "luce", ["f16", "16"]),
    "laguna-xs21":       ("/root/bench/serve_laguna.sh",        8902, "R9700",            "luce_server (Lucebox HIP)",        "luce", []),
    "ds4-flash":         ("/root/bench/serve_ds4flash.sh",      8902, "R9700+Strix",      "luce_server (Lucebox HIP)",        "luce", []),
    "ds41-flash":        ("/root/bench/serve_ds41flash.sh",     8902, "R9700+Strix+SSD",  "luce_server (Lucebox HIP)",        "luce", []),
    "glm53-flash":       ("/root/bench/serve_glm53flash.sh",    8902, "Strix",            "paoai-strix-engine (llama.cpp+Vulkan)", "glm", []),
    "flashnext":         ("/root/strata/run-iq3_s.sh",          8080, "R9700+RAM",        "Strata (HIP)",                     "qwen3.8-flash-next-iq3_s", []),
    # api_model "-" = ask the server: GET /v1/models and use the id it reports
    "kolibri-1":         ("/root/bench/serve_kolibri.sh",       8902, "R9700+Strix",      "llama.cpp-kolibri (patched HIP)",  "-", []),
}

# Per-model request extras. Flash Next reasons by default and that costs most of
# its throughput: with reasoning_effort=off it measures ~93 t/s greedy vs ~68 with
# reasoning on, so the lane must send it to be comparable to the other rows.
EXTRA_BODY = {
    "flashnext": {"reasoning_effort": "off"},
}

PROMPT_SHORT = open("/root/bench/prompt_eval_text.txt").read().strip()
DECODE_PROMPTS = [
    "Write a Python function that implements binary search on a sorted list. Include docstring, type hints, and handle edge cases.",
    "Explain how merge sort works step by step. Include pseudocode and analyze the time complexity.",
]


def unique_prompt(approx_tokens, tag):
    """Unique random text so a probe cannot be served by the engine's prefix cache.
    Tags make the intent visible in the engine log if anyone audits it."""
    import random, string
    words = ["".join(random.choices(string.ascii_lowercase, k=random.randint(4, 9)))
             for _ in range(max(8, approx_tokens // 4))]
    return (f"[probe {tag}] Summarize the following text in one sentence.\n\n" + " ".join(words))


def wait_health(port, timeout_s=900):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        for path in ("/health", "/v1/models"):
            try:
                r = urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3).read()
                if b"ok" in r or b'"object"' in r or b'"data"' in r:
                    return True, round(time.time() - t0, 1)
            except Exception:
                pass
        time.sleep(2)
    return False, round(time.time() - t0, 1)


def stop_servers():
    # Match the server BINARY PATHS only. A bare model name is unsafe here: this
    # script's own cmdline contains "kolibri" / "luce_server" (via --models ...),
    # and a matching pattern makes the lane kill its own session.
    for pat in ("[b]uild-hip/luce_server", "[b]uild-vk/bin/llama-server",
                "[e]ngine/strata", "[s]erve/server.py", "[t]ools/glm/serve.py",
                "[l]lama.cpp-kolibri/build-hip"):
        subprocess.run(f"ps -eo pid,cmd | grep -E '{pat}' | awk '{{print $1}}' | xargs -r kill",
                       shell=True)
    time.sleep(5)


def call(port, model, prompt, n_predict, temp, extra=None):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": n_predict, "temperature": temp, "stream": False}
    if temp:                       # only send sampler params when sampling (greedy = temp 0, no sampler)
        body.update({"top_p": 0.9, "top_k": 40})
    if extra:
        body.update(extra)
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=1800).read())
    wall = time.time() - t0
    u = d.get("usage", {})
    t = u.get("timings") or d.get("timings", {})
    pn = t.get("prompt_n") or u.get("prompt_tokens", 0)
    pms = t.get("prompt_ms") or t.get("prefill_ms", 0)
    dn = t.get("predicted_n") or u.get("completion_tokens", 0)
    dms = t.get("predicted_ms") or t.get("decode_ms", 0)
    return {
        "prompt_tokens": pn,
        "prompt_tps": round(pn / pms * 1000, 2) if pms else round(t.get("prompt_per_second", 0), 2),
        "decode_tokens": dn,
        "decode_tps": round(dn / dms * 1000, 2) if dms else round(t.get("predicted_per_second", 0), 2) or round(dn / wall, 2),
        "accept_rate": t.get("accept_rate", u.get("accept_rate")),
        "spec_decode_ran": t.get("spec_decode_ran", u.get("spec_decode_ran")),
        "wall_s": round(wall, 2),
    }


def path_stats(port, model, n_predict, temp, extra=None):
    runs = [call(port, model, p, n_predict, temp, extra) for p in DECODE_PROMPTS for _ in range(2)]
    return {
        "decode_tps_avg": round(sum(r["decode_tps"] for r in runs) / len(runs), 2),
        "accept_rate_avg": round(sum((r["accept_rate"] or 0) for r in runs) / len(runs), 3),
        "spec_decode_ran": any(bool(r["spec_decode_ran"]) for r in runs),
        "wall_s_avg": round(sum(r["wall_s"] for r in runs) / len(runs), 2),
    }


def run_model(key, n_predict):
    script, port, gpu, engine, api_model, extra = MODELS[key]
    e = {"model": key, "engine": engine, "gpu": gpu, "port": port,
         "protocol": "local-llm-benchmarks: prompt-eval cold(short+long) + 2 decode prompts x2 runs",
         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    log = open(f"{RES}/lane-{key}.log", "w")
    stop_servers()
    proc = subprocess.Popen(["bash", script] + extra, stdout=log, stderr=subprocess.STDOUT,
                            preexec_fn=os.setsid)
    try:
        ok, boot = wait_health(port)
        if not ok:
            e["error"] = f"not healthy after {boot}s (see lane-{key}.log)"
            print(json.dumps(e)); return e
        e["boot_s"] = boot
        if api_model == "-":                      # ask the server what it calls itself
            try:
                m = json.loads(urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/v1/models", timeout=5).read())
                api_model = m["data"][0]["id"]
                e["model_id_reported"] = api_model
            except Exception as ex:
                e["error"] = f"could not read /v1/models: {ex}"
                print(json.dumps(e)); return e
        xb = EXTRA_BODY.get(key, {})
        # Every probe uses unique text: with shared text the engine's prefix cache
        # answers it and "cold prefill" silently becomes a cache hit (seen once as
        # 12879 t/s). The first request after load also pays kernel warmup, so it
        # is measured and discarded.
        first = call(port, api_model, unique_prompt(500, "warmup"), 8, 0.3, xb)
        e["prefill_first_request_tps"] = first["prompt_tps"]
        cold_text = unique_prompt(500, "cold")
        ps1 = call(port, api_model, cold_text, 8, 0.3, xb)
        ps2 = call(port, api_model, cold_text, 8, 0.3, xb)      # same text => cache hit
        pl = call(port, api_model, unique_prompt(2500, "long"), 8, 0.3, xb)
        e["prefill_cold_tps"] = ps1["prompt_tps"]
        e["prefill_cold_tokens"] = ps1["prompt_tokens"]
        e["prefill_warm_tps"] = ps2["prompt_tps"]
        e["prefill_long_tps"] = pl["prompt_tps"]
        e["prefill_long_tokens"] = pl["prompt_tokens"]
        e["decode_sampled"] = path_stats(port, api_model, n_predict, 0.3, xb)
        e["decode_greedy"] = path_stats(port, api_model, n_predict, 0.0, xb)
        if xb: e["request_extras"] = xb
        print(f"{key:20s} engine={engine[:26]:26s} prefill 1st/2nd/2.5K "
              f"{e['prefill_first_request_tps']}/{e['prefill_cold_tps']}/{e['prefill_long_tps']} "
              f"sampled {e['decode_sampled']['decode_tps_avg']} (spec={e['decode_sampled']['spec_decode_ran']}) "
              f"greedy {e['decode_greedy']['decode_tps_avg']} (spec={e['decode_greedy']['spec_decode_ran']}, "
              f"acc={e['decode_greedy']['accept_rate_avg']})")
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM); proc.wait(timeout=20)
        except Exception:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception: pass
        stop_servers()
    try:
        data = json.load(open(OUT))
    except Exception:
        data = {}
    data[key] = e
    with open(OUT, "w") as f:
        json.dump(data, f, indent=2)
    return e


def print_table(path=OUT):
    d = json.load(open(path))
    hdr = (f"{'model':20s} {'engine/server':34s} {'pref_cold':>9s} {'pref_2.5K':>9s} "
           f"{'dec_samp':>8s} {'spec':>5s} {'dec_greedy':>10s} {'spec':>5s} {'accept':>7s} {'boot':>5s}")
    print(hdr); print("-" * len(hdr))
    for k, v in d.items():
        s, g = v.get("decode_sampled", {}), v.get("decode_greedy", {})
        print(f"{k:20s} {v.get('engine','')[:34]:34s} "
              f"{str(v.get('prefill_cold_tps','-')):>9s} {str(v.get('prefill_long_tps','-')):>9s} "
              f"{str(s.get('decode_tps_avg','-')):>8s} {str(s.get('spec_decode_ran','-')):>5s} "
              f"{str(g.get('decode_tps_avg','-')):>10s} {str(g.get('spec_decode_ran','-')):>5s} "
              f"{str(g.get('accept_rate_avg','-')):>7s} {str(v.get('boot_s','-')):>5s}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--table", action="store_true")
    args = ap.parse_args()
    if args.table:
        print_table(); return
    keys = sorted(MODELS) if args.models == "all" else [m.strip() for m in args.models.split(",")]
    for k in keys:
        if k not in MODELS:
            print(f"unknown model {k}", file=sys.stderr); continue
        run_model(k, args.n)
    print(); print_table()


if __name__ == "__main__":
    main()
