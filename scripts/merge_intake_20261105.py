#!/usr/bin/env python3
"""Merge intake_20261105 into progress.json + bench_results.json, regenerate report,
verify cells, and (separately) the caller commits. Run from llm-benchmarks root."""
import json, os, subprocess, sys

ROOT = "/home/caimlas/llm-benchmarks"
PROG = "/tmp/coding-bench/progress.json"
BRES = os.path.join(ROOT, "bench_results.json")
INTAKE = os.path.join(ROOT, "results/intake_20261105-061928.json")

today = "2026-11-05"

# ---- build progress rows ----
def row(name, fname, gpu_label, gpu_field, decode_tps, prefill_tps, sanity, lcb, he, tau2, engine="llama.cpp"):
    r = {
        "name": name, "file": fname, "status": "humaneval_done",
        "gpu": gpu_field, "category": "",
        "decode_tps": decode_tps, "prompt_tps": prefill_tps,
        "engine": engine, "template": "stock", "start_time": f"{today}T00:00:00",
        "benchkit": {"sanity": {"score_pct": sanity[0], "passed": sanity[1], "total": sanity[2]}} if sanity else {},
        "livecodebench": {"pass_at_1": lcb, "wall_time_s": None, "exit_code": 0},
        "tau2": {"reward": tau2, "wall_time_s": None, "exit_code": 0},
        "humaneval": {"pass_at_1": he, "wall_time_s": None, "exit_code": 0},
    }
    return r

# Sanity error for Qwopus in-lane was an infra artifact; use the clean remeasured 92.0
models_new = [
    row("LFM3-1.7B Q4_K_M (3060)",
        "LFM3-1.7B-Q4_K_M.gguf", "3060", "3060",
        176.99, 8391.07, (80.0, 20, 25), 0.2, 0.75, 0.0),
    row("LFM3-1.7B-Uncensored AD-Q6_K (3060)",
        "LFM3-1.7B-Uncensored-AD-Q6_K.gguf", "3060", "3060",
        163.69, 8081.47, (76.0, 19, 25), 0.2533333333, 0.85, 0.0667),
    row("Coding-Monkey-Gemma gemma-coder Q4_K_M (V100)",
        "gemma-coder-Q4_K_M.gguf", "V100", "GPU0",
        62.16, 1701.03, (92.0, 23, 25), 0.6133333333, 0.7333, 0.4),
    row("Qwopus3.8-27B-Coder-Flash Q4_K_M (V100)",
        "Qwopus3.8-27B-Coder-Flash-Q4_K_M.gguf", "V100", "GPU0",
        34.03, 792.94, (92.0, 23, 25), 0.8533333333, 1.0, 0.2),
]

# attach cross-GPU probe notes
models_new[0]["v100_probe"] = {"decode_tps": 279.21, "prompt_tps": 9898.21, "note": "LFM3 base on V100 (llama-bench pp512/tg128)"}
models_new[1]["v100_probe"] = {"decode_tps": 246.08, "prompt_tps": 10962.6, "note": "LFM3-unc Q6 on V100 (llama-bench pp512/tg128)"}
models_new[2]["gpu3060_probe"] = {"decode_tps": 39.61, "prompt_tps": 1340.05, "note": "Coding-Monkey on 3060 (llama-bench pp512/tg128)"}

# tau2 empty-response diagnostic note on both LFM3 rows
models_new[0]["tau2"]["note"] = ("2026-11-05: reward 0.0 = empty-response retry artifact, 7/15 sims terminated "
                                 "empty_response (1.7B struggles with airline tool-use); not a clean agentic score")
models_new[1]["tau2"].setdefault("note", "")
# Qwopus sanity note
models_new[3]["failures"] = [{"benchmark": "sanity+prefill (lane)", "attempt": "remeasure",
    "error": "in-lane sanity+prefill failed because the gate double-loaded Qwen-27B while agent resident (V100 OOM); clean solo remeasure: sanity 92%, prefill 792.94 t/s" }]

prog = json.load(open(PROG))
existing = {m["name"]: m for m in prog["models"]}
for m in models_new:
    nm = m["name"]
    if nm in existing:
        old = existing[nm]
        # preserve lane-recorded benchkit (has results_dir) from the partial row
        oldbk = old.get("benchkit", {})
        m.setdefault("benchkit", oldbk)
        m["file"] = old.get("file", m["file"])
        m["gpu"] = old.get("gpu", m["gpu"])
        prog["models"][prog["models"].index(old)] = m
        print(f"~ updated existing row: {nm}")
    else:
        prog["models"].append(m)
        print(f"+ progress row: {nm}")
json.dump(prog, open(PROG, "w"), indent=1)
print("progress.json updated, n models =", len(prog["models"]))

# ---- prefill probes into bench_results.json ----
bres = json.load(open(BRES))
for m in models_new:
    fname = m["file"]
    base = fname.replace(".gguf", "")
    key = f"{base}.gguf [no-spec]"
    pp = m["prompt_tps"]
    bres.setdefault(key, {})
    bres[key] = {"pp512": pp, "tg128": m["decode_tps"], "prompt_eval_tps": pp}
    print(f"+ bench_results prefill: {key} pp512={pp}")
json.dump(bres, open(BRES, "w"), indent=1)
print("bench_results.json updated")

# ---- regenerate report ----
os.chdir(ROOT)
r = subprocess.run([sys.executable, "scripts/generate_report.py"], capture_output=True, text=True)
print("report rc:", r.returncode)
if r.stderr.strip():
    print("stderr tail:", r.stderr.strip()[-500:])
