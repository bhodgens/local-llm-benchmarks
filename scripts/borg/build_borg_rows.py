#!/usr/bin/env python3
"""Generate borg's report in the repo's report.html style + merge borg rows into report.html.

Outputs:
  1. /root/bench/BORG_REPORT.html   — restyled with report.html's design language
  2. borg rows appended to the repo report.html data (see merge step in repo)

Data sources:
  - /root/bench/coding-eval-progress.json   (LCB + tau2)
  - /root/bench/results/humaneval_local.jsonl (HumanEval pass@1)
  - /root/bench/bench_results_borg.json      (throughput lanes)
"""
import json, glob

PROGRESS = "/root/bench/coding-eval-progress.json"
HE_JSONL = "/root/bench/results/humaneval_local.jsonl"
LANES = "/root/bench/bench_results_borg.json"

# display name, base model, category, engine label, gpu label, decode t/s
LANE_META = {
    "qwen38-27b":        ("Qwen3.8-27B UD-IQ4_XS (borg)",        "Qwen3.8-27B",       "27B Dense", "luce_server", "R9700"),
    "qwen38-27b-vision": ("Qwen3.8-27B Vision UD-IQ4_XS (borg)", "Qwen3.8-27B",       "27B Dense", "luce_server", "R9700"),
    "laguna-xs21":       ("Laguna-XS-2.1 Q4_K_M (borg)",         "Laguna-XS-2.1",     "33B Dense", "luce_server", "R9700"),
    "laguna-s21":        ("Laguna-S-2.1 Q4_K_M (borg, split)",   "Laguna-S-2.1",      "70B Dense", "luce_server", "R9700+Strix"),
    "flashnext":         ("Qwen3.8-Flash-Next IQ3_S (borg)",     "Qwen3.8-Flash-Next","MoE 35B",   "Strata",      "R9700"),
    "swift":             ("Swift-Flash-Next IQ2_XS (borg)",      "Qwen3.8-Flash-Next","MoE 35B",   "Strata",      "R9700"),
    "coder":             ("Coder-Flash-Next IQ1_M (borg)",       "Qwen3.8-Flash-Next","MoE 35B",   "Strata",      "R9700"),
    "glm53-flash":       ("GLM-5.3-Flash EXL3 (borg)",           "GLM-5.3-Flash",     "Other",     "Kyojin EXL3", "Strix"),
    "kolibri-1":         ("Kolibri-1 Q4_K_M (borg)",             "Kolibri-1",         "MoE 35B",   "llama.cpp",   "R9700+Strix"),
}

progress = json.load(open(PROGRESS))
he = {}
for line in open(HE_JSONL):
    d = json.loads(line)
    if d.get("n", 0) > 50:
        he[d["label"]] = d.get("humaneval_pass@1")
try:
    lanes = json.load(open(LANES))
except FileNotFoundError:
    lanes = {}

def lane_tps(key):
    v = lanes.get(key) or {}
    g = (v.get("decode_greedy") or {}).get("decode_tps_avg")
    if g:
        return g
    s = (v.get("decode_sampled") or {}).get("decode_tps_avg")
    if s:
        return s
    # lanes measured outside bench_results_borg.json
    extra = {}
    try:
        for k in ("swift", "coder"):
            d = json.load(open(f"/root/bench/results/tp-{k}.json"))
            extra[k] = d["decode_tps_avg"]
        d = json.load(open("/root/bench/results/tp-laguna-s21.json"))
        extra["laguna-s21"] = d.get("decode_tps_median")
    except Exception:
        pass
    return extra.get(key)

rows = []
for m in progress["models"]:
    key = m["name"]
    if key not in LANE_META:
        continue
    name, base, cat, engine, gpu = LANE_META[key]
    lcb = (m.get("livecodebench") or {}).get("pass_at_1")
    tau2 = (m.get("tau2") or {}).get("reward")
    tau2_n = (m.get("tau2") or {}).get("num_tasks_scored")
    rows.append({
        "key": key, "name": name, "base": base, "category": cat, "engine": engine, "gpu": gpu,
        "lcb": lcb, "tau2": tau2, "tau2_n": tau2_n, "he": he.get(key),
        "tps": lane_tps(key),
        "lcb_note": "12-problem window" if lcb is not None else None,
    })

rows.sort(key=lambda r: -(r["tau2"] if r["tau2"] is not None else -1))
json.dump(rows, open("/root/bench/borg_report_rows.json", "w"), indent=1)
print("wrote /root/bench/borg_report_rows.json with", len(rows), "rows")
for r in rows:
    print(f"  {r['name'][:36]:38} LCB {r['lcb']}  tau2 {r['tau2']}  HE {r['he']}  tps {r['tps']}")
