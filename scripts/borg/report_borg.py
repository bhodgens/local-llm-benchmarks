#!/usr/bin/env python3
"""Render the borg lane results as a self-contained HTML report.

Why this exists separately from scripts/generate_report.py: that generator reads
/tmp/coding-bench/progress.json, which is produced by the CODING-EVAL campaign
(LiveCodeBench pass@1, HumanEval, tau2 reward) on the CUDA host. The borg lanes
measure SERVING THROUGHPUT instead, and their results live in
bench_results_borg.json. This script renders that file so the host's numbers are
reportable in HTML without touching the eval pipeline.

Usage:
  python3 scripts/borg/report_borg.py \\
      --lane-json /root/bench/bench_results_borg.json \\
      --kyojin-json /root/bench/results/kyojin-glm.json \\
      --memory-jsonl /root/bench/results/memory-map.jsonl \\
      --out /root/bench/BORG_REPORT.html
"""
import argparse, json, os, html
from datetime import datetime

MEMORY = {
    "qwen38-27b": (23.41, 3.9), "qwen38-27b-vision": (24.25, 4.1),
    "laguna-xs21": (20.05, 3.3), "kolibri-1": (26.93, 2.7),
    "glm53-flash": (0.06, 78.4), "flashnext": (31.48, 115.9),
    "ds4-flash": (27.65, 107.2), "ds41-flash": (27.69, 118.1),
}

SETTINGS = {
    "qwen38-27b": ("HIP_VISIBLE_DEVICES=0 (R9700)", "131072", "f16", "DFlash2 b16, greedy only"),
    "qwen38-27b-vision": ("HIP_VISIBLE_DEVICES=0", "131072", "f16", "DFlash2 b16, greedy only"),
    "laguna-xs21": ("HIP_VISIBLE_DEVICES=0", "131072", "f16", "DFlash q4; --kvflash auto"),
    "kolibri-1": ("both GPUs visible", "32768", "q8_0", "none (no MTP in this path)"),
    "glm53-flash": ("GGML_VK_VISIBLE_DEVICES=1 (Strix only)", "65536", "q8_0", "MTP depth 4 launched, never engages"),
    "flashnext": ("R9700 as expert cache", "131072", "int8 (32768 resident)", "MTP spec 4 (not reported by API)"),
    "ds4-flash": ("--target-device hip:0 --expert-device hip:1 --peer-access", "18432", "f16", "DSpark, sampled too"),
    "ds41-flash": ("--profile ds41-lucebox (3-tier)", "131072", "f16", "DSpark"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lane-json", default="/root/bench/bench_results_borg.json")
    ap.add_argument("--kyojin-json", default="/root/bench/results/kyojin-glm.json")
    ap.add_argument("--memory-jsonl", default="/root/bench/results/memory-map.jsonl")
    ap.add_argument("--out", default="/root/bench/BORG_REPORT.html")
    args = ap.parse_args()

    lanes = json.load(open(args.lane_json))
    try:
        kyojin = json.load(open(args.kyojin_json))
    except Exception:
        kyojin = None
    mem = {}
    if os.path.exists(args.memory_jsonl):
        for line in open(args.memory_jsonl):
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    if "r9700_used_gb" in r:
                        mem[r["model"]] = r
                except Exception:
                    pass

    e = html.escape
    rows = []
    for key in sorted(lanes):
        v = lanes[key]
        s, g = v.get("decode_sampled", {}), v.get("decode_greedy", {})
        eng = v.get("engine", "")
        dev, ctx, kv, spec = SETTINGS.get(key, ("—", "—", "—", "—"))
        m = mem.get(key, {})
        rows.append(f"""      <tr>
        <td>{e(key)}</td><td>{e(eng)}</td><td>{e(dev)}</td>
        <td class="num">{e(ctx)}</td><td>{e(kv)}</td><td>{e(spec)}</td>
        <td class="num">{v.get('prefill_first_request_tps','—')}</td>
        <td class="num">{v.get('prefill_cold_tps','—')}</td>
        <td class="num">{v.get('prefill_long_tps','—')}</td>
        <td class="num">{s.get('decode_tps_avg','—')}</td>
        <td class="num"><b>{g.get('decode_tps_avg','—')}</b></td>
        <td class="num">{g.get('accept_rate_avg','—')}</td>
        <td class="num">{v.get('boot_s','—')}</td>
        <td class="num">{m.get('r9700_used_gb','—')}</td>
        <td class="num">{m.get('host_ram_used_gb','—')}</td>
      </tr>""")

    kyojin_block = ""
    if kyojin:
        r = kyojin.get("summary") or {}
        kyojin_block = f"""  <h2>Kyojin lane (own harness — see scripts/bench_kyojin_glm.py)</h2>
  <table>
    <tr><th>model</th><th>engine</th><th class="num">prefill t/s</th><th class="num">decode t/s</th><th>note</th></tr>
    <tr><td>glm-5.3-flash EXL3</td><td>Kyojin (ExLlamaV3 ROCm, gfx1151)</td>
        <td class="num">{r.get('probe_server_prefill_tps', 0):.1f} @{r.get('probe_prompt_tokens','?')} tok</td>
        <td class="num">{r.get('code_mean_server_decode_tps', 0):.2f}</td>
        <td>warm run; first start builds a dense-GEMM tune cache for ~13 min</td></tr>
  </table>"""

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>borg lane report — R9700 + Strix Halo</title>
<style>
 body {{ background:#0d0f13; color:#e8ecf1; margin:0; padding:32px 20px 64px;
        font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
 .wrap {{ max-width:1400px; margin:0 auto; }}
 h1 {{ font-size:22px; margin:0 0 6px; }}
 h2 {{ font-size:15px; margin:32px 0 10px; color:#93a0b4; text-transform:uppercase; letter-spacing:.08em; }}
 table {{ width:100%; border-collapse:collapse; font-size:12px; }}
 th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid #222933; vertical-align:top; }}
 th {{ color:#93a0b4; font-size:11px; text-transform:uppercase; letter-spacing:.04em; }}
 td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
 .note {{ color:#93a0b4; font-size:12px; }}
 .warn {{ border-left:3px solid #ff6b6b; padding-left:10px; color:#ffb3b3; font-size:12px; }}
</style></head><body><div class="wrap">
<h1>borg lane report</h1>
<div class="note">Serving throughput on the R9700 + Ryzen AI MAX+ 395 (Strix Halo), ROCm 7.2.1.
Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by scripts/borg/report_borg.py.
This is the THROUGHPUT half of the benchmark picture. LiveCodeBench / HumanEval / tau2
scores — the columns in the repo's main report.html — were not run on this host.</div>

<h2>Lane results</h2>
<table>
  <tr>
    <th>model</th><th>engine/server</th><th>device pinning</th><th class="num">ctx</th><th>KV</th>
    <th>spec decode</th><th class="num">prefill 1st</th><th class="num">prefill cold</th>
    <th class="num">prefill 2.5K</th><th class="num">decode sampled</th><th class="num">decode greedy</th>
    <th class="num">accept</th><th class="num">boot s</th><th class="num">R9700 GB</th><th class="num">RAM GB</th>
  </tr>
{chr(10).join(rows)}
</table>

{kyojin_block}

<h2>Caveats that apply to every row</h2>
<div class="warn">
  <b>No quality scores here.</b> These are speed measurements. Quality claims in BORG_RESULTS.md come from
  publisher evaluations of the same quantizations, not from runs on this host. Comparing engines on speed alone
  is safe; comparing models on capability is not.
</div>
<div class="note" style="margin-top:10px">
  "prefill 1st" is the first request after a load and includes kernel warmup — it reads ~2x low on some engines.
  "decode greedy" is temperature 0, which is where drafters engage; "decode sampled" is the harness default 0.3.
  Memory columns are measured with rocm-smi/free while the model was loaded and idle.
</div>
</div></body></html>
"""
    with open(args.out, "w") as f:
        f.write(doc)
    print("wrote", args.out, len(doc), "bytes")


if __name__ == "__main__":
    main()
