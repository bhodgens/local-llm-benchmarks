#!/usr/bin/env python3
"""Render BORG_REPORT.html in the repo report.html style (dark GitHub theme,
sortable table, bar cells, badges, detail modals) plus a Run Date column.

Restored 2026-11-07 (the original scripts/borg/render_borg_report.py and
BORG_REPORT.html were removed in the bf23aec cleanup; row data recovered from
git history: LCB75 scores from 49b20d1, tau2 self-play scores from 6219bf5,
GLM decode corrected to the Kyojin warm number 33.7 t/s per BORG_RESULTS.md).

Row data lives inline below (BORG_ROWS). Dates = commit dates of the runs
entering the repo. To add a row: append to BORG_ROWS with a bench_date.

Usage (repo host):  python3 scripts/borg/render_borg_report.py [--out BORG_REPORT.html]
"""
import argparse, json, html, os

# One row per borg lane with quality scores. LCB = 75-problem window
# (V100 question set, 49b20d1) EXCEPT GLM-5.3-Flash (12-problem window —
# only run completed there; 25.0% at 1/12 granularity). tau2 = airline
# 15 tasks, SELF-PLAY user simulator — not comparable to V100-simulated runs.
# kolibri-1 excluded from LCB75 per user instruction (keeps its 12-problem 41.7%).
BORG_ROWS = [
    {"name": "Qwen3.8-Flash-Next IQ3_S (borg)", "base": "Qwen3.8-Flash-Next",
     "category": "MoE 35B", "engine": "Strata", "gpu": "R9700",
     "tps": 86.31, "he": 0.8232, "lcb": 0.973, "tau2": 0.7333, "tau2_n": 15,
     "bench_date": "2026-10-07", "notes": "LCB75 97.3% (best on borg); production flashnext serving lane"},
    {"name": "Swift-Flash-Next IQ2_XS (borg)", "base": "Qwen3.8-Flash-Next",
     "category": "MoE 35B", "engine": "Strata", "gpu": "R9700",
     "tps": 42.08, "he": 0.8963, "lcb": 0.947, "tau2": 0.8, "tau2_n": 15,
     "bench_date": "2026-10-07", "notes": "LCB75 94.7%; tau2 0.80 self-play"},
    {"name": "Coder-Flash-Next IQ1_M (borg)", "base": "Qwen3.8-Flash-Next",
     "category": "MoE 35B", "engine": "Strata", "gpu": "R9700",
     "tps": 40.5, "he": 0.8537, "lcb": 0.92, "tau2": 1.0, "tau2_n": 5,
     "bench_date": "2026-10-07", "notes": "only tau2 1.0 (5/5 scored); LCB75 92.0%"},
    {"name": "Qwen3.8-27B UD-IQ4_XS (borg)", "base": "Qwen3.8-27B",
     "category": "27B Dense", "engine": "luce_server", "gpu": "R9700",
     "tps": 138.63, "he": 0.9512, "lcb": 0.88, "tau2": 0.6, "tau2_n": 10,
     "bench_date": "2026-10-07", "notes": "balanced pick: 95.1% HumanEval, 138.6 t/s"},
    {"name": "Qwen3.8-27B Vision UD-IQ4_XS (borg)", "base": "Qwen3.8-27B",
     "category": "27B Dense", "engine": "luce_server", "gpu": "R9700",
     "tps": 127.7, "he": 0.9512, "lcb": 0.88, "tau2": 0.6, "tau2_n": 5,
     "bench_date": "2026-10-07", "notes": "vision sibling, same quality profile"},
    {"name": "Laguna-XS-2.1 Q4_K_M (borg)", "base": "Laguna-XS-2.1",
     "category": "33B Dense", "engine": "luce_server", "gpu": "R9700",
     "tps": 129.69, "he": 0.8902, "lcb": 0.84, "tau2": 0.6, "tau2_n": 5,
     "bench_date": "2026-10-07", "notes": "best prefill measured on borg (2237 t/s @2.5K)"},
    {"name": "Laguna-S-2.1 Q4_K_M (borg, split)", "base": "Laguna-S-2.1",
     "category": "70B Dense", "engine": "luce_server", "gpu": "R9700+Strix",
     "tps": 22.1, "he": None, "lcb": 0.84, "tau2": None, "tau2_n": None,
     "bench_date": "2026-10-07", "notes": "split across both devices (source-patched loader); tau2 N/A (ctx-window failures all 15 tasks)"},
    {"name": "Kolibri-1 Q4_K_M (borg)", "base": "Kolibri-1",
     "category": "MoE 35B", "engine": "llama.cpp", "gpu": "R9700+Strix",
     "tps": 54.32, "he": 0.6098, "lcb": 0.4167, "tau2": 0.2857, "tau2_n": 7,
     "bench_date": "2026-10-06",
     "notes": "LCB is the 12-problem window (excluded from LCB75 per user instruction)"},
    {"name": "GLM-5.3-Flash EXL3 (borg)", "base": "GLM-5.3-Flash",
     "category": "Other", "engine": "Kyojin EXL3", "gpu": "Strix",
     "tps": 33.7, "he": None, "lcb": 0.25, "tau2": 0.6, "tau2_n": 15,
     "bench_date": "2026-10-08",
     "notes": "decode 33.7 t/s = Kyojin warm run (BORG_RESULTS.md); LCB is the 12-problem window; tau2 final 0.60 (v4 runner)"},
]

HOST_SPECS = [
    ("APU", "AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (Strix Halo, gfx1151), 32 threads — 124 GB unified pool, hip:1"),
    ("GPU", "Radeon AI PRO R9700 (gfx1201), 32 GB GDDR6, PCIe Gen5 x16 — hip:0"),
    ("RAM / disk", "124 GB total (APU pool carved from it) · 1.9 TB NVMe"),
    ("OS / ROCm", "Ubuntu, kernel 6.17.0-1032-oem · ROCm 7.2.1"),
    ("Engines", "/root/lucebox (luce_server HIP) · /root/strata (HIP) · /root/kyojin (ExLlamaV3) · llama.cpp-kolibri (patched HIP)"),
]

LCB75_NOTE = ("LiveCodeBench pass@1: 75-problem V100 question set (2026-10-07) except "
              "GLM-5.3-Flash and Kolibri-1 (12-problem date window 2025-04). NOT comparable "
              "to the CUDA host's 75-problem runs.")
TAU2_NOTE = ("tau2-bench: airline, 15 tasks, SELF-PLAY user simulator (no separate sim host) — "
             "not comparable to V100-simulated runs. N/A = context-window failures or not run.")
HE_NOTE = "HumanEval: 164 problems, execution-scored."


def bar(value, max_val, fmt, color):
    if value is None:
        return '<td class="center"><span class="na">N/A</span></td>'
    pct = min(100, value / max_val * 100) if max_val else 0
    return (f'<td class="center"><div class="bar-container">'
            f'<div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div>'
            f'<span class="bar-label">{fmt(value)}</span></div></td>')


pct = lambda v: '%.1f%%' % (v * 100)
tau2f = lambda v: '%.3f' % v
tpsf = lambda v: '%.1f' % v

he_max = 1.0
tps_vals = [r["tps"] for r in BORG_ROWS if r["tps"] is not None]
lcb_vals = [r["lcb"] for r in BORG_ROWS if r["lcb"] is not None]
tau2_vals = [r["tau2"] for r in BORG_ROWS if r["tau2"] is not None]
tps_max = max(tps_vals) if tps_vals else 1
lcb75_max = 0.973   # fixed scale so the 75-problem scores don't dwarf 12-problem ones
tau2_max = max(tau2_vals) if tau2_vals else 1

badge = {"27B Dense": "badge-27b", "MoE 35B": "badge-moe", "33B Dense": "badge-moe",
         "70B Dense": "badge-moe", "Other": "badge-other"}

detail_data = []
body_rows = ""
for i, r in enumerate(BORG_ROWS, 1):
    jname = json.dumps(r["name"])
    detail = {
        "name": r["name"], "category": r["category"], "gpu": r["gpu"],
        "Engine": r["engine"], "Base model": r["base"],
        "Run date": r["bench_date"],
        "LiveCodeBench": LCB75_NOTE if r["lcb"] is not None else "not scored",
        "tau2-bench": (TAU2_NOTE + (f" ({r['tau2_n']} tasks scored)" if r["tau2_n"] else ""))
                      if r["tau2"] is not None else "not scored",
        "HumanEval": HE_NOTE if r["he"] is not None else "not scored",
        "Notes": r["notes"],
    }
    detail_data.append(detail)
    cb = badge.get(r["category"], "badge-other")

    body_rows += f'''<tr>
  <td class="center">{i}</td>
  <td data-sort="{r["name"].lower()}" class="has-detail">
    <span class="model-name" onclick='showDetail({jname})'>{html.escape(r["name"])}</span>
    <span class="info-icon" onclick='showDetail({jname})'>i</span>
  </td>
  <td class="center">{html.escape(r["base"])}</td>
  <td class="center"><span class="badge {cb}">{r["category"]}</span></td>
  <td class="center">{html.escape(r["engine"])}</td>
  <td class="center">{r["gpu"]}</td>
  <td class="center">{r["bench_date"]}</td>
  {bar(r["tps"], tps_max, tpsf, "#bc8cff")}
  {bar(r["he"], he_max, pct, "#238636")}
  {bar(r["lcb"], lcb75_max, pct, "#79c0ff")}
  {bar(r["tau2"], tau2_max, tau2f, "#56d364")}
</tr>'''

detail_json = json.dumps(detail_data)

html_doc = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Benchmark Results — borg</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 5px; font-size: 1.8em; }
h2 { color: #8b949e; margin: 25px 0 10px; font-size: 1.3em; border-bottom: 1px solid #30363d; padding-bottom: 5px; }
.subtitle { color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }
table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
th { background: #161b22; color: #58a6ff; padding: 10px 8px; text-align: left; font-size: 0.85em; text-transform: uppercase; border-bottom: 2px solid #30363d; cursor: pointer; user-select: none; }
th:hover { background: #1c2331; color: #79c0ff; }
th.sorted-asc::after { content: " \\25B2"; color: #3fb950; }
th.sorted-desc::after { content: " \\25BC"; color: #3fb950; }
td { padding: 8px; border-bottom: 1px solid #21262d; font-size: 0.9em; }
tr:hover { background: #161b22; }
.center { text-align: center; }
.na { color: #484f58; }
.bar-container { position: relative; width: 100%; min-width: 90px; background: #21262d; border-radius: 4px; height: 20px; overflow: hidden; }
.bar-fill { height: 100%; border-radius: 4px; }
.bar-label { position: absolute; left: 8px; top: 0; line-height: 20px; font-size: 0.8em; color: #c9d1d9; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }
.badge-27b { background: #1f6feb33; color: #58a6ff; }
.badge-moe { background: #8957e533; color: #bc8cff; }
.badge-other { background: #30363d; color: #8b949e; }
.model-name { color: #58a6ff; cursor: pointer; text-decoration: none; }
.model-name:hover { text-decoration: underline; }
.info-icon { display: inline-block; margin-left: 6px; width: 16px; height: 16px; line-height: 16px; text-align: center; background: #30363d; color: #8b949e; border-radius: 50%; font-size: 0.7em; cursor: pointer; }
.info-icon:hover { background: #1f6feb; color: #fff; }
.modal { display: none; position: fixed; z-index: 10; left: 0; top: 0; width: 100%; height: 100%; background: rgba(1,4,9,0.8); }
.modal-content { background: #161b22; margin: 10% auto; padding: 24px; border: 1px solid #30363d; border-radius: 8px; width: 90%; max-width: 640px; }
.close { float: right; font-size: 1.4em; cursor: pointer; color: #8b949e; }
.close:hover { color: #fff; }
.modal-content h3 { color: #58a6ff; margin-bottom: 12px; }
.modal-content dt { color: #8b949e; font-size: 0.8em; text-transform: uppercase; margin-top: 10px; }
.modal-content dd { margin: 2px 0 0 0; }
.legend { color: #8b949e; font-size: 0.85em; margin: 8px 0 16px; }
.specs td:first-child { color: #8b949e; text-transform: uppercase; font-size: 0.8em; width: 140px; }
</style>
</head>
<body>
<h1>LLM Benchmark Results — borg (R9700 + Strix Halo)</h1>
<div class="subtitle">Serving lanes + coding-eval on the AMD AI Max duo · Generated by scripts/borg/render_borg_report.py</div>

<h2>Host</h2>
<table class="specs">"""

for k, v in HOST_SPECS:
    html_doc += f'<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>'
html_doc += """</table>

<h2>Full Results (click headers to sort, click model name for details)</h2>
<div class="legend"><strong>Benchmarks:</strong> """ + LCB75_NOTE + """<br>
  """ + TAU2_NOTE + """<br>
  """ + HE_NOTE + """<br>
  <strong>tok/s:</strong> greedy decode; the lane each model was tuned to. Two-model split lanes (R9700+Strix) pay a 2-9% co-residency tax.<br>
  <strong>Run Date:</strong> commit date of the run entering the repo.<br>
  <strong>Detail view:</strong> click a model name for engine, device layout, and protocol notes.</div>
<table id="resultsTable">
<thead><tr>
  <th data-type="number" data-key="idx">#</th>
  <th data-type="string" data-key="name">Model</th>
  <th data-type="string" data-key="base">Base</th>
  <th data-type="string" data-key="category">Type</th>
  <th data-type="string" data-key="engine">Engine</th>
  <th data-type="string" data-key="gpu">GPU</th>
  <th data-type="string" data-key="bench_date">Run Date</th>
  <th data-type="number" data-key="tps">tok/s</th>
  <th data-type="number" data-key="he">HumanEval</th>
  <th data-type="number" data-key="lcb">LiveCodeBench</th>
  <th data-type="number" data-key="tau2">tau2-bench</th>
</tr></thead>
<tbody>
""" + body_rows + """</tbody>
</table>

<h2>Key Findings</h2>
<div class="legend">
<ul>
<li><strong>Flash Next fine-tunes punch up:</strong> swift (IQ2_XS, 42 t/s) scores LCB75 94.7% / tau2 0.80 and coder (IQ1_M) posts the only tau2 1.0 — above base Flash Next (LCB75 97.3%, tau2 0.73) on agentic work at half the speed.</li>
<li><strong>Qwen 27B remains the balanced pick:</strong> 95% HumanEval, 139 t/s, tau2 0.60, LCB75 88%.</li>
<li><strong>LCB caveats:</strong> GLM and Kolibri-1 rows are 12-problem-window scores (±8% granularity at 1/12); treat ordering, not absolute numbers, across that boundary.</li>
<li><strong>tau2 caveats:</strong> self-play user simulator (no separate host), and laguna-s21 hit context-window errors on all 15 tasks, so its tau2 is N/A.</li>
<li><strong>GLM decode:</strong> 33.7 t/s is the Kyojin EXL3 warm run (first start builds a dense-GEMM tune cache for ~13 min; the cold 14.2 t/s lane is not the engine's number).</li>
</ul>
</div>

<div class="modal" id="detailModal"><div class="modal-content"><span class="close" onclick="document.getElementById('detailModal').style.display='none'">&times;</span><div id="detailBody"></div></div></div>

<script>
const DETAIL_DATA = """ + detail_json + """;
const DETAIL_MAP = {};
DETAIL_DATA.forEach(d => { DETAIL_MAP[d.name] = d; });
function showDetail(name) {
  const d = DETAIL_MAP[name]; if (!d) return;
  let html = '<h3>' + name + '</h3><dl>';
  for (const [k, v] of Object.entries(d)) {
    if (k === 'name') continue;
    html += '<dt>' + k + '</dt><dd>' + v + '</dd>';
  }
  html += '</dl>';
  document.getElementById('detailBody').innerHTML = html;
  document.getElementById('detailModal').style.display = 'block';
}
document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') document.getElementById('detailModal').style.display = 'none';
});
document.querySelectorAll('#resultsTable th').forEach(th => {
  th.addEventListener('click', () => {
    const tbody = document.querySelector('#resultsTable tbody');
    const rows = [...tbody.rows];
    const asc = !th.classList.contains('sorted-asc');
    document.querySelectorAll('#resultsTable th').forEach(t => t.classList.remove('sorted-asc','sorted-desc'));
    th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
    const idx = th.cellIndex;
    const key = th.dataset.key;
    rows.sort((a, b) => {
      const ac = a.cells[idx], bc = b.cells[idx];
      let av = (ac.querySelector('.bar-label')||ac).textContent.trim();
      let bv = (bc.querySelector('.bar-label')||bc).textContent.trim();
      if (av === 'N/A' || av === '-') av = key === 'tau2' ? -1 : (key === 'idx' ? 0 : '');
      if (bv === 'N/A' || bv === '-') bv = key === 'tau2' ? -1 : (key === 'idx' ? 0 : '');
      if (key === 'idx') { av = parseInt(av); bv = parseInt(bv); return asc ? av - bv : bv - av; }
      const an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn) && /^[-0-9.]/.test(av) && /^[-0-9.]/.test(bv)) return asc ? an - bn : bn - an;
      return asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
</script>
</body>
</html>"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))), "BORG_REPORT.html"))
    args = ap.parse_args()
    with open(args.out, "w") as f:
        f.write(html_doc)
    print("wrote", args.out, len(html_doc), "bytes,", len(BORG_ROWS), "rows")

if __name__ == "__main__":
    main()
