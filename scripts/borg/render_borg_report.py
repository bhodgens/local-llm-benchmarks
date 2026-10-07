#!/usr/bin/env python3
"""Render BORG_REPORT.html in the repo report.html style (dark GitHub theme,
sortable table, bar cells, badges, detail modals) plus a BORG_ROWS export the
repo merge step can splice into report.html."""
import json, html

ROWS = json.load(open("/root/bench/borg_report_rows.json"))

HOST_SPECS = [
    ("APU", "AMD Ryzen AI MAX+ 395 w/ Radeon 8060S (Strix Halo, gfx1151), 32 threads — 124 GB unified pool, hip:1"),
    ("GPU", "Radeon AI PRO R9700 (gfx1201), 32 GB GDDR6, PCIe Gen5 x16 — hip:0"),
    ("RAM / disk", "124 GB total (APU pool carved from it) · 1.9 TB NVMe"),
    ("OS / ROCm", "Ubuntu, kernel 6.17.0-1032-oem · ROCm 7.2.1"),
    ("Engines", "/root/lucebox (luce_server HIP) · /root/strata (HIP) · /root/kyojin (ExLlamaV3) · llama.cpp-kolibri (patched HIP)"),
]

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

lcb_max = max(r["lcb"] for r in ROWS if r["lcb"] is not None)
tau2_max = max(r["tau2"] for r in ROWS if r["tau2"] is not None)
tps_max = max(r["tps"] for r in ROWS if r["tps"] is not None)
he_max = 1.0

badge = {"27B Dense": "badge-27b", "MoE 35B": "badge-moe", "33B Dense": "badge-moe",
         "70B Dense": "badge-moe", "Other": "badge-other"}

detail_data = []
body_rows = ""
for i, r in enumerate(ROWS, 1):
    jname = json.dumps(r["name"])
    detail = {
        "name": r["name"], "category": r["category"], "gpu": r["gpu"],
        "Engine": r["engine"], "Base model": r["base"],
        "LiveCodeBench": "12-problem date window (2025-04), pass@1" if r["lcb"] is not None else "not scored",
        "tau2-bench": ("airline, 15 tasks, self-play user simulator"
                        + (f", {r['tau2_n']} tasks scored" if r["tau2_n"] else "") if r["tau2"] is not None else "not scored"),
        "HumanEval": "164 problems, execution-scored" if r["he"] is not None else "not scored",
        "Notes": "GLM lanes: EXL3 engine on Strix Halo; Laguna-S: split across both devices (source-patched loader)",
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
  {bar(r["tps"], tps_max, tpsf, "#bc8cff")}
  {bar(r["he"], he_max, pct, "#238636")}
  {bar(r["lcb"], lcb_max, pct, "#79c0ff")}
  {bar(r["tau2"], tau2_max, tau2f, "#56d364")}
</tr>'''

detail_json = json.dumps(detail_data)

CSS = open("/dev/null").read()  # inline below; mirrors report.html

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
<div class="legend"><strong>Benchmarks:</strong> LiveCodeBench pass@1 (12-problem date window 2025-04, thinking off — NOT comparable to the V100 75-problem runs) |
  tau2-bench (airline, 15 tasks, <em>self-play</em> user simulator — not comparable to V100-simulated runs) |
  HumanEval (164 problems, execution-scored).<br>
  <strong>tok/s:</strong> greedy decode; the lane each model was tuned to. Two-model split lanes (R9700+Strix) pay a 2-9% co-residency tax.<br>
  <strong>Detail view:</strong> click a model name for engine, device layout, and protocol notes.</div>
<table id="resultsTable">
<thead><tr>
  <th data-type="number" data-key="idx">#</th>
  <th data-type="string" data-key="name">Model</th>
  <th data-type="string" data-key="base">Base</th>
  <th data-type="string" data-key="category">Type</th>
  <th data-type="string" data-key="engine">Engine</th>
  <th data-type="string" data-key="gpu">GPU</th>
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
<li><strong>Flash Next fine-tunes punch up:</strong> swift (IQ2_XS, 42 t/s) scores LCB 41.7% / tau2 0.80 and coder (IQ1_M) posts the only tau2 1.0 — above base Flash Next (LCB 50%, tau2 0.73) on agentic work at half the speed.</li>
<li><strong>kolibri-1 leads LCB (41.7%) but trails everywhere else</strong> (HumanEval 61%, tau2 0.29): competitive-programming style problems suit it; production coding does not.</li>
<li><strong>Qwen 27B remains the balanced pick:</strong> 95% HumanEval, 139 t/s, tau2 0.60, LCB 25%.</li>
<li><strong>LCB caveats:</strong> 12-problem window at borg speeds — every score carries a ±9% granularity (1/12). Treat LCB ordering, not the absolute numbers.</li>
<li><strong>tau2 caveats:</strong> self-play user simulator (no separate V100), and laguna-s21 hit context-window errors on all 15 tasks (reasoning length), so its tau2 is N/A.</li>
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
document.querySelectorAll('#resultsTable th').forEach(th => {
  th.addEventListener('click', () => {
    const key = th.dataset.key, tbody = document.querySelector('#resultsTable tbody');
    const rows = [...tbody.rows];
    const asc = !th.classList.contains('sorted-asc');
    document.querySelectorAll('#resultsTable th').forEach(t => t.classList.remove('sorted-asc','sorted-desc'));
    th.classList.add(asc ? 'sorted-asc' : 'sorted-desc');
    rows.sort((a, b) => {
      let av, bv;
      const ai = [...a.cells].findIndex(c => c.querySelector('.model-name') || c.querySelector('.bar-label') || c.textContent.trim());
      const idx = th.cellIndex;
      const ac = a.cells[idx], bc = b.cells[idx];
      av = parseFloat((ac.querySelector('.bar-label')||ac).textContent) || ac.textContent.trim();
      bv = parseFloat((bc.querySelector('.bar-label')||bc).textContent) || bc.textContent.trim();
      if (typeof av === 'number' && typeof bv === 'number') return asc ? av - bv : bv - av;
      return asc ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    rows.forEach(r => tbody.appendChild(r));
  });
});
</script>
</body>
</html>"""

with open("/root/bench/BORG_REPORT.html", "w") as f:
    f.write(html_doc)

# export rows for the repo merge step
with open("/root/bench/borg_report_rows.json") as f:
    rows = json.load(f)
with open("/root/bench/borg_rows_for_report.html.json", "w") as f:
    json.dump(rows, f, indent=1)

print("wrote /root/bench/BORG_REPORT.html (report.html style) and borg_rows_for_report.html.json")
