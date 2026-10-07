#!/usr/bin/env python3
"""Splice the borg rows into report.html as a separate sortable section.

The main table stays V100/3060-only (different protocols); borg gets its own
section with the same styling, inserted after the main table's Key Findings.
Data: scripts/borg/borg_rows_for_report.json (copied from borg).
"""
import json, html, re, sys

REPORT = "report.html"
ROWS = json.load(open("scripts/borg/borg_rows_for_report.json"))

s = open(REPORT).read()
if "BORG (R9700 + Strix Halo)" in s:
    print("borg section already present; removing old one first")
    s = re.sub(r'<h2>BORG \(R9700 \+ Strix Halo\).*?(?=<h2>|\Z)', '', s, flags=re.S)

lcb_max = max(r["lcb"] for r in ROWS if r["lcb"] is not None)
tau2_max = max(r["tau2"] for r in ROWS if r["tau2"] is not None)
tps_max = max(r["tps"] for r in ROWS if r["tps"] is not None)

bar = lambda v, mx, fmt, color: (
    '<td class="center"><span class="na">N/A</span></td>' if v is None else
    f'<td class="center"><div class="bar-container"><div class="bar-fill" style="width:{min(100, v/mx*100):.0f}%;background:{color}"></div>'
    f'<span class="bar-label">{fmt(v)}</span></div></td>')
pct = lambda v: '%.1f%%' % (v * 100)
tau2f = lambda v: '%.3f' % v
tpsf = lambda v: '%.1f' % v

rows_html = ""
for i, r in enumerate(sorted(ROWS, key=lambda x: -(x["tau2"] or -1)), 1):
    rows_html += f'''<tr>
  <td class="center">{i}</td>
  <td class="center">{html.escape(r["name"])}</td>
  <td class="center">{html.escape(r["base"])}</td>
  <td class="center">{r["category"]}</td>
  <td class="center">{html.escape(r["engine"])}</td>
  <td class="center">{r["gpu"]}</td>
  {bar(r["tps"], tps_max, tpsf, "#bc8cff")}
  {bar(r["he"], 1.0, pct, "#238636")}
  {bar(r["lcb"], lcb_max, pct, "#79c0ff")}
  {bar(r["tau2"], tau2_max, tau2f, "#56d364")}
</tr>'''

section = f'''<h2>BORG (R9700 + Strix Halo) — dual-AMD host, separate protocols</h2>
<div class="subtitle">
  9 model lanes on the R9700 (32 GB GDDR6, hip:0) + Ryzen AI MAX+ 395 (124 GB unified, hip:1), ROCm 7.2.1.<br>
  <strong>Protocols differ from the V100 table:</strong> LiveCodeBench is a 12-problem date window (not 75); tau2 uses a
  self-play user simulator on the same server (not the V100 Qwopus simulator); HumanEval is execution-scored over all 164
  problems. tok/s = greedy decode on each lane's tuned engine (luce_server / Strata / Kyojin EXL3 / llama.cpp).<br>
  Generated {__import__('datetime').datetime.now().strftime('%Y-%m-%d')} by scripts/borg/build_borg_rows.py + scripts/borg/merge_borg_into_report.py.
</div>
<table>
<thead><tr>
  <th>#</th><th>Model</th><th>Base</th><th>Type</th><th>Engine</th><th>GPU</th>
  <th>tok/s</th><th>HumanEval</th><th>LiveCodeBench</th><th>tau2-bench</th>
</tr></thead>
<tbody>
{rows_html}</tbody>
</table>
<div class="subtitle">Key borg findings: Flash-Next fine-tunes punch above their quant (swift IQ2_XS tau2 0.80, coder IQ1_M tau2 1.0);
kolibri-1 leads LCB but trails HumanEval/tau2; Qwen3.8-27B is the balanced pick (95% HE, 139 t/s). Full detail: BORG_RESULTS.md / BORG_REPORT.html.</div>

'''

# insert before "How to Read" style trailing sections if present, else before </body>
anchor = s.find('<div class="subtitle">Benchmarks:')
if anchor == -1:
    anchor = s.rfind('</table>')
    anchor = s.find('</div>', anchor) + len('</div>')
insert_at = s.find('<h2>', s.find('Key Findings'))
if insert_at == -1:
    insert_at = s.rfind('</body>')
s = s[:insert_at] + section + s[insert_at:]

open(REPORT, "w").write(s)
print("report.html: borg section inserted,", len(ROWS), "rows")
