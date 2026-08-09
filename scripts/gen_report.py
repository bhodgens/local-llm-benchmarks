#!/usr/bin/env python3
"""
Regenerate report.html from progress.json.
Includes all models with LCB/tau2/tok/s results, sortable table, detail modal.
Dark theme matching the existing report.
"""
import json, html
from datetime import datetime

with open('/tmp/coding-bench/progress.json') as f:
    data = json.load(f)

models = data['models']

# Collect rows
rows = []
details = []

for m in models:
    name = m.get('name', 'Unknown')

    lcb = m.get('livecodebench', {})
    tau2 = m.get('tau2', {})
    tps = m.get('decode_tps')
    he = m.get('humaneval', {})
    cat = m.get('category', 'Other')

    lcb_val = lcb.get('pass_at_1')
    tau2_val = tau2.get('reward')
    he_val = he.get('pass_at_1')
    tau2_time = tau2.get('wall_time_s')

    # Determine badge class
    if 'MoE' in str(cat) or 'moe' in str(cat):
        badge_cls = 'badge-moe'
    elif '8B' in str(cat):
        badge_cls = 'badge-8b'
    elif '12B' in str(cat) or '14B' in str(cat):
        badge_cls = 'badge-12b'
    elif '27B' in str(cat) or '26B' in str(cat):
        badge_cls = 'badge-27b'
    else:
        badge_cls = 'badge-other'

    row = {
        'name': name,
        'category': cat,
        'gpu': m.get('gpu', '?'),
        'tps': tps,
        'he': he_val,
        'lcb': lcb_val,
        'tau2': tau2_val,
        'tau2_time': tau2_time / 60 if tau2_time else None,
        'file': m.get('file', ''),
        'vram': m.get('vram_mib'),
        'binary': m.get('binary', ''),
        'lcb_time': lcb.get('wall_time_s'),
    }
    rows.append(row)

    # Build detail data
    benchmarks = {}
    if lcb_val is not None:
        benchmarks['LiveCodeBench'] = {'pass@1': f"{lcb_val*100:.1f}%"}
        if lcb.get('wall_time_s'):
            benchmarks['LiveCodeBench']['Wall time'] = f"{lcb['wall_time_s']:.0f}s ({lcb['wall_time_s']/60:.0f} min)"
    if tau2_val is not None:
        benchmarks['tau2-bench'] = {'reward': f"{tau2_val:.4f}"}
        if tau2.get('wall_time_s'):
            benchmarks['tau2-bench']['Wall time'] = f"{tau2['wall_time_s']:.0f}s ({tau2['wall_time_s']/60:.0f} min)"
    if he_val is not None:
        benchmarks['HumanEval'] = {'pass@1': f"{he_val*100:.1f}%"}

    throughput = {}
    if tps:
        throughput['Decode tok/s'] = tps
    if m.get('prompt_tps'):
        throughput['Prompt tok/s'] = m['prompt_tps']

    settings = {'File': m.get('file', '')}
    if m.get('binary'):
        settings['Binary'] = m['binary']
    if m.get('vram_mib'):
        settings['VRAM'] = f"{m['vram_mib']} MiB"

    detail = {
        'name': name,
        'category': cat,
        'gpu': m.get('gpu', '?'),
        'status': m.get('status', ''),
        'settings': settings,
        'benchmarks': benchmarks if benchmarks else None,
        'throughput': throughput if throughput else None,
    }
    details.append(detail)

# Sort by tau2 descending (None last)
rows.sort(key=lambda r: (r['tau2'] is None, -(r['tau2'] or 0)))

def fmt_bar(val, max_val, color, fmt='pct'):
    if val is None:
        return '<span class="na">N/A</span>'
    pct = (val / max_val * 100) if max_val else 0
    if fmt == 'pct':
        label = f"{val*100:.1f}%"
    else:
        label = f"{val:.4f}"
    return f'<div class="bar-container"><div class="bar-fill" style="width:{pct:.0f}%;background:{color}"></div><span class="bar-label">{label}</span></div>'

def fmt_tps(val):
    if val is None:
        return '<span class="na">N/A</span>'
    pct = min(val / 220 * 100, 100)  # max ~220 tok/s
    return f'<div class="bar-container"><div class="bar-fill" style="width:{pct:.0f}%;background:#bc8cff"></div><span class="bar-label">{val:.0f}</span></div>'

# Generate HTML
now = datetime.now().strftime('%Y-%m-%d')
model_count = len(rows)

table_rows = ""
for i, r in enumerate(rows):
    badge_cls = 'badge-other'
    cat_str = str(r['category'])
    if 'MoE' in cat_str or 'moe' in cat_str:
        badge_cls = 'badge-moe'
    elif '8B' in cat_str:
        badge_cls = 'badge-8b'
    elif any(x in cat_str for x in ['12B', '14B']):
        badge_cls = 'badge-12b'
    elif any(x in cat_str for x in ['27B', '26B', '35B']):
        badge_cls = 'badge-27b'

    safe_name = html.escape(r['name'])
    sort_name = r['name'].lower()

    lcb_html = fmt_bar(r['lcb'], 1.0, '#1f6feb', 'pct') if r['lcb'] is not None else '<span class="na">N/A</span>'
    tau2_html = fmt_bar(r['tau2'], 1.0, '#d29922', 'raw') if r['tau2'] is not None else '<span class="na">N/A</span>'
    he_html = fmt_bar(r['he'], 1.0, '#238636', 'pct') if r['he'] is not None else '<span class="na">N/A</span>'
    tps_html = fmt_tps(r['tps']) if r['tps'] is not None else '<span class="na">N/A</span>'
    tau2_time_html = f"{r['tau2_time']:.0f}" if r['tau2_time'] else '<span class="na">-</span>'

    table_rows += f"""<tr>
  <td class="center">{i+1}</td>
  <td data-sort="{sort_name}" class="has-detail">
    <span class="model-name" onclick='showDetail("{safe_name}")'>{safe_name}</span>
    <span class="info-icon" onclick='showDetail("{safe_name}")'>i</span>
  </td>
  <td class="center"><span class="badge {badge_cls}">{html.escape(cat_str)}</span></td>
  <td class="center">{r['gpu']}</td>
  <td class="center">{tps_html}</td>
  <td class="center">{he_html}</td>
  <td class="center">{lcb_html}</td>
  <td class="center">{tau2_html}</td>
  <td class="center">{tau2_time_html}</td>
</tr>
"""

# Filter details to only include models in the table
detail_names = {r['name'] for r in rows}
filtered_details = [d for d in details if d['name'] in detail_names]

# Clean up None values for JSON
def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    return obj

detail_json = json.dumps(clean_json(filtered_details), ensure_ascii=False)

report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Benchmark Results</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1 {{ color: #58a6ff; margin-bottom: 5px; font-size: 1.8em; }}
h2 {{ color: #8b949e; margin: 25px 0 10px; font-size: 1.3em; border-bottom: 1px solid #30363d; padding-bottom: 5px; }}
.subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px 8px; text-align: left; font-size: 0.85em; text-transform: uppercase; border-bottom: 2px solid #30363d; cursor: pointer; user-select: none; position: relative; }}
th:hover {{ background: #1c2331; color: #79c0ff; }}
th.sorted-asc::after {{ content: " \\25B2"; color: #3fb950; }}
th.sorted-desc::after {{ content: " \\25BC"; color: #3fb950; }}
td {{ padding: 8px; border-bottom: 1px solid #21262d; font-size: 0.9em; }}
tr:hover {{ background: #161b22; }}
.center {{ text-align: center; }}
.na {{ color: #484f58; }}
.bar-container {{ position: relative; height: 22px; background: #21262d; border-radius: 3px; overflow: hidden; min-width: 80px; }}
.bar-fill {{ position: absolute; height: 100%; opacity: 0.25; border-radius: 3px; }}
.bar-label {{ position: relative; z-index: 1; line-height: 22px; padding-left: 8px; font-size: 0.85em; font-weight: 600; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }}
.badge-moe {{ background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb44; }}
.badge-8b {{ background: #3fb95022; color: #3fb950; border: 1px solid #3fb95044; }}
.badge-12b {{ background: #d2992222; color: #d29922; border: 1px solid #d2992244; }}
.badge-27b {{ background: #f8514922; color: #f85149; border: 1px solid #f8514944; }}
.badge-other {{ background: #8b949e22; color: #8b949e; border: 1px solid #8b949e44; }}
.note {{ background: #161b22; border-left: 3px solid #58a6ff; padding: 10px 15px; margin: 15px 0; font-size: 0.85em; color: #8b949e; }}
.failed {{ color: #f85149; }}
.model-name {{ cursor: pointer; color: #c9d1d9; text-decoration: none; }}
.model-name:hover {{ color: #58a6ff; }}
.info-icon {{ display: inline-block; width: 16px; height: 16px; line-height: 16px; text-align: center; border-radius: 50%; background: #21262d; color: #8b949e; font-size: 0.7em; margin-left: 6px; cursor: pointer; vertical-align: middle; transition: all 0.15s; }}
.info-icon:hover {{ background: #1f6feb; color: #fff; }}
.has-detail .info-icon {{ color: #58a6ff; }}
.modal-overlay {{ display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: flex-start; padding-top: 40px; }}
.modal-overlay.active {{ display: flex; }}
.modal {{ background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 0; max-width: 700px; width: 90%; max-height: 85vh; overflow-y: auto; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }}
.modal-header {{ padding: 18px 20px 12px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }}
.modal-header h3 {{ color: #58a6ff; font-size: 1.1em; }}
.modal-close {{ background: none; border: none; color: #8b949e; font-size: 1.4em; cursor: pointer; padding: 0 5px; line-height: 1; }}
.modal-close:hover {{ color: #f85149; }}
.modal-body {{ padding: 16px 20px; }}
.modal-section {{ margin-bottom: 16px; }}
.modal-section-title {{ color: #8b949e; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }}
.modal-kv {{ display: grid; grid-template-columns: 1fr 2fr; gap: 4px 12px; font-size: 0.88em; }}
.modal-kv .k {{ color: #8b949e; }}
.modal-kv .v {{ color: #c9d1d9; word-break: break-word; }}
.modal-kv .v.mono {{ font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace; font-size: 0.82em; }}
.modal-bench {{ background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px 12px; margin: 6px 0; }}
.modal-bench .bench-name {{ color: #58a6ff; font-weight: 600; font-size: 0.85em; margin-bottom: 4px; }}
.modal-bench .bench-kv {{ display: grid; grid-template-columns: 1fr 2fr; gap: 2px 12px; font-size: 0.82em; }}
.modal-bench .bench-kv .k {{ color: #8b949e; }}
.modal-bench .bench-kv .v {{ color: #c9d1d9; }}
</style>
</head>
<body>

<h1>LLM Benchmark Results</h1>
<div class="subtitle">
  Hardware: RTX 3060 12GB + Tesla V100 32GB |
  Date: {now} |
  {model_count} models tested |
  Click column headers to sort |
  Click model name or (i) icon for details
</div>

<div class="note">
  <strong>Benchmarks:</strong> LiveCodeBench (competitive programming, 75 problems, pass@1, thinking disabled) |
  tau2-bench (agentic tool use, airline domain, 15 tasks) |
  HumanEval (code completion, only some models completed)<br>
  <strong>tok/s:</strong> Decode throughput, flash attention on. MoE models use full GPU offload where possible.<br>
  <strong>Detail view:</strong> Click any model name to see settings, VRAM usage, wall times, errors.
</div>

<h2>Full Results (click headers to sort, click model name for details)</h2>
<table id="resultsTable">
<thead>
<tr>
  <th data-type="number" data-key="idx">#</th>
  <th data-type="string" data-key="name">Model</th>
  <th data-type="string" data-key="category">Type</th>
  <th data-type="string" data-key="gpu">GPU</th>
  <th data-type="number" data-key="tps">tok/s</th>
  <th data-type="number" data-key="he">HumanEval</th>
  <th data-type="number" data-key="lcb">LiveCodeBench</th>
  <th data-type="number" data-key="tau2">tau2-bench</th>
  <th data-type="number" data-key="tau2_time">tau2 Time (min)</th>
</tr>
</thead>
<tbody>
{table_rows}
</tbody>
</table>

<div class="modal-overlay" id="detailModal">
  <div class="modal">
    <div class="modal-header">
      <h3 id="modalTitle">Model Details</h3>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>

<script>
const DETAIL_DATA = {detail_json};
const DETAIL_MAP = {{}};
DETAIL_DATA.forEach(d => {{ DETAIL_MAP[d.name] = d; }});

function esc(s) {{
  if (s === null || s === undefined) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}}

function kvGrid(obj) {{
  let html = '<div class="modal-kv">';
  for (const [k, v] of Object.entries(obj)) {{
    const cls = (typeof v === 'string' && (v.includes('gguf') || v.includes('--'))) ? 'v mono' : 'v';
    html += `<span class="k">${{esc(k)}}</span><span class="${{cls}}">${{esc(v)}}</span>`;
  }}
  html += '</div>';
  return html;
}}

function showModal(detail) {{
  document.getElementById('modalTitle').textContent = detail.name;
  let body = '';
  if (detail.category) {{
    body += `<div style="margin-bottom:12px;color:#8b949e;font-size:0.85em">${{esc(detail.category)}} on ${{esc(detail.gpu || '?')}}</div>`;
  }}
  if (detail.settings) {{
    body += '<div class="modal-section"><div class="modal-section-title">Settings</div>';
    body += kvGrid(detail.settings);
    body += '</div>';
  }}
  if (detail.throughput) {{
    body += '<div class="modal-section"><div class="modal-section-title">Throughput</div>';
    body += kvGrid(detail.throughput);
    body += '</div>';
  }}
  if (detail.benchmarks) {{
    body += '<div class="modal-section"><div class="modal-section-title">Benchmark Details</div>';
    for (const [benchName, kv] of Object.entries(detail.benchmarks)) {{
      body += '<div class="modal-bench">';
      body += `<div class="bench-name">${{esc(benchName)}}</div>`;
      body += '<div class="bench-kv">';
      for (const [k, v] of Object.entries(kv)) {{
        body += `<span class="k">${{esc(k)}}</span><span class="v">${{esc(v)}}</span>`;
      }}
      body += '</div></div>';
    }}
    body += '</div>';
  }}
  document.getElementById('modalBody').innerHTML = body;
  document.getElementById('detailModal').classList.add('active');
}}

function showDetail(name) {{
  if (DETAIL_MAP[name]) showModal(DETAIL_MAP[name]);
}}

function closeModal() {{
  document.getElementById('detailModal').classList.remove('active');
}}

document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeModal();
}});

// Sortable table
document.addEventListener('DOMContentLoaded', function() {{
  const table = document.getElementById('resultsTable');
  const tbody = table.querySelector('tbody');
  const headers = table.querySelectorAll('th');
  let sortDir = 'desc';
  let sortKey = 'tau2';

  function getCellValue(row, key) {{
    const cells = row.querySelectorAll('td');
    switch(key) {{
      case 'idx': return parseInt(cells[0].textContent);
      case 'name': return cells[1].getAttribute('data-sort') || cells[1].textContent.toLowerCase();
      case 'category': return cells[2].textContent.trim();
      case 'gpu': return cells[3].textContent.trim();
      case 'tps': return parseFloat(cells[4].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'he': return parseFloat(cells[5].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'lcb': return parseFloat(cells[6].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'tau2': return parseFloat(cells[7].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'tau2_time': return parseFloat(cells[8].textContent.replace(/[^0-9.]/g, '')) || 99999;
      default: return 0;
    }}
  }}

  function sortTable(key, dir) {{
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {{
      const av = getCellValue(a, key);
      const bv = getCellValue(b, key);
      if (typeof av === 'string') return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      return dir === 'asc' ? av - bv : bv - av;
    }});
    rows.forEach((row, i) => {{
      tbody.appendChild(row);
      row.querySelector('td').textContent = i + 1;
    }});
  }}

  headers.forEach((th) => {{
    th.addEventListener('click', function() {{
      const key = th.getAttribute('data-key');
      if (sortKey === key) {{
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      }} else {{
        sortKey = key;
        sortDir = key === 'name' || key === 'category' ? 'asc' : 'desc';
      }}
      headers.forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
      th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');
      sortTable(key, sortDir);
    }});
  }});

  const tau2Header = table.querySelector('th[data-key="tau2"]');
  if (tau2Header) tau2Header.classList.add('sorted-desc');
}});
</script>
</body>
</html>
"""

with open('/home/caimlas/llm-benchmarks/report.html', 'w') as f:
    f.write(report_html)

print(f"Report generated: {model_count} models")
print(f"Saved to: /home/caimlas/llm-benchmarks/report.html")

# Show BTL-4 position
for i, r in enumerate(rows):
    if 'BTL-4' in r['name']:
        print(f"\nBTL-4 position: #{i+1} (sorted by tau2)")
        print(f"  LCB: {r['lcb']*100:.1f}%")
        print(f"  tau2: {r['tau2']}")
        print(f"  tok/s: {r['tps']}")
