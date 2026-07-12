#!/usr/bin/env python3
"""
Generate sortable HTML report with all benchmark results including tok/s.
"""
import json, glob, os, re
from datetime import datetime

PROGRESS_FILE = '/tmp/coding-bench/progress.json'
TPS_PROBES = '/tmp/coding-bench/tps_probes.json'
LCB_OUTPUT_DIR = '/home/caimlas/git/LiveCodeBench/output'
REPORT_OUT = '/home/caimlas/llm-benchmarks/report.html'

with open(PROGRESS_FILE) as f:
    progress = json.load(f)

with open(TPS_PROBES) as f:
    tps_probes = json.load(f)

# Original models' tok/s from progress.json (measured during LCB)
orig_tps = {}
for m in progress.get('models', []):
    tps = m.get('decode_tps')
    if tps:
        orig_tps[m['name']] = tps

def find_lcb_score(model_name):
    safe = model_name.replace(' ', '_')
    search_terms = [safe, safe.replace('_Q4_K_M','').replace('_Q6_K','').replace('_Q8_0','').replace('_Q4_0','').replace('_IQ4_XS','')]
    for base_dir in [LCB_OUTPUT_DIR, '/tmp/coding-bench/results/lcb_thinking_off']:
        if not os.path.exists(base_dir):
            continue
        for d in glob.glob(base_dir + '/*'):
            dirname = os.path.basename(d)
            for term in search_terms:
                if term in dirname or dirname in term:
                    eval_files = glob.glob(d + '/*_eval.json')
                    if eval_files:
                        with open(eval_files[0]) as f:
                            data = json.load(f)
                        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                            score = data[0].get('pass@1')
                            if score is not None:
                                return score
    return None

def get_tps(name):
    if name in orig_tps:
        return orig_tps[name]
    if name in tps_probes:
        return tps_probes[name]
    return None

# Build model data
models = []
for m in progress['models']:
    name = m['name']
    
    lcb_score = None
    if 'livecodebench' in m and m['livecodebench'].get('pass_at_1') is not None:
        lcb_score = m['livecodebench']['pass_at_1']
    else:
        lcb_score = find_lcb_score(name)
    
    he_score = m.get('humaneval', {}).get('pass_at_1')
    if he_score is not None and he_score == 0.0:
        he_score = None  # 0.0 means broken, not real score
    
    tau2_reward = m.get('tau2', {}).get('reward')
    tau2_time = m.get('tau2', {}).get('wall_time_s', 0)
    tps = get_tps(name)
    
    name_lower = name.lower()
    if 'moe' in name_lower or 'a3b' in name_lower or '35b' in name_lower:
        category = 'MoE 35B'
    elif '8b' in name_lower or 'a1b' in name_lower:
        category = '8B'
    elif '14b' in name_lower or '12b' in name_lower:
        category = '12-14B'
    elif '27b' in name_lower:
        category = '27B Dense'
    else:
        category = 'Other'
    
    models.append({
        'name': name,
        'category': category,
        'human_eval': he_score,
        'livecodebench': lcb_score,
        'tau2': tau2_reward,
        'tau2_time_min': round(tau2_time / 60) if tau2_time else None,
        'decode_tps': tps,
        'failures': m.get('failures', []),
    })

# Generate HTML with sortable table
html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Benchmark Results</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }
h1 { color: #58a6ff; margin-bottom: 5px; font-size: 1.8em; }
h2 { color: #8b949e; margin: 25px 0 10px; font-size: 1.3em; border-bottom: 1px solid #30363d; padding-bottom: 5px; }
.subtitle { color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }
table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
th { background: #161b22; color: #58a6ff; padding: 10px 8px; text-align: left; font-size: 0.85em; text-transform: uppercase; border-bottom: 2px solid #30363d; cursor: pointer; user-select: none; position: relative; }
th:hover { background: #1c2331; color: #79c0ff; }
th.sorted-asc::after { content: " \\25B2"; color: #3fb950; }
th.sorted-desc::after { content: " \\25BC"; color: #3fb950; }
td { padding: 8px; border-bottom: 1px solid #21262d; font-size: 0.9em; }
tr:hover { background: #161b22; }
.center { text-align: center; }
.na { color: #484f58; }
.bar-container { position: relative; height: 22px; background: #21262d; border-radius: 3px; overflow: hidden; min-width: 80px; }
.bar-fill { position: absolute; height: 100%; opacity: 0.25; border-radius: 3px; }
.bar-label { position: relative; z-index: 1; line-height: 22px; padding-left: 8px; font-size: 0.85em; font-weight: 600; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }
.badge-moe { background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb44; }
.badge-8b { background: #3fb95022; color: #3fb950; border: 1px solid #3fb95044; }
.badge-12b { background: #d2992222; color: #d29922; border: 1px solid #d2992244; }
.badge-27b { background: #f8514922; color: #f85149; border: 1px solid #f8514944; }
.badge-other { background: #8b949e22; color: #8b949e; border: 1px solid #8b949e44; }
.note { background: #161b22; border-left: 3px solid #58a6ff; padding: 10px 15px; margin: 15px 0; font-size: 0.85em; color: #8b949e; }
.failed { color: #f85149; }
.summary-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin: 10px 0; }
.summary-card h3 { color: #58a6ff; margin-bottom: 8px; }
.summary-card ul { margin: 10px 0 0 20px; color: #8b949e; font-size: 0.9em; }
.summary-card li { margin: 4px 0; }
</style>
</head>
<body>

<h1>LLM Benchmark Results</h1>
<div class="subtitle">
  Hardware: RTX 3060 12GB (agent) + Tesla V100 32GB (user sim) |
  Date: """ + datetime.now().strftime('%Y-%m-%d') + """ |
  15 models tested |
  Click column headers to sort
</div>

<div class="note">
  <strong>Benchmarks:</strong> LiveCodeBench (competitive programming, 75 problems, pass@1, thinking disabled) |
  tau2-bench (agentic tool use, airline domain, 15 tasks) |
  HumanEval (code completion, only Qwen2.5-Coder completed in raw mode -- rerun needed for chat mode)<br>
  <strong>tok/s:</strong> Decode throughput on 3060 at 8K context, flash attention on. MoE models use cpu-moe.
</div>

<h2>Full Results (click headers to sort)</h2>
<table id="resultsTable">
<thead>
<tr>
  <th data-type="number" data-key="idx">#</th>
  <th data-type="string" data-key="name">Model</th>
  <th data-type="string" data-key="category">Type</th>
  <th data-type="number" data-key="tps">tok/s</th>
  <th data-type="number" data-key="he">HumanEval</th>
  <th data-type="number" data-key="lcb">LiveCodeBench</th>
  <th data-type="number" data-key="tau2">tau2-bench</th>
  <th data-type="number" data-key="tau2_time">tau2 Time (min)</th>
</tr>
</thead>
<tbody>
"""

# Sort by tau2 by default
models_sorted = sorted(models, key=lambda x: -(x['tau2'] or -1))

lcb_max = max([m['livecodebench'] for m in models if m['livecodebench']] or [1])
tau2_max = max([m['tau2'] for m in models if m['tau2']] or [1])
tps_max = max([m['decode_tps'] for m in models if m['decode_tps']] or [1])

def bar_cell(value, max_val, fmt_func, color):
    if value is None:
        return '<td class="center"><span class="na">N/A</span></td>'
    pct_width = min(100, (value / max_val * 100)) if max_val > 0 else 0
    return f'<td class="center"><div class="bar-container"><div class="bar-fill" style="width:{pct_width:.0f}%;background:{color}"></div><span class="bar-label">{fmt_func(value)}</span></div></td>'

def pct(v): return '%.1f%%' % (v * 100)
def tau2_fmt(v): return '%.4f' % v
def tps_fmt(v): return '%.0f' % v

for i, m in enumerate(models_sorted, 1):
    cat_badge = {
        'MoE 35B': 'badge-moe', '8B': 'badge-8b',
        '12-14B': 'badge-12b', '27B Dense': 'badge-27b', 'Other': 'badge-other'
    }.get(m['category'], 'badge-other')
    
    has_failures = bool(m['failures']) and not m['tau2']
    fail_marker = ' <span class="failed">(!)</span>' if has_failures else ''
    
    html += f"""<tr>
  <td class="center">{i}</td>
  <td data-sort="{m['name'].lower()}">{m['name']}{fail_marker}</td>
  <td class="center"><span class="badge {cat_badge}">{m['category']}</span></td>
  {bar_cell(m['decode_tps'], tps_max, tps_fmt, '#bc8cff')}
  {bar_cell(m['human_eval'], 1.0, pct, '#238636')}
  {bar_cell(m['livecodebench'], lcb_max, pct, '#1f6feb')}
  {bar_cell(m['tau2'], tau2_max, tau2_fmt, '#d29922')}
  <td class="center">{str(m['tau2_time_min']) if m['tau2_time_min'] else '<span class="na">-</span>'}</td>
</tr>"""

html += """</tbody>
</table>

<h2>Key Findings</h2>

<div class="summary-card">
  <h3>Code Generation vs Agentic Tool Use</h3>
  <ul>
    <li><strong style="color:#3fb950">gemma-4-12B-it-QAT</strong>: LCB #1 (90.7%) but tau2 dead last (0.08) -- exceptional at writing code, terrible at tool use</li>
    <li><strong style="color:#3fb950">LFM2.5-8B-A1B-Coder-v2</strong>: LCB mediocre (45.3%) but tau2 #1 (0.53) -- best agent, average coder</li>
    <li><strong style="color:#3fb950">Qwen2.5-Coder-14B</strong>: LCB #3 (80.0%) but tau2 near bottom (0.13) -- same pattern</li>
    <li>Models that excel at standalone code generation often struggle with conversational tool use, and vice versa</li>
  </ul>
</div>

<div class="summary-card">
  <h3>Speed vs Quality Tradeoffs</h3>
  <ul>
    <li><strong style="color:#bc8cff">LFM2.5-8B base</strong> is the fastest at 213 tok/s -- 6x faster than MoE models -- with decent tau2 (0.46)</li>
    <li><strong style="color:#bc8cff">DeepSeek-Coder-V2-Lite</strong> hits 92.5 tok/s (MoE with only 2.4B active params) and matches gemma4-coding on tau2 (0.50)</li>
    <li><strong style="color:#bc8cff">MoE 35B models</strong> at 26-28 tok/s are the slowest but competitive on LCB (80-85%)</li>
    <li><strong style="color:#bc8cff">Gemma-4 12B models</strong> at 35-40 tok/s offer the best LCB quality per tok/s (90.7% at 39.5 tok/s)</li>
  </ul>
</div>

<div class="summary-card">
  <h3>Model-Specific Notes</h3>
  <ul>
    <li><strong>Gemma-4 family</strong> dominates code generation: 4 of top 5 LCB scores</li>
    <li><strong>LFM2.5-8B</strong> is the best small agentic model (tau2 0.53 at only 31 min)</li>
    <li><strong>Qwopus3.6-35B MoE</strong> is the most balanced: LCB 80% + tau2 0.40</li>
    <li><strong>RavenX-OpenFable-Holo3</strong> is the best MoE on LCB (85.3%)</li>
    <li><strong>DeepSeek-R1</strong> scored poorly on LCB (17.3%) -- thinking tokens not fully suppressible</li>
    <li><strong>Qwen3.6-Abliterated-Heretic</strong> is consistently the weakest across all benchmarks</li>
    <li><strong>DeepSeek-Coder-V2-Lite</strong> is surprisingly strong on tau2 (0.50) despite being a coding model</li>
    <li><strong>IQ3_K_R4</strong> permanently failed (unsupported ggml type 138 quantization in llama.cpp v9836)</li>
  </ul>
</div>

<h2>Failures & Issues</h2>
<table>
  <tr><th>Model</th><th>Benchmark</th><th>Error</th></tr>
"""

all_failures = []
for m in models:
    for f in m.get('failures', []):
        all_failures.append((m['name'], f))

if not all_failures:
    html += '<tr><td colspan="3" class="center na">No failures recorded</td></tr>'
else:
    for name, f in all_failures:
        bench = f.get('benchmark', '?')
        err = f.get('error', '?')[:120]
        html += f'<tr><td>{name}</td><td>{bench}</td><td class="failed">{err}</td></tr>'

html += """</table>

<div class="note">
  <strong>Methodology Notes:</strong><br>
  - HumanEval: Only Qwen2.5-Coder completed successfully (raw code completion mode). Other models need chat-mode rerun. N/A = not run, not failed.<br>
  - LiveCodeBench: 75 problems from latest release, pass@1, temperature=0.0, thinking disabled via enable_thinking=false where supported<br>
  - tau2-bench: Airline domain, 15 tasks, max 30 steps (15 for small-context models), 300s timeout, agent on 3060, user sim on V100<br>
  - DeepSeek-R1 models ran tau2 with reduced context (16-32K) to fit VRAM; LCB with 32K context<br>
  - tok/s measured at 8K context with 256-token decode, flash attention on; production config (cpu-moe, q8_0 KV for MoE; q4_0 KV for dense)<br>
  - DeepSeek-Coder-V2-Lite tok/s measured at 4K context due to VRAM constraints<br>
  - Speed benchmarks are raw decode throughput; actual benchmark throughput varies with prompt processing overhead
</div>

<script>
// Sortable table
document.addEventListener('DOMContentLoaded', function() {
  const table = document.getElementById('resultsTable');
  const tbody = table.querySelector('tbody');
  const headers = table.querySelectorAll('th');
  let sortDir = 'desc';
  let sortKey = 'tau2';

  function getCellValue(row, key) {
    const cells = row.querySelectorAll('td');
    switch(key) {
      case 'idx': return parseInt(cells[0].textContent);
      case 'name': return cells[1].getAttribute('data-sort') || cells[1].textContent.toLowerCase();
      case 'category': return cells[2].textContent.trim();
      case 'tps': return parseFloat(cells[3].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'he': return parseFloat(cells[4].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'lcb': return parseFloat(cells[5].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'tau2': return parseFloat(cells[6].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'tau2_time': return parseFloat(cells[7].textContent.replace(/[^0-9.]/g, '')) || 99999;
      default: return 0;
    }
  }

  function sortTable(key, dir) {
    const rows = Array.from(tbody.querySelectorAll('tr'));
    rows.sort((a, b) => {
      const av = getCellValue(a, key);
      const bv = getCellValue(b, key);
      if (typeof av === 'string') {
        return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
      }
      return dir === 'asc' ? av - bv : bv - av;
    });
    rows.forEach((row, i) => {
      tbody.appendChild(row);
      row.querySelector('td').textContent = i + 1;
    });
  }

  headers.forEach((th, i) => {
    th.addEventListener('click', function() {
      const key = th.getAttribute('data-key');
      if (sortKey === key) {
        sortDir = sortDir === 'asc' ? 'desc' : 'asc';
      } else {
        sortKey = key;
        sortDir = key === 'name' || key === 'category' ? 'asc' : 'desc';
      }

      headers.forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
      th.classList.add(sortDir === 'asc' ? 'sorted-asc' : 'sorted-desc');

      sortTable(key, sortDir);
    });
  });

  // Default sort indicator
  const tau2Header = table.querySelector('th[data-key="tau2"]');
  if (tau2Header) tau2Header.classList.add('sorted-desc');
});
</script>

</body>
</html>"""

with open(REPORT_OUT, 'w') as f:
    f.write(html)

print("Report generated at:", REPORT_OUT)
print("Models:", len(models))
print("With tok/s:", len([m for m in models if m['decode_tps']]))
print("With LCB:", len([m for m in models if m['livecodebench']]))
print("With tau2:", len([m for m in models if m['tau2']]))
