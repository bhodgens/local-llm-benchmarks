#!/usr/bin/env python3
"""
Merge all benchmark results into a single JSON and generate HTML report.
"""
import json, glob, os, re
from datetime import datetime

PROGRESS_FILE = '/tmp/coding-bench/progress.json'
LCB_OUTPUT_DIR = '/home/caimlas/git/LiveCodeBench/output'
BENCH_RESULTS = '/home/caimlas/llm-benchmarks/bench_results.json'
REPORT_OUT = '/tmp/coding-bench/results/report.html'

# Load progress
with open(PROGRESS_FILE) as f:
    progress = json.load(f)

# Merge LCB scores from eval files (authoritative)
def find_lcb_score(model_name):
    safe = model_name.replace(' ', '_')
    # Try all possible dir name patterns
    parts = safe.split('_')
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

# Speed benchmark data
speed_data = {}
if os.path.exists(BENCH_RESULTS):
    with open(BENCH_RESULTS) as f:
        speed_data = json.load(f)

# Build unified results
models = []
for m in progress['models']:
    name = m['name']
    
    # LCB
    lcb_score = None
    if 'livecodebench' in m and m['livecodebench'].get('pass_at_1') is not None:
        lcb_score = m['livecodebench']['pass_at_1']
    else:
        lcb_score = find_lcb_score(name)
    
    # HumanEval
    he_score = m.get('humaneval', {}).get('pass_at_1')
    
    # tau2
    tau2_reward = m.get('tau2', {}).get('reward')
    tau2_time = m.get('tau2', {}).get('wall_time_s', 0)
    
    # Speed (from bench_results)
    decode_tps = m.get('decode_tps')
    
    # Failures
    failures = m.get('failures', [])
    
    # Determine model category
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
        'decode_tps': decode_tps,
        'failures': failures,
    })

# Sort by tau2 reward (primary benchmark)
models.sort(key=lambda x: -(x['tau2'] or -1))

# Save merged JSON
with open('/tmp/coding-bench/results/final_results.json', 'w') as f:
    json.dump(models, f, indent=2)

# Generate HTML
def pct(v):
    if v is None:
        return '<span class="na">N/A</span>'
    return '%.1f%%' % (v * 100)

def tau2_fmt(v):
    if v is None:
        return '<span class="na">N/A</span>'
    return '%.4f' % v

def tps_fmt(v):
    if v is None:
        return '<span class="na">?</span>'
    return '%.0f' % v

def bar_cell(value, max_val, fmt_func, color='#4a9'):
    if value is None:
        return '<td class="center">N/A</td>'
    pct_width = min(100, (value / max_val * 100)) if max_val > 0 else 0
    return f'<td class="center"><div class="bar-container"><div class="bar-fill" style="width:{pct_width:.0f}%;background:{color}"></div><span class="bar-label">{fmt_func(value)}</span></div></td>'

lcb_max = max([m['livecodebench'] for m in models if m['livecodebench']] or [1])
tau2_max = max([m['tau2'] for m in models if m['tau2']] or [1])

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LLM Benchmark Results - 3060 (RTX 3060 12GB)</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0d1117; color: #c9d1d9; padding: 20px; }}
h1 {{ color: #58a6ff; margin-bottom: 5px; font-size: 1.8em; }}
h2 {{ color: #8b949e; margin: 25px 0 10px; font-size: 1.3em; border-bottom: 1px solid #30363d; padding-bottom: 5px; }}
.subtitle {{ color: #8b949e; margin-bottom: 20px; font-size: 0.9em; }}
table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
th {{ background: #161b22; color: #58a6ff; padding: 10px 8px; text-align: left; font-size: 0.85em; text-transform: uppercase; border-bottom: 2px solid #30363d; }}
td {{ padding: 8px; border-bottom: 1px solid #21262d; font-size: 0.9em; }}
tr:hover {{ background: #161b22; }}
.center {{ text-align: center; }}
.na {{ color: #484f58; }}
.bar-container {{ position: relative; height: 22px; background: #21262d; border-radius: 3px; overflow: hidden; min-width: 80px; }}
.bar-fill {{ position: absolute; height: 100%; opacity: 0.3; border-radius: 3px; }}
.bar-label {{ position: relative; z-index: 1; line-height: 22px; padding-left: 8px; font-size: 0.85em; font-weight: 600; }}
.badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em; font-weight: 600; }}
.badge-moe {{ background: #1f6feb22; color: #58a6ff; border: 1px solid #1f6feb44; }}
.badge-8b {{ background: #3fb95022; color: #3fb950; border: 1px solid #3fb95044; }}
.badge-12b {{ background: #d2992222; color: #d29922; border: 1px solid #d2992244; }}
.badge-27b {{ background: #f8514922; color: #f85149; border: 1px solid #f8514944; }}
.badge-other {{ background: #8b949e22; color: #8b949e; border: 1px solid #8b949e44; }}
.cat-cell {{ white-space: nowrap; }}
.note {{ background: #161b22; border-left: 3px solid #58a6ff; padding: 10px 15px; margin: 15px 0; font-size: 0.85em; color: #8b949e; }}
.failed {{ color: #f85149; }}
.summary-card {{ background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 15px; margin: 10px 0; }}
.summary-card h3 {{ color: #58a6ff; margin-bottom: 8px; }}
</style>
</head>
<body>

<h1>LLM Benchmark Results</h1>
<div class="subtitle">
  Hardware: RTX 3060 12GB + Tesla V100 32GB (user sim) | 
  Benchmark date: {datetime.now().strftime('%Y-%m-%d')} |
  15 models tested
</div>

<div class="note">
  <strong>Benchmarks:</strong> HumanEval (code completion, 164 problems) |
  LiveCodeBench (competitive programming, 75 problems, pass@1, thinking disabled) |
  tau2-bench (agentic tool use, airline domain, 15 tasks)
</div>

<h2>Full Results Table</h2>
<table>
<tr>
  <th>#</th>
  <th>Model</th>
  <th>Type</th>
  <th>HumanEval</th>
  <th>LiveCodeBench</th>
  <th>tau2-bench</th>
  <th>tau2 Time</th>
</tr>
"""

for i, m in enumerate(models, 1):
    cat_badge = {
        'MoE 35B': 'badge-moe', '8B': 'badge-8b',
        '12-14B': 'badge-12b', '27B Dense': 'badge-27b', 'Other': 'badge-other'
    }.get(m['category'], 'badge-other')
    
    fail_marker = ' <span class="failed">(failed)</span>' if m['failures'] and not m['tau2'] else ''
    
    html += f"""<tr>
  <td class="center">{i}</td>
  <td>{m['name']}{fail_marker}</td>
  <td class="cat-cell"><span class="badge {cat_badge}">{m['category']}</span></td>
  {bar_cell(m['human_eval'], 1.0, pct, '#238636')}
  {bar_cell(m['livecodebench'], lcb_max, pct, '#1f6feb')}
  {bar_cell(m['tau2'], tau2_max, tau2_fmt, '#d29922')}
  <td class="center">{str(m['tau2_time_min']) + ' min' if m['tau2_time_min'] else '<span class="na">-</span>'}</td>
</tr>"""

html += """</table>

<h2>Key Findings</h2>

<div class="summary-card">
  <h3>Code Generation vs Agentic Tool Use</h3>
  <p>The most striking finding is the divergence between code generation and agentic capability:</p>
  <ul style="margin: 10px 0 0 20px; color: #8b949e; font-size: 0.9em;">
    <li><strong style="color:#3fb950">gemma-4-12B-it-QAT</strong>: LCB #1 (90.7%) but tau2 dead last (0.08) - exceptional at writing code, terrible at tool use</li>
    <li><strong style="color:#3fb950">LFM2.5-8B-A1B-Coder-v2</strong>: LCB mediocre (45.3%) but tau2 #1 (0.53) - best agent, average coder</li>
    <li><strong style="color:#3fb950">Qwen2.5-Coder-14B</strong>: LCB #3 (80.0%) but tau2 near bottom (0.13) - same pattern</li>
    <li>Models that excel at standalone code generation often struggle with conversational tool use, and vice versa</li>
  </ul>
</div>

<div class="summary-card">
  <h3>Benchmark Rankings</h3>
  <table style="margin-top: 10px;">
    <tr><th>Benchmark</th><th>Winner</th><th>Score</th></tr>"""

# Top performers per benchmark
he_top = sorted([m for m in models if m['human_eval']], key=lambda x: -x['human_eval'])
if he_top:
    html += f"<tr><td>HumanEval</td><td>{he_top[0]['name']}</td><td>{pct(he_top[0]['human_eval'])}</td></tr>"

lcb_top = sorted([m for m in models if m['livecodebench']], key=lambda x: -x['livecodebench'])
if lcb_top:
    html += f"<tr><td>LiveCodeBench</td><td>{lcb_top[0]['name']}</td><td>{pct(lcb_top[0]['livecodebench'])}</td></tr>"

tau2_top = sorted([m for m in models if m['tau2']], key=lambda x: -x['tau2'])
if tau2_top:
    html += f"<tr><td>tau2-bench</td><td>{tau2_top[0]['name']}</td><td>{tau2_fmt(tau2_top[0]['tau2'])}</td></tr>"

html += """  </table>
</div>

<div class="summary-card">
  <h3>Model-Specific Notes</h3>
  <ul style="margin: 10px 0 0 20px; color: #8b949e; font-size: 0.9em;">
    <li><strong>Gemma-4 family</strong> dominates code generation: 4 of top 5 LCB scores</li>
    <li><strong>LFM2.5-8B</strong> is the best small agentic model (tau2 0.53 at only 31 min)</li>
    <li><strong>Qwopus3.6-35B MoE</strong> is the most balanced: LCB 80% + tau2 0.40</li>
    <li><strong>RavenX-OpenFable-Holo3</strong> is the best MoE on LCB (85.3%)</li>
    <li><strong>DeepSeek-R1</strong> scored poorly on LCB (17.3%) due to thinking tokens not being fully suppressible</li>
    <li><strong>Qwen3.6-Abliterated-Heretic</strong> is consistently the weakest across all benchmarks</li>
    <li><strong>DeepSeek-Coder-V2-Lite</strong> is surprisingly strong on tau2 (0.50) despite being a coding model</li>
  </ul>
</div>

<h2>Failures & Issues</h2>
<table>
  <tr><th>Model</th><th>Benchmark</th><th>Error</th></tr>
"""

# Collect all failures
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
  - HumanEval: Only Qwen2.5-Coder completed successfully (code completion format). Other models need chat-mode rerun.<br>
  - LiveCodeBench: 75 problems from latest release, pass@1, temperature=0.0, thinking disabled via enable_thinking=false<br>
  - tau2-bench: Airline domain, 15 tasks, max 30 steps per task, 300s timeout, agent on 3060, user sim on V100<br>
  - Thinking-on 32K LCB run failed (timeouts at 27 tok/s for MoE models)<br>
  - DeepSeek-R1 models ran with reduced context (16-32K) to fit VRAM<br>
  - IQ3_K_R4 model permanently failed (unsupported ggml type 138 quantization format)<br>
  - Speed benchmarks at 8K context, flash attention on, q8_0 KV cache
</div>

</body>
</html>"""

with open(REPORT_OUT, 'w') as f:
    f.write(html)

print("HTML report generated at:", REPORT_OUT)
print("Final JSON at: /tmp/coding-bench/results/final_results.json")
print("\nModels with complete data:", len([m for m in models if m['livecodebench'] and m['tau2']]))
print("Models with partial data:", len([m for m in models if not m['livecodebench'] or not m['tau2']]))
