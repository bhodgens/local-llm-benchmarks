#!/usr/bin/env python3
"""
Generate sortable HTML report with all benchmark results including tok/s.
Model names are clickable to show a detail modal with settings, VRAM,
wall times, failures, and errors.
"""
import json, glob, os, re, html
from datetime import datetime

PROGRESS_FILE = '/tmp/coding-bench/progress.json'
TPS_PROBES = '/tmp/coding-bench/tps_probes.json'
LCB_OUTPUT_DIR = '/home/caimlas/git/LiveCodeBench/output'
REPORT_OUT = '/home/caimlas/llm-benchmarks/report.html'

with open(PROGRESS_FILE) as f:
    progress = json.load(f)

try:
    with open(TPS_PROBES) as f:
        tps_probes = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    tps_probes = {}

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


def build_detail(m):
    """Extract all available detail fields for a model entry."""
    detail = {}

    # Settings
    settings = {}
    if m.get('binary'): settings['Binary'] = m['binary']
    if m.get('thinking') is not None: settings['Thinking model'] = str(m['thinking'])
    if m.get('dspark') is not None: settings['DSpark'] = str(m['dspark'])
    if m.get('mtp_enabled') is not None: settings['MTP'] = str(m['mtp_enabled'])
    if m.get('quant'): settings['Quant'] = m['quant']
    if m.get('bpw'): settings['Bits/weight'] = m['bpw']
    if m.get('file'): settings['File'] = m['file']
    if settings: detail['settings'] = settings

    # VRAM / context
    vram = {}
    for k, label in [('vram_262k_mb','262K ctx'), ('vram_200k_mb','200K ctx'),
                      ('vram_131k_mb','131K ctx'), ('vram_100k_mb','100K ctx')]:
        if m.get(k): vram[label] = f"{m[k]} MiB ({m[k]/1024:.1f} GB)"
    if m.get('max_ctx_3060'): vram['Max ctx (3060)'] = str(m['max_ctx_3060'])
    if m.get('max_ctx_3060_nodraft'): vram['Max ctx no-draft'] = str(m['max_ctx_3060_nodraft'])
    if m.get('max_ctx_3060_dspark'): vram['Max ctx dspark'] = str(m['max_ctx_3060_dspark'])
    if vram: detail['vram'] = vram

    # Benchmark details with wall times
    benchmarks = {}
    for bench_key, bench_label in [
        ('livecodebench', 'LiveCodeBench'),
        ('livecodebench_3060', 'LCB (3060)'),
        ('livecodebench_3060_dspark', 'LCB (3060+dspark)'),
        ('tau2', 'tau2-bench'),
        ('tau2_3060', 'tau2 (3060)'),
        ('tau2_3060_dspark', 'tau2 (3060+dspark)'),
        ('humaneval', 'HumanEval'),
        ('humaneval_chat', 'HumanEval (chat)'),
        ('aider_diff', 'Aider (diff)'),
    ]:
        if bench_key in m and isinstance(m[bench_key], dict):
            b = {}
            for sub_k, sub_v in m[bench_key].items():
                if sub_k == 'wall_time_s' and sub_v:
                    b['Wall time'] = f"{sub_v:.0f}s ({sub_v/60:.0f} min)"
                elif sub_k == 'pass_at_1' and sub_v is not None:
                    b['pass@1'] = f"{sub_v*100:.1f}%"
                elif sub_k == 'pass_rate' and sub_v is not None:
                    b['pass rate'] = f"{sub_v*100:.0f}%"
                elif sub_k == 'reward' and sub_v is not None:
                    b['reward'] = f"{sub_v:.4f}"
                elif sub_k == 'task_pass_rate' and sub_v is not None:
                    b['task pass'] = f"{sub_v:.1%}"
                elif sub_k == 'passed' and sub_v is not None:
                    b['passed'] = sub_v
                elif sub_k == 'total' and sub_v is not None:
                    b['total'] = sub_v
                elif sub_k == 'exit_code' and sub_v is not None and sub_v != 0:
                    b['exit code'] = sub_v
                elif sub_k == 'error':
                    b['error'] = sub_v
            if b: benchmarks[bench_label] = b
    if benchmarks: detail['benchmarks'] = benchmarks

    # Throughput
    throughput = {}
    if m.get('decode_tps'): throughput['Decode tok/s'] = m['decode_tps']
    if m.get('decode_tps_3060'): throughput['Decode tok/s (3060)'] = m['decode_tps_3060']
    if m.get('decode_tps_3060_dspark'): throughput['Decode tok/s (3060+dspark)'] = m['decode_tps_3060_dspark']
    if m.get('prompt_tps'): throughput['Prompt tok/s'] = m['prompt_tps']
    if throughput: detail['throughput'] = throughput

    # Errors
    if m.get('error'): detail['hard_error'] = str(m['error'])[:500]
    if m.get('lcb_error'): detail['lcb_error'] = str(m['lcb_error'])[:500]
    if m.get('tau2_error'): detail['tau2_error'] = str(m['tau2_error'])[:500]

    # Failures / adaptive retry
    if m.get('failures'):
        detail['failures'] = []
        for f in m['failures']:
            entry = {
                'benchmark': f.get('benchmark', ''),
                'attempt': str(f.get('attempt', '')),
                'error': str(f.get('error', ''))[:400],
                'timestamp': f.get('timestamp', '')[:19],
            }
            s = f.get('settings', '')
            if isinstance(s, list):
                entry['settings'] = ' '.join(str(x) for x in s)
            elif s:
                entry['settings'] = str(s)
            detail['failures'].append(entry)

    # Timing
    timing = {}
    if m.get('start_time'): timing['Started'] = m['start_time'][:19]
    if m.get('end_time'): timing['Finished'] = m['end_time'][:19]
    if m.get('start_time') and m.get('end_time'):
        try:
            s = datetime.fromisoformat(m['start_time'])
            e = datetime.fromisoformat(m['end_time'])
            dur = (e - s).total_seconds()
            timing['Duration'] = f"{dur/60:.0f} min" if dur > 60 else f"{dur:.0f}s"
        except:
            pass
    if timing: detail['timing'] = timing

    if m.get('status'): detail['status'] = m['status']

    return detail


# Build model data
models = []
for m in progress['models']:
    name = m['name']

    lcb_score = None
    if 'livecodebench' in m and m['livecodebench'].get('pass_at_1') is not None:
        lcb_score = m['livecodebench']['pass_at_1']
    else:
        lcb_score = find_lcb_score(name)
    # Also check 3060-specific LCB
    if lcb_score is None and 'livecodebench_3060' in m:
        lcb_score = m['livecodebench_3060'].get('pass_at_1')

    he_score = m.get('humaneval', {}).get('pass_at_1')
    if he_score is not None and he_score == 0.0:
        he_score = None

    tau2_reward = m.get('tau2', {}).get('reward')
    tau2_time = m.get('tau2', {}).get('wall_time_s', 0)
    if tau2_reward is None and 'tau2_3060' in m:
        tau2_reward = m['tau2_3060'].get('reward')
        tau2_time = m['tau2_3060'].get('wall_time_s', 0)

    tps = get_tps(name)
    if tps is None:
        tps = m.get('decode_tps_3060')
    if tps is None:
        tps = m.get('decode_tps_3060_dspark')

    name_lower = name.lower()
    if 'bonsai' in name_lower:
        category = '27B Ternary'
    elif 'a4b' in name_lower or '26b' in name_lower:
        category = '26B MoE'
    elif 'moe' in name_lower or 'a3b' in name_lower or '35b' in name_lower:
        category = 'MoE 35B'
    elif '8b' in name_lower or 'a1b' in name_lower:
        category = '8B'
    elif '14b' in name_lower or '12b' in name_lower:
        category = '12-14B'
    elif '9b' in name_lower:
        category = '9B'
    elif '27b' in name_lower:
        category = '27B Dense'
    else:
        category = 'Other'

    detail = build_detail(m)
    detail['name'] = name
    detail['category'] = category
    detail['gpu'] = m.get('gpu', '3060')

    models.append({
        'name': name,
        'category': category,
        'gpu': m.get('gpu', '3060'),
        'human_eval': he_score,
        'livecodebench': lcb_score,
        'tau2': tau2_reward,
        'tau2_time_min': round(tau2_time / 60) if tau2_time else None,
        'decode_tps': tps,
        'failures': m.get('failures', []),
        'detail': detail,
    })

# Check which models have extra detail data
has_detail = sum(1 for m in models if len(m['detail']) > 4)  # >4 means more than name/cat/gpu/failures

# Embed full detail as JSON
detail_json = json.dumps([m['detail'] for m in models], ensure_ascii=False)

# Generate HTML
html_doc = """<!DOCTYPE html>
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

/* Model name click + info icon */
.model-name { cursor: pointer; color: #c9d1d9; text-decoration: none; }
.model-name:hover { color: #58a6ff; }
.info-icon { display: inline-block; width: 16px; height: 16px; line-height: 16px; text-align: center;
  border-radius: 50%; background: #21262d; color: #8b949e; font-size: 0.7em; margin-left: 6px;
  cursor: pointer; vertical-align: middle; transition: all 0.15s; }
.info-icon:hover { background: #1f6feb; color: #fff; }
.has-detail .info-icon { color: #58a6ff; }

/* Modal */
.modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
  background: rgba(0,0,0,0.7); z-index: 1000; justify-content: center; align-items: flex-start; padding-top: 40px; }
.modal-overlay.active { display: flex; }
.modal { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 0;
  max-width: 700px; width: 90%; max-height: 85vh; overflow-y: auto; box-shadow: 0 8px 32px rgba(0,0,0,0.5); }
.modal-header { padding: 18px 20px 12px; border-bottom: 1px solid #30363d; display: flex; justify-content: space-between; align-items: center; }
.modal-header h3 { color: #58a6ff; font-size: 1.1em; }
.modal-close { background: none; border: none; color: #8b949e; font-size: 1.4em; cursor: pointer; padding: 0 5px; line-height: 1; }
.modal-close:hover { color: #f85149; }
.modal-body { padding: 16px 20px; }
.modal-section { margin-bottom: 16px; }
.modal-section-title { color: #8b949e; font-size: 0.75em; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.modal-kv { display: grid; grid-template-columns: 1fr 2fr; gap: 4px 12px; font-size: 0.88em; }
.modal-kv .k { color: #8b949e; }
.modal-kv .v { color: #c9d1d9; word-break: break-word; }
.modal-kv .v.mono { font-family: 'SF Mono', 'Cascadia Code', Consolas, monospace; font-size: 0.82em; }
.modal-error { background: #f8514911; border: 1px solid #f8514933; border-radius: 6px; padding: 8px 12px;
  font-size: 0.82em; color: #f85149; margin: 4px 0; font-family: monospace; white-space: pre-wrap; word-break: break-word; }
.modal-fail-entry { background: #161b22; border-left: 3px solid #f85149; padding: 8px 12px; margin: 6px 0; font-size: 0.82em; }
.modal-fail-entry .fail-header { color: #f85149; font-weight: 600; margin-bottom: 4px; }
.modal-fail-entry .fail-settings { color: #8b949e; font-family: monospace; font-size: 0.9em; margin-top: 4px; }
.modal-bench { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 8px 12px; margin: 6px 0; }
.modal-bench .bench-name { color: #58a6ff; font-weight: 600; font-size: 0.85em; margin-bottom: 4px; }
.modal-bench .bench-kv { display: grid; grid-template-columns: 1fr 2fr; gap: 2px 12px; font-size: 0.82em; }
.modal-bench .bench-kv .k { color: #8b949e; }
.modal-bench .bench-kv .v { color: #c9d1d9; }
</style>
</head>
<body>

<h1>LLM Benchmark Results</h1>
<div class="subtitle">
  Hardware: RTX 3060 12GB + Tesla V100 32GB |
  Date: """ + datetime.now().strftime('%Y-%m-%d') + """ |
  """ + str(len(models)) + """ models tested |
  Click column headers to sort |
  Click model name or (i) icon for details
</div>

<div class="note">
  <strong>Benchmarks:</strong> LiveCodeBench (competitive programming, 75 problems, pass@1, thinking disabled) |
  tau2-bench (agentic tool use, airline domain, 15 tasks) |
  HumanEval (code completion, only Qwen2.5-Coder completed in raw mode -- rerun needed for chat mode)<br>
  <strong>tok/s:</strong> Decode throughput on 3060 at 8K context, flash attention on. MoE models use cpu-moe.<br>
  <strong>Detail view:</strong> Click any model name to see settings, VRAM usage, wall times, errors, and retry history.
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
        '12-14B': 'badge-12b', '27B Dense': 'badge-27b', '27B Ternary': 'badge-27b',
        '26B MoE': 'badge-moe', '9B': 'badge-8b', 'Other': 'badge-other'
    }.get(m['category'], 'badge-other')

    has_failures = bool(m['failures']) and not m['tau2']
    fail_marker = ' <span class="failed">(!)</span>' if has_failures else ''

    detail_count = len(m['detail']) - 3  # subtract name, category, gpu
    detail_class = 'has-detail' if detail_count > 1 else ''
    escaped_name = html.escape(m['name'], quote=True)
    js_name = json.dumps(m['name'])

    html_doc += f"""<tr>
  <td class="center">{i}</td>
  <td data-sort="{m['name'].lower()}" class="{detail_class}">
    <span class="model-name" onclick='showDetail({js_name})'>{escaped_name}{fail_marker}</span>
    <span class="info-icon" onclick='showDetail({js_name})'>i</span>
  </td>
  <td class="center"><span class="badge {cat_badge}">{m['category']}</span></td>
  <td class="center">{m['gpu']}</td>
  {bar_cell(m['decode_tps'], tps_max, tps_fmt, '#bc8cff')}
  {bar_cell(m['human_eval'], 1.0, pct, '#238636')}
  {bar_cell(m['livecodebench'], lcb_max, pct, '#1f6feb')}
  {bar_cell(m['tau2'], tau2_max, tau2_fmt, '#d29922')}
  <td class="center">{str(m['tau2_time_min']) if m['tau2_time_min'] else '<span class="na">-</span>'}</td>
</tr>"""

html_doc += """</tbody>
</table>

<h2>Key Findings</h2>

<div class="summary-card">
  <h3>Code Generation vs Agentic Tool Use</h3>
  <ul>
    <li><strong style="color:#3fb950">Ternary-Bonsai-27B (dspark)</strong>: tau2 DOMINANT at 0.80 (previous best was 0.53) while maintaining solid LCB (62.7%). Best agent model by far, at only 1.71 bits/weight ternary</li>
    <li><strong style="color:#3fb950">gemma-4-26B-A4B-it-QAT</strong>: LCB #1 (89.3%) at 83 tok/s -- the A4B MoE (4B active params) is extremely efficient. Best code-gen quality/speed ratio</li>
    <li><strong style="color:#3fb950">gemma-4-12B-it-QAT</strong>: LCB #2 (86.7%) with tau2 0.47 -- the smaller QAT model is also strong on agent tasks</li>
    <li><strong style="color:#3fb950">LFM2.5-8B-Coder-v2</strong>: Now #2 on tau2 (0.53) -- good agent, mediocre coder (45% LCB)</li>
    <li><strong style="color:#f85149">Qwen3.5-9B-DSV4-Flash</strong>: Weak at code gen (18.7% LCB) but strong agent (tau2 0.47) -- reasoning distill hurts coding</li>
    <li>Models that excel at standalone code generation often struggle with conversational tool use, and vice versa. Ternary-Bonsai breaks this pattern: strong at both</li>
  </ul>
</div>

<div class="summary-card">
  <h3>Ternary Quantization: 1.71 bpw Is Viable</h3>
  <ul>
    <li><strong>Ternary-Bonsai-27B</strong> at 1.71 bits/weight (7.2GB deployed) scores tau2=0.80 and LCB=62.7% -- beating most Q4_K_M models that are 2-3x larger</li>
    <li>DSpark speculative decoding worked on the PrismML llama.cpp fork (built at ~/git/llama.cpp-prismml), providing lossless 1.34x speedup</li>
    <li>At 36.3 tok/s on V100 with 262K context, it is practical for production agent workloads</li>
    <li>Qwen3.6-27B-MTP (Q4_K_M, 17.1GB) scored LCB=53.3% and tau2=0.40 -- the ternary model outperforms the full-precision Q4 across both benchmarks at 1/3 the size</li>
  </ul>
</div>

<div class="summary-card">
  <h3>Quantization Degradation: Q1_0 vs Q2_0 (Bonsai-27B)</h3>
  <ul>
    <li><strong>Bonsai-27B Q1_0 (1.13 bpw, 3.8GB)</strong>: LCB drops from 62.7% to 46.7% (-16pp), tau2 collapses from 0.45 to 0.14 (-0.31) vs Q2_0</li>
    <li>262K context now fits on 3060 (9.7GB vs 12.8GB for Q2_0 which OOM'd), throughput +6% (28.3 vs 26.6 tok/s)</li>
    <li>Agentic tool-use is far more sensitive to weight precision than code generation (retains 75% LCB quality but only 31% tau2 quality)</li>
  </ul>
</div>

<div class="summary-card">
  <h3>Speed vs Quality Tradeoffs</h3>
  <ul>
    <li><strong style="color:#bc8cff">LFM2.5-8B base</strong> is the fastest at 213 tok/s -- 6x faster than MoE models -- with decent tau2 (0.46)</li>
    <li><strong style="color:#bc8cff">gemma-4-26B-A4B-it-QAT</strong> hits 83 tok/s despite being a "26B" model -- only 4B params active per token via MoE</li>
    <li><strong style="color:#bc8cff">Qwen3.5-9B-DSV4-Flash</strong> at 81 tok/s is the fastest 9B model, strong for agent workloads</li>
    <li><strong style="color:#bc8cff">MoE 35B models</strong> at 26-28 tok/s are the slowest but competitive on LCB (80-85%)</li>
    <li><strong style="color:#bc8cff">Qwen3.6-27B-MTP</strong> at 30 tok/s is slow for a 27B but competitive across benchmarks</li>
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
    html_doc += '<tr><td colspan="3" class="center na">No failures recorded</td></tr>'
else:
    for name, f in all_failures:
        bench = f.get('benchmark', '?')
        err = str(f.get('error', '?'))[:120]
        html_doc += f'<tr><td>{html.escape(name)}</td><td>{html.escape(bench)}</td><td class="failed">{html.escape(err)}</td></tr>'

html_doc += """</table>

<div class="note">
  <strong>Methodology Notes:</strong><br>
  - HumanEval: Only Qwen2.5-Coder completed successfully (raw code completion mode). Other models need chat-mode rerun. N/A = not run, not failed.<br>
  - LiveCodeBench: 75 problems from latest release, pass@1, temperature=0.0, thinking disabled via enable_thinking=false where supported<br>
  - tau2-bench: Airline domain, 15 tasks, max 30 steps (15 for small-context models), 300s timeout, agent on 3060, user sim on V100<br>
  - DeepSeek-R1 models ran tau2 with reduced context (16-32K) to fit VRAM; LCB with 32K context<br>
  - tok/s measured at 8K context with 256-token decode, flash attention on; production config (cpu-moe, q8_0 KV for MoE; q4_0 KV for dense)<br>
  - DeepSeek-Coder-V2-Lite tok/s measured at 4K context due to VRAM constraints<br>
  - Speed benchmarks are raw decode throughput; actual benchmark throughput varies with prompt processing overhead<br>
  - tau2 user sim varies by GPU: Bonsai Q2_0 (27B, V100) when agent on 3060; Gemma QAT (12B, 3060) when agent on V100. Cross-GPU tau2 comparisons are not fully objective.
</div>

<!-- Detail Modal -->
<div class="modal-overlay" id="detailModal" onclick="if(event.target===this)closeModal()">
  <div class="modal">
    <div class="modal-header">
      <h3 id="modalTitle">Model Details</h3>
      <button class="modal-close" onclick="closeModal()">&times;</button>
    </div>
    <div class="modal-body" id="modalBody"></div>
  </div>
</div>

<script>
const DETAIL_DATA = """ + detail_json + """;
const DETAIL_MAP = {};
DETAIL_DATA.forEach(d => { DETAIL_MAP[d.name] = d; });

function esc(s) {
  if (s === null || s === undefined) return '';
  const d = document.createElement('div');
  d.textContent = String(s);
  return d.innerHTML;
}

function kvGrid(obj, valueClass) {
  let html = '<div class="modal-kv">';
  for (const [k, v] of Object.entries(obj)) {
    const cls = valueClass && (typeof v === 'string' && (v.includes('gguf') || v.includes('--'))) ? 'v mono' : 'v';
    html += `<span class="k">${esc(k)}</span><span class="${cls}">${esc(v)}</span>`;
  }
  html += '</div>';
  return html;
}

function showModal(detail) {
  document.getElementById('modalTitle').textContent = detail.name;
  let body = '';

  // Status badge
  if (detail.status) {
    const statusColor = detail.status === 'completed' ? '#3fb950' : '#d29922';
    body += `<div style="margin-bottom:12px"><span class="badge" style="background:${statusColor}22;color:${statusColor};border:1px solid ${statusColor}44">${esc(detail.status)}</span>`;
    if (detail.category) body += ` <span style="color:#8b949e;font-size:0.85em">${esc(detail.category)} on ${esc(detail.gpu || '?')}</span>`;
    body += '</div>';
  } else if (detail.category) {
    body += `<div style="margin-bottom:12px;color:#8b949e;font-size:0.85em">${esc(detail.category)} on ${esc(detail.gpu || '?')}</div>`;
  }

  // Settings
  if (detail.settings) {
    body += '<div class="modal-section"><div class="modal-section-title">Settings</div>';
    body += kvGrid(detail.settings);
    body += '</div>';
  }

  // VRAM
  if (detail.vram) {
    body += '<div class="modal-section"><div class="modal-section-title">VRAM & Context</div>';
    body += kvGrid(detail.vram);
    body += '</div>';
  }

  // Throughput
  if (detail.throughput) {
    body += '<div class="modal-section"><div class="modal-section-title">Throughput</div>';
    body += kvGrid(detail.throughput);
    body += '</div>';
  }

  // Benchmarks
  if (detail.benchmarks) {
    body += '<div class="modal-section"><div class="modal-section-title">Benchmark Details</div>';
    for (const [benchName, kv] of Object.entries(detail.benchmarks)) {
      body += '<div class="modal-bench">';
      body += `<div class="bench-name">${esc(benchName)}</div>`;
      body += '<div class="bench-kv">';
      for (const [k, v] of Object.entries(kv)) {
        body += `<span class="k">${esc(k)}</span><span class="v">${esc(v)}</span>`;
      }
      body += '</div></div>';
    }
    body += '</div>';
  }

  // Timing
  if (detail.timing) {
    body += '<div class="modal-section"><div class="modal-section-title">Timing</div>';
    body += kvGrid(detail.timing);
    body += '</div>';
  }

  // Errors
  if (detail.hard_error) {
    body += '<div class="modal-section"><div class="modal-section-title">Hard Error</div>';
    body += `<div class="modal-error">${esc(detail.hard_error)}</div>`;
    body += '</div>';
  }
  if (detail.lcb_error && !detail.hard_error) {
    body += '<div class="modal-section"><div class="modal-section-title">LCB Error</div>';
    body += `<div class="modal-error">${esc(detail.lcb_error)}</div>`;
    body += '</div>';
  }
  if (detail.tau2_error && !detail.hard_error) {
    body += '<div class="modal-section"><div class="modal-section-title">tau2 Error</div>';
    body += `<div class="modal-error">${esc(detail.tau2_error)}</div>`;
    body += '</div>';
  }

  // Failures / adaptive retry
  if (detail.failures && detail.failures.length > 0) {
    body += '<div class="modal-section"><div class="modal-section-title">Failures & Retry History</div>';
    for (const f of detail.failures) {
      body += '<div class="modal-fail-entry">';
      body += `<div class="fail-header">${esc(f.benchmark || '?')}`;
      if (f.attempt) body += ` - attempt ${esc(f.attempt)}`;
      if (f.timestamp) body += ` <span style="color:#484f58">(${esc(f.timestamp)})</span>`;
      body += '</div>';
      body += `<div>${esc(f.error)}</div>`;
      if (f.settings) body += `<div class="fail-settings">Settings: ${esc(f.settings)}</div>`;
      body += '</div>';
    }
    body += '</div>';
  }

  document.getElementById('modalBody').innerHTML = body;
  document.getElementById('detailModal').classList.add('active');
}

function showDetail(name) {
  if (DETAIL_MAP[name]) {
    showModal(DETAIL_MAP[name]);
  }
}

function closeModal() {
  document.getElementById('detailModal').classList.remove('active');
}

document.addEventListener('keydown', function(e) {
  if (e.key === 'Escape') closeModal();
});

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
      case 'gpu': return cells[3].textContent.trim();
      case 'tps': return parseFloat(cells[4].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'he': return parseFloat(cells[5].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'lcb': return parseFloat(cells[6].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'tau2': return parseFloat(cells[7].textContent.replace(/[^0-9.]/g, '')) || -1;
      case 'tau2_time': return parseFloat(cells[8].textContent.replace(/[^0-9.]/g, '')) || 99999;
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
    f.write(html_doc)

print("Report generated at:", REPORT_OUT)
print("Models:", len(models))
print("With detail data:", has_detail)
print("With tok/s:", len([m for m in models if m['decode_tps']]))
print("With LCB:", len([m for m in models if m['livecodebench']]))
print("With tau2:", len([m for m in models if m['tau2']]))
