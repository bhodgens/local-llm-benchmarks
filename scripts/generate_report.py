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

def _search_terms_for(name):
    """Candidate LCB output-dir tokens for a progress entry name, including
    variants with noise tokens (MTP, Sharp, quant suffixes) stripped — LCB dir
    names don't always carry the same tokens as the entry name (e.g. entry
    'Qwythos-9B-... MTP Q4_K_M' vs dir 'Qwythos-9B-... Q4_K_M')."""
    safe = name.replace(' ', '_')
    terms = [safe, safe.replace('_Q4_K_M','').replace('_Q6_K','').replace('_Q8_0','').replace('_Q4_0','').replace('_IQ4_XS','')]
    for t in list(terms):
        stripped = t
        for noise in ('_MTP', '_Sharp', '_dspark', '_3060', '_V100'):
            stripped = stripped.replace(noise, '')
        if stripped and stripped not in terms:
            terms.append(stripped)
    return terms


def find_lcb_score(model_name):
    search_terms = _search_terms_for(model_name)
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

def find_bench_date(model_name):
    """Benchmark run day (YYYY-MM-DD): explicit timestamps first, then the
    mtime of the matching LCB output file as a fallback proxy."""
    for k in ('timestamp', 'start_time', 'end_time'):
        v = model_name.get(k)
        if v:
            return str(v)[:10]
    name = model_name['name']
    search_terms = _search_terms_for(name)
    # Fuzzy fallbacks: strip parentheticals and known suffix tokens to find the LCB dir
    base = re.sub(r'\(.*?\)', '', name).strip()
    for junk in (' Q4_K_M', ' Q4_K_S', ' Q5_K_S', ' Q6_K', ' Q8_0', ' Q4_0', ' Q2_0',
                 ' MTP', ' BF16', ' GGUF', ' EXL3'):
        base = base.replace(junk, '')
    base = base.strip()
    if base and base.lower() not in [t.lower() for t in search_terms]:
        search_terms.append(base)
    # First-word prefix fallback (e.g. 'gemma4-coding fable5-composer2.5' -> 'gemma4-coding')
    first_word = base.split()[0] if base.split() else ''
    if len(first_word) >= 8 and first_word.lower() not in [t.lower() for t in search_terms]:
        search_terms.append(first_word)
    # Curated aliases: display name fragment -> exact LCB output dir
    ALIASES = {
        'BTL-3 Full Q4_K_M': 'BTL-3-merged-Q4_K_M',
    }
    for frag, dirname_alias in ALIASES.items():
        if frag.lower() in name.lower():
            search_terms.append(dirname_alias)
    # BTL-4 style: 'BTL-4 Q4_K_M' -> 'BTL-4' would also match BTL-4-IQ2_XXS; handled by
    # best-mtime? No - wrong dir could win. Prefer term with highest similarity: pick dirs
    # whose name starts with the base when base is short.
    best = None
    for base_dir in [LCB_OUTPUT_DIR, '/tmp/coding-bench/results/lcb_thinking_off']:
        if not os.path.exists(base_dir):
            continue
        for d in glob.glob(base_dir + '/*'):
            dirname = os.path.basename(d)
            for term in search_terms:
                tl = term.lower().replace('_', ' ').replace('-', ' ')
                dl = dirname.lower().replace('_', ' ').replace('-', ' ')
                if (term and (term in dirname or dirname in term)) or \
                   (len(tl) >= 6 and (tl in dl or dl.startswith(tl))):
                    eval_files = glob.glob(d + '/*_eval.json')
                    if eval_files:
                        mt = os.path.getmtime(eval_files[0])
                        if best is None or mt > best:
                            best = mt
    if best is not None:
        return datetime.fromtimestamp(best).strftime('%Y-%m-%d')
    return None

def get_tps(name):
    if name in orig_tps:
        return orig_tps[name]
    if name in tps_probes:
        return tps_probes[name]
    return None


# ECC-off retest measurements (bench_results.json [ecc-off] keys), used to fill
# tok/s cells for rows whose original runs predate the speed_norm convention
# (e.g. Nail stock, which has no decode_tps of its own). Only used when the
# entry itself has no recorded decode_tps.
try:
    with open('/home/caimlas/llm-benchmarks/bench_results.json') as _f:
        _ecc_results = json.load(_f)
except (FileNotFoundError, json.JSONDecodeError):
    _ecc_results = {}

_ECC_FILL = {
    # progress entry name -> bench_results.json key (8K-ctx chat probe, ECC-off)
    'Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL': 'Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf [ecc-off]',
    'Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL [Sharp]': 'Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf [ecc-off]',
    'Ornith-1.5-35B-A3B Q4_K_M [Sharp]': 'Ornith-1.5-35B-Q4_K_M.gguf [ecc-off]',
}


# Base-model lineage per entry. Sources:
#   [gguf] = general.base_model.* from the GGUF header (authoritative)
#   [hf]   = upstream model card / base_model tag (authoritative)
#   [infer]= family/architecture inference (flagged in lineage note)
# value = (base_label, lineage_note); unknown bases intentionally omitted -> shown as em dash.
BASE_MODELS = {
    # --- Qwen3.8-27B family ---
    'Qwen3.8-27B Q4_K_S': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [gguf]'),
    'Qwen3.8-27B Q4_K_M': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [gguf]'),
    'Qwen3.8-27B UD-IQ2_XXS': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [gguf]'),
    'Qwen3.8-27B Q4_K_M MTP': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [gguf]'),
    'Qwen3.8-27B Q4_K_M MTP [Sharp]': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [gguf]'),
    'Qwen3.8-27B Heretic Q4_K_M': ('Qwen3.8-27B', 'Qwen3.8-27B abliterated (Heretic) [infer: family]'),
    'Qwen3.8-27B Uncensored Q4_K_M MTP': ('Qwen3.8-27B', 'Qwen3.8-27B abliterated [infer: family]'),
    'Qwen3.8-27B AEON Ultimate Uncensored Q4_K_M MTP': ('Qwen3.8-27B', 'Aeon Uncensored bf16 merge [infer: family]'),
    'Qwen3.8-27B UD-IQ3_S (GGUF, ~3.5bpw)': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [gguf]'),
    'Qwen3.8-27B UD-Q4_K_S (GGUF, ~4.5bpw)': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [gguf]'),
    'Qwen3.8-27B EXL3 3.5bpw [Mia-AiLab]': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [infer: family]'),
    'Qwen3.8-27B EXL3 4.0bpw [turboderp]': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [infer: family]'),
    'Qwen3.8-27B EXL3 4.5bpw [darkbit1001]': ('Qwen3.8-27B', 'Qwen/Qwen3.8-27B [infer: family]'),
    'Carnice-V3 Q4_K_M': ('Qwen3.8-27B', 'base_model: Qwen/Qwen3.8-27B [gguf]'),
    'Carnice-V3 Q4_K_M [Sharp]': ('Qwen3.8-27B', 'base_model: Qwen/Qwen3.8-27B [gguf]'),
    'Qwopus3.8-27B-Flash MTP Q4_K_M': ('Qwen3.8-27B', 'Qwopus3.8 line [infer: family/name]'),
    'Whittle-MoE-27B-A18B v2.1 Q4_K_M': ('Qwen3.8-27B', 'base_model: Qwen/Qwen3.8-27B-FP8, dense->MoE sparsification (logic65) [hf]'),
    'Whittle-MoE-27B-A18B v2.1 Q4_K_M (3060)': ('Qwen3.8-27B', 'base_model: Qwen/Qwen3.8-27B-FP8, dense->MoE sparsification (logic65) [hf]'),
    'Whittle-MoE-27B-A18B v2.1 BF16': ('Qwen3.8-27B', 'base_model: Qwen/Qwen3.8-27B-FP8, dense->MoE sparsification (logic65) [hf]'),

    # --- Qwen3.6-27B family ---
    'Qwen3.6-27B-MTP Q4_K_M': ('Qwen3.6-27B', 'Qwen/Qwen3.6-27B [infer: family]'),
    'Qwen3.6-27B-FableFusion-MTP Q4_K_M': ('Qwen3.6-27B', 'FableFusion merge on Qwen3.6-27B [infer: family/name]'),
    'Qwen3.6-27B-FableFusion-711 Q4_K_M MTP': ('Qwen3.6-27B', 'FableFusion-711 merge on Qwen3.6-27B [infer: family/name]'),
    'ThinkingCap-Qwen3.6-27B Q4_K_M (V100)': ('Qwen3.6-27B', 'base_model: Qwen/Qwen3.6-27B (bottlecapai) [hf]'),
    'ThinkingCap-Qwen3.6-27B Q4_K_M [Sharp]': ('Qwen3.6-27B', 'base_model: Qwen/Qwen3.6-27B (bottlecapai) [hf]'),
    'BTL-3 Full Q4_K_M (V100)': ('Qwen3.6-27B', 'base_model: Qwen/Qwen3.6-27B pinned rev (badtheorylabs) [hf]'),
    'BTL-3-Compact AVQ2': ('Qwen3.6-27B', 'lineage Qwen3.6-27B -> BTL-3 RL-0013 (badtheorylabs) [hf]'),
    'BTL-3-Compact AVQ2 (V100)': ('Qwen3.6-27B', 'lineage Qwen3.6-27B -> BTL-3 RL-0013 (badtheorylabs) [hf]'),
    'Qwopus3.6-27B-v2-MTP Q4_K_M': ('Qwen3.6-27B', 'Qwopus3.6-27B-v2 <- Qwen/Qwen3.6-27B (Jackrong) [hf]'),
    'Qwopus3.6-27B-Coder-MTP Q5_K_S (speed-only)': ('Qwen3.6-27B', 'Coder <- Qwopus3.6-27B-v2 <- Qwen/Qwen3.6-27B [hf]'),
    'Qwopus3.6-27B-Coder-Compat-MTP Q4_K_M (speed-only)': ('Qwen3.6-27B', 'Coder <- Qwopus3.6-27B-v2 <- Qwen/Qwen3.6-27B [hf]'),

    # --- Qwen3.5-27B family ---
    'Qwythos-27B-MTP Q4_K_M': ('Qwen3.5-27B', 'base_model: Qwen/Qwen3.5-27B (empero-ai, full-param SFT->DPO->ESFT) [hf]'),
    'Qwythos-27B-MTP Q4_K_M [Sharp]': ('Qwen3.5-27B', 'base_model: Qwen/Qwen3.5-27B (empero-ai) [hf]'),

    # --- Qwen3.6-35B-A3B family ---
    'Qwen3.6-35B-A3B IQ3_K_R4': ('Qwen3.6-35B-A3B', 'Qwen/Qwen3.6-35B-A3B [gguf]'),
    'Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M': ('Qwen3.6-35B-A3B', 'base_model: Qwen/Qwen3.6-35B-A3B [gguf]'),
    'Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL': ('Qwen3.6-35B-A3B', 'base_model: Qwen/Qwen3.6-35B-A3B [gguf]'),
    'Nail-Qwen3.6-35B-A3B-UD-Q4_K_XL [Sharp]': ('Qwen3.6-35B-A3B', 'base_model: Qwen/Qwen3.6-35B-A3B [gguf]'),
    'RavenX-OpenFable-Holo3 Q4_K_M': ('Qwen3.6-35B-A3B', 'base_model: nightmedia Qwen3.6-35B-A3B-Holo3-Qwopus-BF16 [gguf]'),
    'RavenX-OpenFable-Holo3 Q4_K_M [Sharp]': ('Qwen3.6-35B-A3B', 'base_model: nightmedia Qwen3.6-35B-A3B-Holo3-Qwopus-BF16 [gguf]'),
    'Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M': ('Qwen3.6-35B-A3B', 'Qwopus3.6 coder line, MoE sibling of 27B-v2 [infer: family]'),
    'Hermes3.6-35B-A3B Genesis V5 APEX-Compact': ('Qwen3.6-35B-A3B', 'gguf name: Qwen3.6-35B-A3B-Uncensored-HauhauCS [gguf]'),
    'Ornith-1.5-35B-A3B Q4_K_M': ('Ornith-1.0 35B', 'Ornith-1.5 extends Ornith-1.0 (built on Qwen3.5 + Gemma4) (ornith-ai) [hf]'),
    'Ornith-1.5-35B-A3B Q4_K_M [Sharp]': ('Ornith-1.0 35B', 'Ornith-1.5 extends Ornith-1.0 (built on Qwen3.5 + Gemma4) (ornith-ai) [hf]'),
    'Ornith-1.5-35B-A3B NVFP4 [FreeToken]': ('Ornith-1.0 35B', 'Ornith-1.5 extends Ornith-1.0 (built on Qwen3.5 + Gemma4) (ornith-ai) [hf]'),
    'BTL-4 IQ2_XXS (3060)': ('Ornith-1.0 35B', 'base_model: Ornith 1.0 35B [gguf]'),
    'BTL-4 Q4_K_M': ('Ornith-1.0 35B', 'base_model: Ornith 1.0 35B [gguf]'),

    # --- Qwen3.5-9B / 4B family ---
    'Qwythos-9B-Claude-Mythos-5-1M MTP Q4_K_M': ('Qwen3.5-9B', 'base_model: Qwen/Qwen3.5-9B (empero-ai, Claude-Mythos traces) [hf]'),
    'Mythos-9B-MTP Q4_K_M': ('Qwen3.5-9B', 'Qwythos-9B-Claude-Mythos-5-1M <- Qwen/Qwen3.5-9B [hf]'),
    'Qwen3.5-9B-DeepSeek-V4-Flash Q4_K_M': ('Qwen3.5-9B', 'DeepSeek-V4 reasoning distill of Qwen3.5-9B [infer: family/name]'),
    'Qwen3.5-4B-MTP Q4_K_M (ThumbLLM)': ('Qwen3.5-4B', 'base_model: Qwen/Qwen3.5-4B [gguf]'),

    # --- LFM ---
    'LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M': ('LFM2.5-8B-A1B', 'base_model: LiquidAI/LFM2.5-8B-A1B [gguf]'),
    'LFM2.5-8B-A1B base Q4_K_M': ('LFM2.5-8B-A1B-Base', 'base_model: LiquidAI/LFM2.5-8B-A1B-Base [gguf]'),
    'LFM2.5-8B-A1B Q6_K': ('LFM2.5-8B-A1B-Base', 'base_model: LiquidAI/LFM2.5-8B-A1B-Base [gguf]'),

    # --- Gemma 4 ---
    'gemma-4-12B-it-QAT Q4_0': ('gemma-4-12B-it', 'google QAT release (gg-hf-qat) [gguf]'),
    'gemma-4-12B-it-QAT Q4_0 (3060 128K)': ('gemma-4-12B-it', 'google QAT release (gg-hf-qat) [gguf]'),
    'gemma-4-12B-it-QAT w4a16 [FreeToken]': ('gemma-4-12B-it', 'google w4a16-ct checkpoint [gguf-lineage]'),
    'gemma-4-26B-A4B-it-QAT Q4_0': ('gemma-4-26B-A4B-it', 'google QAT release [gguf]'),
    'gemma4-coding Q4_K_M': ('gemma-4-12B', 'Gemma4 Coding merged fp16 [infer: gguf name]'),
    'gemma4-v2 Q4_K_M': ('gemma-4-12B', 'Gemma4 v2 merged fp16 [infer: gguf name]'),
    'gemma4-v2-agentic Q3_K_M (3060+MTP)': ('gemma-4-12B', 'gemma4-v2 agentic merge [infer: family]'),
    'gemma4-v2-agentic Q4_K_M (3060+MTP)': ('gemma-4-12B', 'gemma4-v2 agentic merge [infer: family]'),
    'gemma4-coding fable5-composer2.5 Q4_K_M': ('gemma-4-12B', 'merge: gemma4-coding + fable5-composer2.5 [infer: name]'),
    'RavenX-OpenFable-Coderagent gemma4 Q4_K_M': ('gemma-4-12B', 'coderagent gemma4 merge [infer: name]'),

    # --- misc / other families ---
    'Ternary-Bonsai-27B Q2_0 (dspark)': ('Qwen3.6-27B', 'Derived from Qwen3.6-27B, ternary QAT (Prism ML) [hf]'),
    'Ternary-Bonsai-27B Q2_0 (dspark) [Sharp]': ('Qwen3.6-27B', 'Derived from Qwen3.6-27B, ternary QAT (Prism ML) [hf]'),
    'Ternary-Bonsai-27B Q2_0 (3060)': ('Qwen3.6-27B', 'Derived from Qwen3.6-27B, ternary QAT (Prism ML) [hf]'),
    'Ternary-Bonsai-27B Q2_0 (3060+dspark)': ('Qwen3.6-27B', 'Derived from Qwen3.6-27B, ternary QAT (Prism ML) [hf]'),
    'Bonsai-27B Q1_0 (3060)': ('Qwen3.6-27B', '1-bit Bonsai, derived from Qwen3.6-27B (Prism ML) [hf]'),
    'Qwythos-27B-v1 Q4_K_M': ('Qwen3.5-27B', 'base_model: Qwen/Qwen3.5-27B (empero-ai) [hf]'),
    'qwen2.5-coder-14b-instruct Q4_K_M': ('Qwen2.5-Coder-14B', 'Qwen/Qwen2.5-Coder-14B-Instruct [gguf]'),
    'DeepSeek-R1-0528-Qwen3-8B Q8_0': ('Qwen3-8B', 'DeepSeek-R1-0528 distill to Qwen3-8B (official DeepSeek) [infer: known release]'),
    'DeepSeek-R1-0528-Qwen3-8B Q4_K_M': ('Qwen3-8B', 'DeepSeek-R1-0528 distill to Qwen3-8B (official DeepSeek) [infer: known release]'),
    'DeepSeek-Coder-V2-Lite IQ4_XS': ('—', 'standalone upstream model (DeepSeek-Coder-V2-Lite)'),
    'Neutrino-8B (FV5 ternary)': ('Qwen3-8B', 'Base: Qwen/Qwen3-8B, ternary QAT by Fermion Research [hf]'),
    'Instella-MoE-16B-A3B-SFT INT4 (bitsandbytes NF4)': ('AMD Instella-MoE-16B-A3B', 'AMD Instella SFT [infer: name]'),
    'Nanbeige4-3B-Thinking Q8_0': ('Nanbeige4-3B', '[infer: name]'),
    'Nanbeige4-3B-Thinking Q4_K_M': ('Nanbeige4-3B', '[infer: name]'),
    'Muse-Glimmer-30B-UD-Q4_K_XL': ('—', 'standalone (own muse-glimmer arch) [gguf]'),
    'Muse-Glimmer-30B-UD-Q4_K_XL [Sharp]': ('—', 'standalone (own muse-glimmer arch) [gguf]'),
    'Laguna-S-2.1 UD-IQ3_S': ('Laguna-S-2.1', 'base_model: poolside/Laguna-S-2.1 [gguf]'),
    'Laguna-S-2.1 UD-IQ3_S (V100)': ('Laguna-S-2.1', 'base_model: poolside/Laguna-S-2.1 [gguf]'),
}

def get_base(name):
    return BASE_MODELS.get(name, ('—', ''))



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

    # BenchKit suites
    bk = m.get('benchkit', {})
    if isinstance(bk, dict) and bk:
        bkd = {}
        for suite, d in bk.items():
            if isinstance(d, dict) and d.get('score_pct') is not None:
                entry = {'score': f"{d['score_pct']:.0f}% ({d.get('passed','?')}/{d.get('total','?')})"}
                if d.get('wall_time_s'):
                    entry['wall time'] = f"{d['wall_time_s']/60:.0f} min"
                bkd[suite] = entry
        if bkd:
            detail['benchkit'] = bkd

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
    # dspark draft-pair probes (Bonsai ecc retest, MiniCPM draft head)
    for dk, label in [('ecc_dspark_retry', 'Bonsai dspark (draft n=4) decode tok/s'),
                      ('dspark_draft', 'DSpark draft-head decode tok/s')]:
        ds = m.get(dk)
        if isinstance(ds, dict):
            val = ds.get('dspark_decode_tps') or ds.get('decode_tps')
            if val:
                line = str(val)
                if ds.get('plain_baseline'):
                    line += f" (plain {ds['plain_baseline']}, {ds.get('delta_pct','?')}%)"
                throughput[label] = line
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
    if tps is None and name in _ECC_FILL:
        _ecc = (_ecc_results.get(_ECC_FILL[name]) or {}).get('ecc_off') or {}
        tps = _ecc.get('decode_tps_probe')
    if tps is None:
        tps = m.get('decode_tps_3060')
    if tps is None:
        tps = m.get('decode_tps_3060_dspark')

    # dspark draft-pair measurements (ecc_dspark_retry / dspark_draft subdicts):
    # show as annotated main tok/s + detail-modal line, never as a separate column
    dspark_note = None
    for dk, label in [('ecc_dspark_retry', 'Bonsai dspark (draft n=4, ECC-off retest)'),
                      ('dspark_draft', 'MiniCPM dspark (draft n=4)')]:
        ds = m.get(dk)
        if isinstance(ds, dict) and ds.get('dspark_decode_tps') or isinstance(ds, dict) and ds.get('decode_tps'):
            val = ds.get('dspark_decode_tps') or ds.get('decode_tps')
            base = ds.get('plain_baseline')
            if val:
                dspark_note = f"{label}: {val} tok/s" + (f" vs plain {base} ({ds.get('delta_pct','?'):+}%)" if base else "")
                if tps is None:
                    tps = val

    template = m.get('template') or ('stock' if 'livecodebench' in m or m.get('gpu') else None)
    engine = m.get('engine') or 'llama.cpp'
    bk = m.get('benchkit', {})
    bk_score = bk.get('sanity', {}).get('score_pct') if isinstance(bk, dict) else None

    # MTP acceptance (from mtp_acceptance summary recorded by orchestrators)
    mtp_acc = None
    mtp_tps = None
    macc = m.get('mtp_acceptance')
    if isinstance(macc, dict):
        n3 = macc.get('MTP-n3') or macc.get('MTP-n5')
        if isinstance(n3, dict):
            a = n3.get('acceptance_avg')
            if isinstance(a, (int, float)):
                mtp_acc = round(a * 100, 1)  # percent
                mtp_tps = n3.get('decode_tps')

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
    base_label, base_note = get_base(name)
    if base_note:
        detail['lineage'] = base_note

    models.append({
        'name': name,
        'category': category,
        'gpu': m.get('gpu', '3060'),
        'base_model': base_label,
        'human_eval': he_score,
        'livecodebench': lcb_score,
        'tau2': tau2_reward,
        'tau2_time_min': round(tau2_time / 60) if tau2_time else None,
        'decode_tps': tps,
        'mtp_acceptance': mtp_acc,
        'mtp_tps': mtp_tps,
        'template': template,
        'engine': engine,
        'benchkit_sanity': bk_score,
        'bench_date': find_bench_date(m),
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
.badge-sharp { background: #f0883e22; color: #f0883e; border: 1px solid #f0883e44; }
.badge-stock { background: #8b949e18; color: #6e7681; border: 1px solid #8b949e30; }
.badge-ft { background: #a371f722; color: #a371f7; border: 1px solid #a371f744; }
.badge-lcpp { background: #8b949e14; color: #768390; border: 1px solid #8b949e28; }
.badge-exl3 { background: #f0883e22; color: #f0883e; border: 1px solid #f0883e44; }
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
  <th data-type="string" data-key="base">Base</th>
  <th data-type="string" data-key="category">Type</th>
  <th data-type="string" data-key="template">Template</th>
  <th data-type="string" data-key="engine">Engine</th>
  <th data-type="number" data-key="bk">Sanity %</th>
  <th data-type="string" data-key="bench_date">Run Date</th>
  <th data-type="string" data-key="gpu">GPU</th>
  <th data-type="number" data-key="tps">tok/s</th>
  <th data-type="number" data-key="he">HumanEval</th>
  <th data-type="number" data-key="lcb">LiveCodeBench</th>
  <th data-type="number" data-key="tau2">tau2-bench</th>
  <th data-type="number" data-key="mtp">MTP acc %</th>
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
    engine_badge = ('<span class="badge badge-ft">FreeToken</span>' if m.get('engine') == 'freetoken'
                    else '<span class="badge badge-exl3">EXL3</span>' if m.get('engine') == 'exllamav3'
                    else '<span class="badge badge-lcpp">llama.cpp</span>')
    tmpl_badge = ''
    if m.get('template') == 'sharp':
        tmpl_badge = '<span class="badge badge-sharp">Sharp</span>'
    elif m.get('template') == 'stock':
        tmpl_badge = '<span class="badge badge-stock">stock</span>'
    else:
        tmpl_badge = '<span class="na">-</span>'
    escaped_name = html.escape(m['name'], quote=True)
    js_name = json.dumps(m['name'])
    base_label, base_note = get_base(m['name'])

    html_doc += f"""<tr>
  <td class="center">{i}</td>
  <td data-sort="{m['name'].lower()}" class="{detail_class}">
    <span class="model-name" onclick='showDetail({js_name})'>{escaped_name}{fail_marker}</span>
    <span class="info-icon" onclick='showDetail({js_name})'>i</span>
  </td>
  <td class="center"><span title="{html.escape(base_note)}">{html.escape(base_label)}</span></td>
  <td class="center"><span class="badge {cat_badge}">{m['category']}</span></td>
  <td class="center">{tmpl_badge}</td>
  <td class="center">{engine_badge}</td>
  {bar_cell(m['benchkit_sanity'], 100.0, lambda v: f'{v:.0f}%', '#39c5cf')}
  <td class="center">{'<span class="na">-</span>' if not m['bench_date'] else m['bench_date']}</td>
  <td class="center">{m['gpu']}</td>
  {bar_cell(m['decode_tps'], tps_max, tps_fmt, '#bc8cff')}
  {bar_cell(m['human_eval'], 1.0, pct, '#238636')}
  {bar_cell(m['livecodebench'], lcb_max, pct, '#79c0ff')}
  {bar_cell(m['tau2'], tau2_max, tau2_fmt, '#56d364')}
  {bar_cell(m['mtp_acceptance'], 100.0, lambda v: f'{v:.1f}%', '#ff7b72')}
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

<div class="summary-card">
  <h3>EXL3 on V100: Instruction-Set Wall (2026-09-04)</h3>
  <ul>
    <li><strong style="color:#f85149">EXL3 quants of Qwen3.8-27B cannot run on the V100</strong> - exllamav3 GEMM/GEMV kernels unconditionally require cp.async + mma.m16n8k16 (Ampere sm_80+ ISA). Verified 4 ways: wheel arch list (8.0-12.0), kernel source (no __CUDA_ARCH__ guards), ptxas probe (nvcc -arch=sm_70 rejects both instructions), runtime (cudaErrorNoKernelImageForDevice)</li>
    <li>No alternative engine exists: vLLM/SGLang have no EXL3 support (open feature request), GPTQModel's EXL3 path is sm_75+. Fix requires an upstream pre-Ampere kernel path</li>
    <li><strong style="color:#3fb950">GGUF fallback benchmarked instead</strong> (bpw-matched, same base): UD-IQ3_S (3.44bpw ~ Mia 3.5bpw) LCB 0.707 @ 29.1 t/s; UD-Q4_K_S (4.49bpw ~ darkbit 4.5bpw) LCB 0.693 @ 33.1 t/s; both 84% sanity</li>
    <li>Q4_K_S is +14% faster at equal quality (LCB delta inside noise band; same-base runs span 0.693-0.760). Unsloth UD quants did NOT beat plain Q4_K_S (0.747)</li>
    <li>EXL3 quality ceiling foregone (turboderp KLD ladder, weighted vs BF16): 3bpw 0.000399 / 4bpw 0.000124 / 5bpw 0.000038 - ~3.2x fidelity per bpw step, inaccessible until an Ampere+ GPU with &gt;=16GB VRAM is available</li>
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

  // BenchKit
  if (detail.benchkit) {
    body += '<div class="modal-section"><div class="modal-section-title">BenchKit Suites</div>';
    for (const [suite, kv] of Object.entries(detail.benchkit)) {
      body += `<div class="modal-bench">`;
      body += `<div class="bench-name">${esc(suite)}</div>`;
      body += '<div class="bench-kv">';
      for (const [k, v] of Object.entries(kv)) {
        body += `<span class="k">${esc(k)}</span><span class="v">${esc(v)}</span>`;
      }
      body += '</div></div>';
    }
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
    // Column position is derived from the header row's data-key order, so the
    // switch below can never go stale when columns are added/reordered.
    const keyToIdx = {};
    table.querySelectorAll('thead th').forEach((th, i) => {
      keyToIdx[th.getAttribute('data-key')] = i;
    });
    const cells = row.querySelectorAll('td');
    const ci = keyToIdx[key] !== undefined ? keyToIdx[key] : -1;
    if (ci < 0) return 0;
    const txt = cells[ci].textContent.trim();
    switch(key) {
      case 'idx': return parseInt(cells[ci].textContent);
      case 'name': return cells[ci].getAttribute('data-sort') || txt.toLowerCase();
      case 'base': case 'category': case 'template': case 'engine': case 'gpu':
        return txt === '—' ? '' : txt.toLowerCase();
      case 'bench_date': return txt === '-' ? '' : txt;
      default:
        // numeric columns: strip non-numeric chars; N/A or '-' -> -1 (or 99999 for time)
        const n = parseFloat(txt.replace(/[^0-9.]/g, ''));
        if (!isNaN(n)) return n;
        return key === 'tau2_time' ? 99999 : -1;
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
        sortDir = key === 'name' || key === 'category' || key === 'base' || key === 'bench_date' ? 'asc' : 'desc';
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
