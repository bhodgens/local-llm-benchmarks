# LLM Benchmark Failure Analysis
Generated: 2026-07-31

## Summary
- **Total models tracked**: 38
- **Complete (LCB + tok/s)**: 28
- **Partial results**: 4
- **Failed**: 6 models across 3 distinct failure modes

---

## Failure Mode 1: LCB Model Registration (FIXABLE)
### Affected: FableFusion-711 Q4_K_M MTP

**Symptom**: LCB runner exits with code 2 after 3.4s
**Root cause**: `KeyError: 'local/fable-711'` - model not registered in `lcb_runner/lm_styles.py`
**Fix**: Add entry to `LanguageModelStore` in `/home/caimlas/git/LiveCodeBench/lcb_runner/lm_styles.py`:
```python
LanguageModel(
    "local/fable-711",
    "FableFusion-711 Q4_K_M MTP",
    LMStyle.OpenAIChat,
    datetime(2024, 4, 1),
),
```
**Status**: Not yet fixed. tok/s completed successfully (30.6 avg).
**Effort to re-run**: ~20 min (register model + relaunch LCB 75)

---

## Failure Mode 2: LCB Assertion Error (MODEL LIMITATION)
### Affected: Nanbeige4-3B-Thinking Q4_K_M, Nanbeige4-3B-Thinking Q8_0

**Symptom**: `AssertionError` at `base_runner.py:63: assert len(result) == args.n`
**Root cause**: The LCB runner expects exactly `n=1` result per problem, but the model returns
empty or malformed responses for some problems. This is a model capability issue:
- Nanbeige-3B is a 3B parameter model with thinking/reasoning tokens
- LCB pass@1 = 0.000 (Q4_K_M) and 0.133 (Q8_0) - extremely low coding capability
- The thinking tokens may confuse the LCB output parser
- Model generates ~82 tok/s server-side but quality is insufficient for coding tasks
**Fix**: Could try `--reasoning off` flag (attempted but still assertion error). The model
is simply too small/weak for LCB coding benchmarks.
**Status**: Model limitation, not a runner bug. Recommend skip.

---

## Failure Mode 3: tau2 Runner Configuration (FIXABLE)
### Affected: Neutrino-8B, Qwythos-27B-v1, FableFusion-711

**Symptom**: tau2 runner exits immediately (code 1, ~8s wall time)
**Root cause**: tau2-bench CLI flags differ from our custom setup:
- `--user-port` is not a valid tau2-bench argument
- tau2-bench expects `--user-llm` and `--agent-llm` with separate API endpoints
- User simulator needs its own llama-server instance on a different port
**Fix**: Need to start two llama-server instances (one for agent, one for user sim)
and use correct tau2-bench syntax:
```bash
tau2 run --domain airline \
  --agent-llm openai/MODEL --agent-llm-args '{"api_base": "http://127.0.0.1:PORT/v1"}' \
  --user-llm openai/gemma-4-12B-it --user-llm-args '{"api_base": "http://127.0.0.1:8081/v1"}'
```
**Status**: Not yet fixed for new models. Existing tau2 results in report are from working configs.

---

## Failure Mode 4: GGUF Conversion - Custom Architecture (BLOCKED)
### Affected: Instella-MoE-16B-A3B-SFT

**Symptom**: `convert_hf_to_gguf.py` fails with "Model InstellaMoEForCausalLM is not supported"
**Root cause**: Custom AMD MoE architecture not supported by llama.cpp converters:
- `model_type: deepseek_v3` but `architectures: ['InstellaMoEForCausalLM']`
- Qwen3 converter: fails on `model.layers.X.mlp.experts.Y.gate_proj.weight` (MoE expert tensors)
- DeepSeekV3 converter: fails on `model.layers.0.self_attn.gate_proj.weight` (attention tensors)
- The model uses a hybrid architecture mixing Qwen attention with custom MoE routing
**Hardware blockers (INITIAL ASSESSMENT -- REVISED)**:
- V100 (sm_70): No native bf16 (needs sm_80+). bf16 upcast to fp32 = 63 GB (won't fit 32 GB).
- RTX 3060 (12 GB, sm_86): bf16 supported, but 16B @ bf16 = 31.0 GB won't fit 12 GB.

**REVISED FEASIBILITY (INT4 quantization via bitsandbytes)**:
- Sparse MoE: 64 experts, only 6 active per token (9.4% activation)
- INT4 quantized total: ~7.7 GB -- FITS in 3060 12 GB VRAM with room for KV cache
- bitsandbytes NF4 + double quantization via HF transformers (trust_remote_code=True)
- transformers 5.13.0 installed, accelerate 1.14.0 installed, bitsandbytes 0.50.0 installed
- Testing INT4 load + generation on 3060 now
- Expected throughput: low (MoE expert swapping overhead + INT4 dequant) but benchmarkable

**Only supported inference**: HF transformers >= 4.57.1 with trust_remote_code=True.
  SGLang is ROCm-only (MI300X). vLLM: unsupported. llama.cpp: unsupported.
**FarSkip architecture**: Two-residual-stream dataflow requires C++ re-implementation in llama.cpp
  -- this is a compute-graph change, not a tensor rename. Substantial engineering effort.
**Status**: SUCCESS -- Running on 3060 at 1.08 tok/s, 11.55 GB VRAM.
Required 6 patches for transformers 5.13.0 compat + custom expert weight fusion loader.
Output is coherent. Too slow for LCB benchmarking (1.08 tok/s) but proves the model works on NVIDIA consumer hardware.

---

## Neutrino-8B Draft Head Analysis

The Neutrino-8B draft head was benchmarked with and without speculative decoding:

| Mode | tok/s (avg) | LCB pass@1 | Notes |
|------|-------------|------------|-------|
| Plain (no draft) | 41.3 | 0.08 | Baseline |
| Draft head ON | 7.3 | N/A | 5.7x SLOWER |

**Finding**: The draft head makes inference **5.7x slower** (7.3 vs 41.3 tok/s).
This suggests the draft model is poorly calibrated or the fermion-fv5 fork's
speculative decoding implementation has overhead issues. The draft head should
not be used for this model.

---

## Failure Mode 5: BTL-4 IQ2_XXS Benchmark Issues (2026-08-09)

### Affected: BTL-4 IQ2_XXS on 3060

Four separate issues occurred during the benchmark run. All diagnosed and resolved.

---

### Issue 5a: LCB Model Not Registered (FIXED)

**Symptom**: LCB runner exits immediately with `KeyError: 'local/btl4-iq2xxs'`
**Root cause**: Model name not registered in LiveCodeBench's `lm_styles.py`
**Fix**: Added entries for both quants:
```python
LanguageModel("local/btl4-iq2xxs", "BTL-4-IQ2_XXS", LMStyle.OpenAIChat, datetime(2024, 4, 1)),
LanguageModel("local/btl4-q4km", "BTL-4-Q4_K_M", LMStyle.OpenAIChat, datetime(2024, 4, 1)),
```
File: `/home/caimlas/git/LiveCodeBench/lcb_runner/lm_styles.py`
**Status**: Fixed. LCB ran successfully on retry (50.67% pass@1).

---

### Issue 5b: tau2 User Sim GPU Conflict (FIXED)

**Symptom**: tau2-bench connection errors on all 15 tasks. Every task fails after 3 retries.
**Root cause**: ExLlamaV3 (Gemma backend on port 8080) requires Ampere+ (sm_80+).
The V100 is sm_70 (Volta), so Gemma cannot be moved to V100 as user sim.
Meanwhile BTL-4 takes ~10GB on the 3060, leaving no room for Gemma's ExLlamaV3 (also ~10GB).
Both models need GPU memory but cannot share the 3060 simultaneously.
**Fix**: Used Qwythos-27B (llama.cpp, port 8081 on V100) as tau2 user simulator instead of Gemma.
Qwythos runs fine on sm_70 via llama.cpp and is already configured as a systemd service.
**Status**: Fixed. tau2 ran successfully (0.38 reward, 5/15 tasks passed).

---

### Issue 5c: tau2 Stale Results Blocking Non-Interactive Run (FIXED)

**Symptom**: tau2 crashes with `EOFError: EOF when reading a line`
**Root cause**: A results.json from the failed first run (issue 5b) existed at
`tau2-bench/data/simulations/tau2_3060_BTL-4-IQ2XXS/results.json`.
tau2 prompts "Do you want to resume the run? (y/n)" interactively.
The benchmark script runs non-interactively (subprocess), so the prompt hits EOF and crashes.
**Fix**: Delete stale results directory before re-running:
`rm -rf /home/caimlas/git/tau2-bench/data/simulations/tau2_3060_BTL-4-IQ2XXS/`
**Status**: Fixed. Added to checklist for future runs.

---

### Issue 5d: LCB Output Directory Mismatch (FIXED)

**Symptom**: LCB completes 75 problems but pass@1 parses as None in the script.
**Root cause**: The script looked for results in `output/local_btl4-iq2xxs/` but
LCB writes to `output/BTL-4-IQ2_XXS/` (derived from the model display name in lm_styles.py,
not the --model CLI argument). The eval JSON existed but in a different directory.
**Fix**: Changed `output_dir` in the script from `local_btl4-iq2xxs` to `BTL-4-IQ2_XXS`.
**Status**: Fixed. Eval JSON parsed correctly on manual check.

---

## Models With Complete Results

| Model | LCB pass@1 | tok/s | tau2 reward | GPU |
|-------|-----------|-------|-------------|-----|
| gemma-4-26B-A4B Q4_0 | 0.893 | 83.3 | 0.400 | 3060 |
| gemma-4-12B Q4_0 (128K) | 0.907 | 38.2 | 0.429 | 3060 |
| RavenX-OpenFable-Holo3 Q4_K_M | 0.853 | 26.8 | - | V100 |
| ThinkingCap-Qwen3.6-27B Q4_K_M | 0.867 | 29.8 | 0.467 | V100 |
| BTL-3 Full Q4_K_M | 0.867 | 29.6 | 0.400 | V100 |
| Hermes3.6-35B Genesis V5 | 0.827 | 26.5 | 0.000 | V100 |
| Qwythos-27B-v1 Q4_K_M | 0.827 | 30.3 | skipped | V100 |
| qwen2.5-coder-14b Q4_K_M | 0.800 | 34.6 | 0.133 | V100 |
| Qwopus3.6-35B-A3B-Coder Q4_K_M | 0.800 | 27.2 | 0.400 | V100 |
| Laguna-S-2.1 UD-IQ3_S | 0.813 | 3.0 | - | V100 |
| Ternary-Bonsai-27B Q2_0 (dspark) | 0.627 | 36.3 | 0.800 | V100 |
| Qwen3.6-27B-MTP Q4_K_M | 0.533 | 30.1 | 0.400 | V100 |
| gemma4-v2 Q4_K_M | 0.587 | - | 0.133 | 3060 |
| Neutrino-8B (FV5) | 0.080 | 41.3 | failed | V100 |
| Nanbeige4-3B Q4_K_M | 0.000 | 10.0 | 0.133 | 3060 |
| BTL-4 IQ2_XXS | 0.507 | 72.9 | 0.380 | 3060 |

Full sortable report: `/home/caimlas/llm-benchmarks/report.html`
