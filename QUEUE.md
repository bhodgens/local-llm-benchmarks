# Benchmark Queue

Models queued for benchmarking. Not yet downloaded or run.

---

## 1. MiniMax-H3 (realrebelai/MiniMax-H3_GGUFs)

- Source: https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/tree/main
- Target GPU: V100 (CUDA0, 32GB)
- Requested quant: Q4_K_M

### Available Q4_K_M files

| File | Size | Type |
|------|------|------|
| MiniMax-H3-FL2VA-Q4_K_M.gguf | 19.9 GB | Text-to-Video (FL2VA = Flow matching Language-to-Video-Audio) |
| qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf | 14.6 GB | Vision-Language (Qwen3-VL-32B + MiniMax-H3) |

### BLOCKER: Not a text-only LLM

This repo is a multimodal/video generation model. Tagged: text-to-video, minimax, comfyui.
The FL2VA variant is a video generation model requiring ComfyUI + separate VAE files
(https://huggingface.co/Comfy-Org/MiniMax-H3/tree/main/vae). It cannot run in llama.cpp
as a chat/coding model and is not benchmarkable with HumanEval/LCB/Aider.

The qwen3vl-32B variant is a vision-language model (may support text generation via
llama.cpp if the architecture is supported). This is the more viable candidate for the
coding benchmark suite, but requires --mmproj and vision support.

### Download URL (qwen3vl variant - more viable for text benchmarks)
```
https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf
```
Dest: /home/files/llms/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf

### Download URL (FL2VA video variant)
```
https://huggingface.co/realrebelai/MiniMax-H3_GGUFs/resolve/main/MiniMax-H3-FL2VA-Q4_K_M.gguf
```
Dest: /home/files/llms/MiniMax-H3-FL2VA-Q4_K_M.gguf

### Needs resolution before benchmarking
1. Which Q4_K_M file? (FL2VA video vs qwen3vl text+vision)
2. llama.cpp architecture support for MiniMax-H3 / qwen3vl-32B unverified
3. If FL2VA: cannot use coding benchmark suite, needs ComfyUI workflow instead
4. VAE files needed separately for FL2VA variant

---

## 2. Qwen3.6-35B-A3B-Escha-W2 (EschaLabs)

- Source: https://huggingface.co/EschaLabs/Qwen3.6-35B-A3B-Escha-W2
- Target GPU: V100 (CUDA0, 32GB)
- Quant: 2-bit eschamoe (NOT Q4_K_M, NOT GGUF)

### Model details

| Property | Value |
|----------|-------|
| Base | Qwen3.6-35B-A3B (MoE, 256 experts) |
| Quantization | 2-bit eschamoe (mixed 2/3-bit per projection), int8 dense |
| Format | Safetensors (3 shards, 12.3 GB total) |
| Min GPU | 16 GB VRAM, NVIDIA Ampere (sm_80) required |
| Runtime | escha (SGLang engine or ZML engine) -- NOT llama.cpp |
| API | OpenAI-compatible /v1 on port 30000 |
| License | Apache-2.0 |

### BLOCKER: V100 is sm_70 (Volta), below sm_80 (Ampere) minimum

The escha runtime requires Ampere or newer. V100 compute capability 7.0 < 8.0 minimum.
Triton (used by SGLang engine) does not support sm_70. The model will not run on V100.

### BLOCKER: Not GGUF, requires escha runtime

This is safetensors with custom eschamoe quantization. Cannot be loaded by llama.cpp.
Requires installing escha runtime (separate repo: EschaLabs/escha-runtime-qwen3moe).
Needs Python 3.12, torch==2.9.x, CUDA 12.8+.

### Download
```
hf download EschaLabs/Qwen3.6-35B-A3B-Escha-W2 --local-dir /home/files/llms/escha-w2
```
(3 safetensors shards: 5.37 + 5.31 + 1.62 GB = 12.3 GB)

### Needs resolution before benchmarking
1. V100 (sm_70) does not meet sm_80 minimum -- consider 3060 (sm_86) instead?
2. Requires escha/SGLang runtime install (not llama.cpp)
3. torch==2.9.x + CUDA 12.8 dependency chain may conflict with existing setup
4. Not Q4_K_M -- this is a 2-bit quant, much more aggressive

---

## 3. BTL-4 (badtheorylabs/BTL-4) -- ACTIVE

- Source: https://huggingface.co/badtheorylabs/BTL-4
- GGUF: https://huggingface.co/bartowski/badtheorylabs_BTL-4-GGUF
- Base: Ornith-1.0-35B (qwen3_5_moe arch, MoE)
- Tags: agentic, tool-use, code, reasoning, image-text-to-text
- License: Apache-2.0
- Claimed: LCB v6 66.1%, BFCL v4 73.5%, SWE-bench Verified 78.4%
- Generation settings: temp=1.0, top_p=0.95, ctx 262144 native
- Reasoning: deepseek format, must strip reasoning from old turns

### Phase 1: IQ2_XXS on 3060 (12GB, sm_86) -- COMPLETE
- File: badtheorylabs_BTL-4-IQ2_XXS.gguf (9.78 GB)
- Full GPU offload, --reasoning off, --jinja, --reasoning-format deepseek
- Harness: tok/s + LCB (75 problems) + tau2 (airline, 15 tasks)
- Script: scripts/run_btl4_iq2xxs.py
- Results: tok/s=72.9, LCB=50.7%, tau2=0.38 (5/15 passed), VRAM=9905 MiB
- User sim: Qwythos-27B on V100 (Gemma ExLlamaV3 needs Ampere+, cannot use V100)

### Phase 2: Q4_K_M on V100 (32GB, sm_70) -- QUEUED
- File: badtheorylabs_BTL-4-Q4_K_M.gguf (21.4 GB)
- Full GPU offload, same harness
- Script: TBD (run_btl4_q4km.py)

---

## Queued: ThumbLLM model (Qwen3.5-4B-MTP Q4_K_M)

- Source release: https://github.com/TeksEdge/ThumbLLM/releases/tag/thumbllm-qwen3.5-4b-mtp-q4_k_m-cpu-win-x64-v0.1.0
- Actual model: unsloth/Qwen3.5-4B-MTP-GGUF -> `Qwen3.5-4B-Q4_K_M.gguf` (2.83 GB)
- Note: the ThumbLLM release itself is a Windows-x64 CPU-only bundle (exe, 252 MB)
  and does not run on this Linux box. Benchmark the underlying model file instead.
- Target GPU: V100 or 3060 (fits both; MTP heads are the interesting axis ->
  benchmark no-spec vs MTP n=3 like the other Qwen MTP models)
- Harness: tok/s + BenchKit gate + LCB 75; register LCB id `local/qwen35-4b-mtp-q4km`
- Status: QUEUED (not downloaded, not run)

---

## Queued: Qwopus3.8-27B-Flash MTP Q4_K_M (V100)

- Source: https://huggingface.co/Jackrong/Qwopus3.8-27B-Flash-GGUF
- File: `Qwopus3.8-27B-Flash-MTP-Q4_K_M.gguf` (16.81 GB) — Qwen3.8-27B fine-tune
  with bundled NextN/MTP head; fits V100 32GB with full MTP
- Target GPU: V100 (CUDA0)
- **Production-optimal settings** (per model card battery + house findings):
  - `--flash-attn on`, `--cache-type-k q8_0 --cache-type-v q8_0`
  - `--spec-type draft-mtp --spec-draft-n-max 3` (self-speculative MTP;
    model card reports 80.7% weighted draft acceptance, +12.8% decode)
  - `--ctx-size 65536 --ubatch-size 512 --gpu-layers 99`
  - thinking disabled (`--reasoning-budget 0` or template flag) for coding runs;
    LCB rows both thinking-on and thinking-off if time allows
- Harness: tok/s probe + BenchKit sanity:25 + LCB 75 (thinking off);
  record MTP acceptance rates (house report column) alongside decode t/s
- Baselines in progress.json: plain Qwen3.8 Q4_K_M MTP n3 = 35.6 t/s @ 0.674 acc,
  LCB 0.760; this run tests whether Flash's claimed acceptance gain (+14.6pp)
  holds at Q4_K_M on V100
- LCB id to register: `local/qwopus38-flash-q4km-mtp`
- Status: QUEUED (not downloaded, not run)

---

## Disk Space

Current: 366 GB free on /home (1.5 TB used of 1.9 TB)
Required: ~10 GB (IQ2_XXS) + ~21 GB (Q4_K_M) = ~31 GB total
Status: Sufficient

---

# QUEUED (2026-09-06): LCB empty-content audit remediation

Empirical audit of all 52 LCB output dirs found 19 with >=10% empty generations
(empty `output_list` = model returned empty content = scored 0). Root causes:

1. **CONFIRMED + FIXED**: thinking-family models missing from the
   `LCB_DISABLE_THINKING` allowlist (oai_runner.py) thought by default and burned
   the 4096-token budget. Qwythos-9B-Mythos rerun DONE: 0.40 -> **0.587** (+18.7pp),
   0/75 empty with kwarg sent. Same mechanism suspected: Qwen3.8-27B family (17-29% empty despite kwarg being
   sent - sent BEFORE 'qwen38' was added to the allowlist), Heretic-35B (85%),
   DSV4-Flash (81%), R1-8B Q4 (79%), Nanbeige (87%).
2. **UNKNOWN cause**: LFM2.5-8B trio (35-43% empty, NOT a thinking model),
   Carnice-V3 (17%), Muse-Glimmer (20%), K2-Horizon (27%). Zero timeouts/errors
   in runner+server logs; ~70% empty-position overlap across the 3 LFM variants
   -> template/model-level (LFM2.5 returns empty content for certain LCB prompt
   shapes; LFM has no enable_thinking kwarg to send).

### Remediation queue (rerun LCB 75 after current jobs finish)
- **IN PROGRESS (2026-09-06 05:13)**: `scripts/lcb_remediation_reruns.py` running
  all 13 rerunnable models sequentially on V100 (dedicated port 18096, /props
  identity check, never overwrites a score with None). Broken artifacts per model
  -> `<dir>.empty-content-broken`.
- [x] Tier 1: Qwen3.8-27B Q4_K_M, Q4_K_M MTP, Heretic, Uncensored MTP, AEON,
      UD-IQ3_S, UD-Q4_K_S (7 on disk; **Q4_K_S GGUF deleted** -> invalidation note only)
- [x] Tier 2 partially: Heretic-35B-A3B queued in script.
      **DELETED, NOT RERUNNABLE (invalidation notes added to progress.json)**:
      Qwen3.5-9B-DSV4-Flash (61 empty), DeepSeek-R1-0528 Q4_K_M (59 empty),
      Nanbeige4-3B Q8_0 (65 empty). Recorded scores are lower bounds only.
- [x] Tier 3: LFM base/Q6_K/Clean-RealWorld queued in script (rerun doubles as the
      diagnosis: if empty-rate stays high with kwarg sent, cause is template-level)
- [x] Tier 4: Carnice-V3, Muse-Glimmer queued in script. K2-Horizon BF16: 27% empty,
      file deleted earlier -> not rerunnable.
- Rerun trigger for reference: Qwythos-9B-Mythos 0.40 -> 0.587 (+18.7pp), 0/75 empty.

---

# QUEUED (2026-09-06): tok/s re-test, top-5 LCB + top-5 tau2 (V100 ECC disabled)

Purpose: measure ECC-off impact on V100 decode/prompt tok/s. Protocol identical to
original speed_norm: llama-bench (pp512/tg128, -r 3, fa on, q8_0 kv) + 256-token
decode probe. Record next to old numbers; report delta %.
Note: ECC change affects V100 only -> 3060-lane rows are control re-measurements
(two of them have no prior tok/s at all).

### Top-5 LCB (tie at 92% -> 6 entries)
- [ ] Qwythos-27B-v1 Q4_K_M          - V100 - orig flags: run_qwythos_27b (262K, jinja)
- [ ] Qwythos-27B-MTP Q4_K_M         - V100 - orig flags + draft-mtp n3 (42.9 t/s config)
- [ ] Nail-35B UD-Q4_K_XL [Sharp]    - V100 - run_sharp_template.py flags
- [ ] BTL-4 Q4_K_M                   - V100 - run_btl4_q4km_v100.py flags
- [ ] Ornith-1.5-35B Q4_K_M [Sharp]  - V100 - run_sharp_template.py flags
- [ ] gemma-4-12B-it-QAT Q4_0 (3060 128K) - 3060 control

### Top-5 tau2 (tie at 0.50 -> 6 entries)
- [ ] Ternary-Bonsai-27B Q2_0 (dspark) - V100 - PrismML binary + dspark draft config
- [ ] Carnice-V3 Q4_K_M                - V100
- [ ] Qwopus3.6-27B-v2-MTP Q4_K_M      - V100 - cpu-moe 262K config
- [ ] LFM2.5-8B-A1B-Clean-RealWorld-v2 - 3060 control
- [ ] gemma4-coding Q4_K_M             - 3060 - NO prior tok/s (first measurement)
- [ ] DeepSeek-Coder-V2-Lite IQ4_XS    - 3060 - NO prior tok/s (first measurement)

Sequencing: after LCB remediation Tier 1/2 (both need V100; do tok/s re-tests
first per model since the server is already up - probe adds ~3 min per model).

---

# DONE (2026-09-06): ECC tok/s re-tests

Executed via scripts/ecc_toks_retest.py + ecc_toks_followup*.py. Results and
verdict: ecc_retest_results.md + bench_results.json [ecc-off*] keys.
**Verdict: ECC-off = +0% to +5.4% decode on V100, within control-lane noise
(control 3060 moved +4.2% with no HW change). No material tok/s impact.**
Blocked follow-ups (documented in ecc_retest_results.md): Nail/Ornith 262K
cpu-moe configs fail on current llama.cpp build (create_context); Bonsai
dspark PrismML probe aborted (VRAM fit conflicts, low-value per house rule);
DS-Coder-V2-Lite GGUF deleted.
