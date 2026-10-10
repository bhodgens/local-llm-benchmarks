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

### Phase 2: Q4_K_M on V100 (32GB, sm_70) -- COMPLETE
- Done (pre-2026-09-07): LCB 0.92, tau2 0.20, 88.8 t/s (progress entry `BTL-4 Q4_K_M`).
- Also benched: DogukanUrker community Q4_K_M build (91.7 t/s, LCB 0.92, tau2 0.40)
  - statistically same model, see 2026-09-07 block below.

---

## Queued: ThumbLLM model (Qwen3.5-4B-MTP Q4_K_M) -- COMPLETE

- Benched earlier (see progress `Qwen3.5-4B-MTP Q4_K_M (ThumbLLM)`):
  LCB 0.56, tau2 0.2727 (backfilled via tau2_backfill_3models.py).

---

## Queued: Qwopus3.8-27B-Flash MTP Q4_K_M (V100) -- COMPLETE

- Benched 2026-09-06/07: sanity 96%, LCB 0.773 (clean rerun), tau2 0.2667
  (backfill), MTP n3 42.9 t/s. Interesting result: its LCB strength does NOT
  transfer to agentic tool use (0.267 vs plain Qwen3.8 MTP 0.40).

---

## Disk Space

Note: the "366 GB free" figure above predates 2026-09-07. Current: ~38 GB free
(after 20 GB BTL-4 + 2.7 GB MiniCPM + 0.35 GB draft downloads + 37 GB pip cache
purge). /home sits near capacity; prune before adding >30 GB models.

---

## Queued (2026-09-07): Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic Q4_K_M

- Source: https://huggingface.co/medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic
- Quant: Q4_K_M (community GGUF when published; otherwise quant from safetensors)
- Family: Qwen3.8-27B abliteration ("OBLITERATED") x Mythos-class agentic tune.
  Note the Qwen3.8 family needed the allowlist fix for thinking suppression
  (qwen38 token now in oai_runner.py); verify enable_thinking:false live before
  LCB, fall back to --reasoning-budget 0 if the template ignores it.
- Target: V100 full offload (~16 GB at Q4_K_M)
- Harness: full lane per house pattern (speed probe -> sanity:25 -> LCB 75
  thinking-off -> tau2 airline/15/seed42/conc2 w/ LFM user sim on 3060)
- Context of interest: benchmarks BOTH against the Qwen3.8 family results
  (0.4667-0.50 tau2, 0.80-0.88 LCB) and against Qwythos-9B-Mythos (the
  Mythos-trace line, LCB 0.587/tau2 0.40)
- Status: QUEUED (not downloaded, not run)

---

# OPEN ITEMS (2026-09-07 audit)

## Genuinely runnable, needs user decision
- **MiniCPM5-2B tau2 variance check**: GPU lanes scored 0.571 (3060) vs 0.333
  (V100); traced to 3 bistable agent-loop tasks, McNemar p=0.42 on LCB (noise).
  Optional: 3-seed tau2 average for a defensible single number (~45 min).

## Blocked / parked (was: MiniMax-H3, Escha-W2)
- MiniMax-H3 GGUF repo = multimodal video/VL models, not text-benchmarkable
  (FL2VA needs ComfyUI; qwen3vl variant needs --mmproj vision path).
- Escha-W2 = eschamoe safetensors, needs escha/SGLang runtime, sm_80+ minimum;
  V100 (sm_70) hard-blocked, same class as EXL3. Not benchmarkable here.

## Harness debt (from ECC retest, unresolved)
- Nail/Ornith 262K cpu-moe serving fails `create_context` on current llama.cpp
  build (worked originally) - regression worth a bisect if 262K MoE matters.
- 262K-ctx true-ECC-off tok/s deltas for Nail/Ornith remain unmeasured
  (all ECC verdicts are from 8K-ctx configs; verdict direction unlikely to change).

---
---

# EXECUTED (2026-10-06): K2-Horizon-Uno / Bonsai2 + CRACK / Xing4.0 batch

All staged lanes ran 2026-10-06 via scripts/run_bench0918_batch.py (10-min idle watcher).
Results: Bonsai2 44.4t/s 76.0% LCB 0.533 tau2 | CRACK 43.4t/s 73.3% 0.133 (REJECTED, decay
gate fired -> 1bit CRACK not benched) | PTQ1_0 43.4t/s 73.3% 0.467; +MTP n1 54.4t/s same sanity
| Xing V100 50.8t/s 48.0% 0.267; 3060 probe 17.2t/s, MTP n3 hurts | K2-Uno INVALID (merged
diffusion adapter destroys AR -> comma soup; not benchmarkable). Full detail: skill ref
bench0918-results.md + /tmp/coding-bench/bench0918_results.json.

---

# STAGED (historical 2026-09-18): K2-Horizon-Uno / Bonsai2 + CRACK / Xing4.0 / dspark sidecar

Prep complete (execution record above). All files staged under /var/tmp/llms/bench-0918/
(/home was 100% full; /var/tmp has room). All downloads byte-exact vs Content-Length,
GGUF v3 verified. Engines validated by smoke test.

## A. Bonsai 2 27B official PQ2_0 (prism-ml/Ternary-Bonsai-2-27B-gguf) - V100
- File: bonsai2/Ternary-Bonsai-2-27B-PQ2_0.gguf (7.21 GB, PQ2_0 ternary 2.13 bpw)
- Engine: PrismML fork prebuilt b10685 (staged: prismml-bin/llama-prism-b10685-7dffb15/)
  - prebuilt carries sm_86/89 REAL + sm_70/75/80/90 PTX -> V100 works via PTX JIT
    (first-load JIT took ~min; subsequent loads 12s). Local ~/git/llama.cpp-prismml
    build is CPU-only; use the prebuilt, or rebuild with -DCMAKE_CUDA_ARCHITECTURES="70;86".
- Smoke: OK on V100 beside prod (14.5 t/s chat, thinking on, 2K ctx, coherent)
- Lane: V100, dedicated port, --jinja. Full 4 legs (speed/sanity/LCB/tau2).
- 128K ctx: CORRECTED - hybrid GDN KV is cheap. Prior gen-1 Ternary-Bonsai-27B Q2_0
  (same 7.2 GB) measured 200K ctx = 11.4 GB on 3060; 100K w/ dspark = 9.1 GB. Bonsai 2
  should be similar -> 3060 CAN host at 128K+; V100 chosen for headroom + faster decode
  (gen-1: 36.3 t/s V100 vs 26.6 t/s 3060).
- DSpark sidecar (user-directed): drafter converted + quantized:
  - Source: prism-ml/Ternary-Bonsai-27B-gguf dspark-bf16 (gen-1; Bonsai 2 ships NO drafter
    of its own - demo download_models.sh: "Bonsai 2 has no dspark drafter")
  - Converted via fork's gguf_dspark_to_dflash.py (branch dflash-converter-strip,
    fetched to bench-0918/gguf_dspark_to_dflash.py), --drop-shared-tensors,
    donor tokenizer = Bonsai2 PQ2_0 -> 77 tensors 2.1 GB -> llama-quantize Q4_0
  - Sidecar: drafter/Ternary-Bonsai-27B-dspark-dflash-Q4_0.gguf (592 MB, matches docs ~0.6 GB)
  - Enable: -md <sidecar> --spec-type draft-dspark --spec-draft-n-max 4 -ngld 999 -np 1
  - VERDICT (2026-09-18 live test, V100 beside prod): ENGAGEMENT FAILS. Sidecar loads and
    drafts (log: block_size=4, mask_token_id=248319, lineage=dspark) but acceptance is
    5/1096 = 0.46% on code prompts -> decode 9-11.4 t/s WITH vs 14.5 t/s WITHOUT = net -35%.
    As docs warn, drafters are target-specific; gen-1 drafter does not transfer to Bonsai 2,
    and Bonsai 2 ships no drafter of its own. BENCH BONSAI 2 PLAIN - sidecar not used.
    (Sidecar kept at drafter/ for provenance; demo-repo pipeline validated end-to-end.)

## B. CRACK ternary PQ2_0 (dealignai/Bonsai-2-27B-Ternary-CRACK-GGUF) - V100
- File: crack/Bonsai-2-27B-PQ2_0-CRACK.gguf (7.21 GB, byte-identical layout to A)
- Re-download notice 2026-09-17/18: earlier build had reasoning-mode token-loop bug;
  this download is post-fix (grabbed 2026-09-18).
- Lane: V100, same engine. Legs: speed/sanity/LCB/tau2 (+ tok/s A-vs-B delta).
- Vision mmproj (Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf) NOT staged - not needed for
  text benchmarks. Fetch from prism-ml repo if vision leg is wanted.

## C. CRACK 1bit PTQ1_0 (dealignai/Bonsai-2-27B-1bit-CRACK-GGUF) - CONDITIONAL
- File: crack1bit/Bonsai-2-27B-PTQ1_0-CRACK.gguf (5.95 GB)
- User gate: bench on 3060 ONLY IF B's CRACK does not decay substantially vs official
  Bonsai 2 27B. Decision after A+B results.
- If triggered: prior gen-1 Q1_0 (3.8 GB) fit 262K ctx in 9.7 GB on 3060 -> PTQ1_0
  5.95 GB will fit 128K+ comfortably. Engine: PrismML prebuilt (sm_86 real).

## D. Xing4.0-29B-A4B IQ4_NL (XingChen-AGI) - 3060 tok/s-only + V100 full legs
- Files: xing/xing4_0-29b-IQ4_NL-0000{1,2,3}-of-00003.gguf (20.10 GB total, IQ4_NL)
- Arch xing4_0 (MLA + MoE 64e/4a + mHC + NextN MTP; 41 blocks; 256K ctx natively).
  NOT in upstream llama.cpp. Engine: vendor fork branch xing4_0-port (PR #29012),
  cloned+built to ~/git/llama.cpp-xing4 (sm_70+sm_86 real cubins, build OK).
- CPU smoke: OK (loads, thinks, 5 t/s CPU). MTP head present (nextn_predict_layers=1)
  -> try --spec-type draft-mtp --spec-draft-n-max 3 after baseline.
- 3060: tok/s probe only (user directive, no full intel). Weights 20.1 GB > 12 GB VRAM
  -> needs --cpu-moe style offload OR partial -ngl; expect modest tok/s. If a 3060
  fit proves impractical, record the blocker per house rules.
- V100: full 4 legs. 20.1 GB weights + KV fits in 32 GB (stop prod first).

## E. K2-Horizon-7B-Uno (IFM) - 3060 Q4_K_M target lane
- USER LINK IS A PEFT LoRA ONLY (r=128, alpha=8192, 1.4 GB adapter) - no GGUF exists
  of base+Uno. Base = IFM/K2-Horizon-7B (9B MoE, arch k2-horizon, 36 blk, 512K ctx).
- Staged: k2/ = base 36 BF16 shards (14.4 GB) + Uno adapter + tokenizer + trust_remote_code
  files (modeling/config k2_horizon .py, needed by transformers).
- Shortcuts taken: (1) NANI-Nithin community Q4_K_M GGUF staged at k2-base-q4km/
  (5.59 GB; header says k2-horizon, general.name=Checkpoint_0002500 = same source ckpt
  as IFM official GGUF). (2) llama.cpp-k2horizon fork (MBZUAI-IFM) local build already
  had sm_70+sm_86 cubins; single-commit fork adding k2-horizon arch + chat template +
  diffusion-arch registration. Engine for k2-horizon GGUFs = this fork, not upstream.
- Uno application path: RESOLVED 2026-10-04 -> Path 2 (full merge), user decision.
  Rationale: scale-64 all-projection adapter (349M params, 7 targets x 36 layers)
  would tax the tok/s lane at serve time and break comparability with non-adapter rows.

### MERGE COMPLETE (2026-10-04): K2-Horizon-7B-Uno Q4_K_M BUILT, awaiting lane
- Artifact: /var/tmp/llms/bench-0918/k2-uno-q4km/K2-Horizon-7B-Uno-Q4_K_M.gguf
  5.59 GB, sha256 26446f5922376f97a287cf4ded8c52f68ba5728cc1726f5f3aab3ecd83da1497
- Pipeline: scripts/merge_k2_uno_full.py (peft merge_and_unload BF16 -> single
  safetensors -> k2horizon fork convert --outtype bf16 -> fork llama-quantize Q4_K_M)
- Fidelity checks: adapter keys required prefixing (raw file had non-PEFT keys,
  peft silently loaded ZERO adapter = no-op merge -> caught via missing-keys warning,
  fixed by rewriting keys with base_model.model. prefix, k2/adapter_model_prefixed
  since deleted); post-merge delta assert L0 q_proj max|merged-base| = 0.279 > 1e-3;
  327/327 tensors, no lora residue; Q4_K_M mixture 217x Q4_K + 37x Q6_K + 73x F32.
- CPU smoke (fork llama-cli): loads, 36.7 t/s decode; bare-template output is commas
  at temp 0 without --jinja - quality verdict deferred to the real lane (expected).
- GOTCHA for reruns: fork convert_hf_to_gguf rejects --outtype q4_k_m (valid choices
  f32/f16/bf16/q8_0/tq/only); must go bf16 gguf -> llama-quantize (requantize from
  q8_0 is DISABLED - a q8_0 intermediate is a dead end for Q4_K_M).
- Disk casualties (all re-downloadable): drafter/ sidecar (verdict already recorded),
  mtp-donor/ (graft done, sha recorded), NANI k2-base-q4km comparator, base shards,
  bf16 intermediates. Re-pull comparator before lane: NANI-Nithin HF.
- 3060 fit: 5.59 GB + KV. Measure KV at load; 128K if KV <= ~5.5 GB budget. Else V100.
- NOTE: Uno is a DIFFUSION decoding method (discrete diffusion drafting). The AR
  pathway runs as standard causal LM under llama.cpp; whether the diffusion pathway
  is even exercised outside the authors' vllm/xllm stack is unknown. Bench = AR
  pathway via this fork; record that caveat in results.

## F. Bonsai 2 PTQ1_0 + grafted Qwen3.8 MTP head (sudoingX/bonsai2-small-gpu) - V100
- User-directed: use the community MTP draft head instead of the failed gen-1 dspark sidecar.
- Repo: ~/git/bonsai2-small-gpu (graft tools, serve lines, sweeps). Recipe targets PTQ1_0.
- Staged: bonsai2/Ternary-Bonsai-2-27B-PTQ1_0.gguf (recipe sha 53107f53...),
  mtp-donor/Qwen3.8-27B-UD-Q4_K_M.gguf (recipe-pinned unsloth donor, sha 322e194f...;
  NOT Swift - finetune head would sit in a shifted hidden space).
- Graft (CPU-only, done at prep): extract_head.py (16 tensors incl. donor embed copy,
  expect sha 89a3144a...) -> merge.py (867 tensors, 7,012,820,512 bytes, expect sha
  83a396ee...) -> strip round-trip must reproduce the original PTQ1_0 sha.
- Serve arm (12gb-mtp.sh line, adapted -np/--port per lane):
  --spec-type draft-mtp --spec-draft-n-max 1 ONLY (n-max 2/3 net-negative on PTQ1_0
  small-batch economics: 3-token verify = 2.4 single steps). 131072 ctx, -ctk/-ctv q4_0.
  Community acceptance 0.51-0.95 (domain), +8% overall / +17% code on 3060; V100 TBD.
- IDENTITY CAVEAT (from their sweeps): greedy output differs between flag off/on at
  near-ties on stock kernels (PTQ1_0 CUDA not batch-invariant). Bench = throughput rows;
  record as speed arm, not lossless, unless we build the kernel branch.
- BONUS ARM (their pr-ptq1-mmv kernel branch, PrismML PRs #217/#218): 1.53x decode flag-off
  (40.5 vs 26.3 tg128), byte-identical greedy WITH GGML_CUDA_BATCH_INVARIANT=1 -> genuinely
  lossless MTP (50.1 tok/s on 3060 @131K). Fork cloned to ~/git/llama.cpp-sudoingx (bonsai2
  branch, both patches stacked) and BUILT sm_70+sm_86 real cubins (llama-server/bench/cli,
  2026-09-20; build/bin/). F-arm C ready to run without further compilation.
- Also: PTQ1_0 doubles as the OFFICIAL comparator for the 1bit-CRACK gate (same packing family).

## Execution order (when user says go)
1. B on V100 (stop Carnice) -> A on V100 -> A+B delta, CRACK-decay verdict
2. F arms on V100: PTQ1_0 baseline + mtp n-max 1 (prebuilt) [+ kernel-branch arm if built]
3. D on V100 (full legs) -> D 3060 tok/s probe (needs Nail stopped)
4. E on 3060 (needs Nail stopped; 128K check first) - decide merge vs lora path
5. C only if B-vs-PTQ1_0 verdict is no-substantial-decay

---

# ACTIVE QUEUE (2026-09-08)

# ACTIVE (2026-09-09): PROD DEPLOY - Carnice-V3 (V100 orchestrator) + Nail 35B-A3B (3060 coder/reviewer)
State: LIVE as of 2026-09-09 ~21:05 UTC. Services: caimlas-carnice (V100, :8081, enabled at boot;
replaced caimlas-qwythos, now disabled) and caimlas-nail (3060, :8080, enabled at boot).
- Carnice-V3 Q4_K_M: gpu-layers 99, 262k ctx, batch 2048/ubatch 512, t8, kv q4_0, parallel 8,
  kv-unified, cont-batching, temp 0.7. Lane numbers: tau2 0.5333 (8/15, post-audit), LCB 77.3,
  31.5 t/s. No MTP variant of this finetune exists.
- Nail-Qwen3.6-35B-A3B UD-Q4_K_XL: n-cpu-moe 28 + no-mmap, 262k ctx, batch 2048/ubatch 512, t6,
  kv q8_0, parallel 2, kv-unified, reasoning off. Unit files: deploy/caimlas-{carnice,nail}.service.
- Batching sweeps (scripts/nail_batch_sweep.py, logs in /tmp/coding-bench/logs/nail_sweep_*):
  parallel axis at full cpu-moe: agg t/s saturates ~23.3-25.0 for p=2..8 while p95 scales ~linearly
  (cpu-moe is RAM-bandwidth-bound; single stream 17 t/s). Threads: 6 = 8 > 4. Partial-expert
  residency (--n-cpu-moe N, first N layers' experts on CPU, REST on GPU): N=40 no-op, 36/32/28 =
  25.4/27.9/30.3 agg, N=24 OOM (KV cache alloc 2720 MiB failed) -> N=28 is the fit boundary.
  --no-mmap A/B at N=28: 24.0 single / 31.9 agg (+11%/+5%) -> adopted. Net vs all-CPU baseline:
  single +41%, aggregate +37%, p95 24s.
- NOTE: historical "85 t/s" Nail figure was an 8K-ctx full-offload probe (ecc retest config), NOT
  the prod config; do not compare against prod numbers. See hardware-tok-s-retests.md.
- NOTE: --n-cpu-moe semantics = first N layers on CPU (not a GPU-pin count). Passing it together
  with --cpu-moe is a silent no-op of the partial flag (cost one wasted ladder).

## 1. Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic Q4_K_M — READY, needs download
- Source: https://huggingface.co/medismera/Qwen3.8-27B-OBLITERATED-Mythos-Class-Agentic
- Check repo for a Q4_K_M GGUF; if only safetensors, quantize locally (disk: ~38 GB
  free, prune before). ~16 GB target on V100 full offload.
- Full lane (speed probe -> sanity:25 -> LCB 75 -> tau2). Thinking suppression
  required: verify enable_thinking:false live pre-LCB (qwen38 already in
  oai_runner.py allowlist); tau2 agent served with --chat-template-kwargs
  '{"enable_thinking": false}' per 2026-09-08 empty-turn fix.
- Compare against: Qwen3.8-27B family (LCB 0.80-0.88, tau2 0.4667-0.50) and
  Qwythos-9B-Mythos (LCB 0.587, tau2 0.40).

## 2. MiniCPM5-2B tau2 3-seed average — optional, decided-against for now
- 0.571 (3060) vs 0.333 (V100) traced to bistable agent loops; a 3-seed mean
  would give one defensible number (~45 min). Parked unless requested.

## Blocked / parked
- MiniMax-H3: video/VL models, not text-benchmarkable as staged.
- Escha-W2: eschamoe runtime needs sm_80+; V100 sm_70 hard-blocked.

## Harness debt (non-urgent)
- Nail/Ornith 262K cpu-moe create_context failure on current llama.cpp build.
- Remaining 1-2-infra-error tau2 rows: score movement from a sweep would be
  within noise; skip unless a specific row matters.
- LFM-empty LCB empties (Tier 3 diagnosis): template-level, needs LCB prompt
  adaptation, not a rerun. Parked.

---

# ARCHIVE (all items below executed/closed; kept for provenance)

# CLOSED (2026-09-06/07): LCB empty-content audit + ECC tok/s + new models

1. **LCB empty-content remediation** - DONE, commit b0fb999. Qwen3.8 family
   +9-17pp across 7 variants; Heretic-35B and LFM trio diagnosed as REAL
   (not artifacts); 5 deleted-file models carry invalidation notes.
2. **ECC tok/s re-tests** - DONE, commit c7761a3. Verdict: +0% to +5.4%,
   within control-lane noise. See ecc_retest_results.md.
3. **Bonsai dspark retry** - DONE. WORKS: 37.69 t/s (prior 36.3, +3.8%).
   Root cause of earlier failures: caimlas-qwythos holding V100 VRAM, NOT
   fit-logic conflicts. Recipe: PrismML fork + `-fit off`, V100 must be
   exclusive (stop production service first). Result in progress.json
   `Ternary-Bonsai-27B Q2_0 (dspark)` -> ecc_dspark_retry.
4. **DogukanUrker-BTL-4 Q4_K_M** (community quant from tweet) - DONE.
   V100: 91.7 t/s, sanity 92%, LCB 0.92 (0 empty), tau2 0.40 - statistically
   indistinguishable from our badtheorylabs build (88.8/92%/0.92/0.20-ish).
   The upstream Q4_K_M is healthy on V100 full-offload; no --n-cpu-moe needed.
5. **MiniCPM5-2B Q8_0** (openbmb) - DONE on BOTH GPUs per user config
   (131K ctx, f16 KV, temp 1.0/top-p 0.95 serving; temp 0.0 for benches).
   3060: 112.7 t/s, sanity 72%, LCB 0.573 (0 empty), tau2 0.571.
   V100: 166.0 t/s, sanity 64%, LCB 0.520 (0 empty), tau2 0.333.
   Read: fast little model; quality mid-pack at 2B; sanity/LCB/tau2 deltas
   between GPUs are run variance, not silicon (same quant). Note:
   openbmb/MiniCPM5-2B-DSpark exists (official draft model) - unexplored.

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

---

# STAGED (2026-10-04): Victoria (rmonsurate/Victoria) — QUEUED, not downloaded

- Source: https://huggingface.co/rmonsurate/Victoria
- What it is: 44%-expert-pruned (512→288/layer) + QAT 4-bit retrain of
  Qwen/Qwen3.8-Flash-Next. MoE, 5.9B active. GGUF arch = `qwen4exp`,
  262144 ctx natively. Claimed: Terminal-Bench 2.1 70.0% avg@3 (nvfp4 build)
  / 75.28% single-run (GGUF build), HumanEval 93.2 avg@5 (GGUF build).
  Note: GGUF and NVFP4 are DIFFERENT checkpoints; scores are per-build.
- License: qwen-community-1.0 (other).

## Builds
- nvfp4/: vLLM, NVIDIA Blackwell (B200/B300) ONLY -> HARD-BLOCKED here
  (V100 sm_70 / 3060 sm_86), same blocked class as 27B EXL3. Ignore.
- gguf/: "bitexact" Q4_K_M-class, includes MTP draft head as 32 extra
  tensors. Two sets, same resident VRAM:
  - full-LUT: victoria-s410-bitexact-0000{1,2,3}-of-00003 (155.20 GB dl)
  - tbl8:     victoria-s410-bitexact-tbl8-0000{1,2,3}-of-00003 (107.20 GB dl,
    LUT stored 8-bit; README says GPU memory use is the same) <- TAKE tbl8

## BLOCKER: arch not in mainline or any of our forks
- Mainline llama.cpp REFUSES the files ("expected 1256, got 1224 tensors") —
  the draft head is extra. Engine = rmonsurate's build:
  - prebuilt: hf.co/rmonsurate/llama.cpp-victoria (Linux CUDA zip; stock
    b11276 + one draft-head patch)
  - or build: github.com/rmonsurate/llama.cpp branch qwen4exp-draft-mtp
  - (vendor-fork scan rule applies: patch is one commit, review before running)
- Not covered by xing4/k2-horizon/prismml forks we already have.

## Fit
- Weights 49.17 GiB -> V100 (32GB) needs partial/moe offload (n-cpu-moe
  style); 3060 (12GB) would be CPU-dominant. Same fit problem class as
  Nail-35B on 3060. No obvious VRAM-fit lane without offload losses.
- Serve lines from card (CUDA UNMEASURED by vendor — M3 Max numbers only):
  - head on:  -fa on -c 8192 --spec-type draft-mtp --spec-draft-n-max 3
  - head off: -fa on -c 8192
  - MTP n-max 3 caveat: our F-arm data says n-max >1 can be net-negative on
    small-batch economics; A/B both modes.

## Download (when user says go; /home near full -> /var/tmp/llms)
```
hf download rmonsurate/Victoria --include "gguf/victoria-s410-bitexact-tbl8-*" --local-dir /var/tmp/llms/victoria
```
(107.20 GB; 3 shards 41.56+54.40+11.24 GB)

## Lane (per house pattern, prep-only until execution approved)
1. Stage download + vendor prebuilt, CPU-side header verification only.
2. Fit probe on V100 beside prod first; if no fit, stop-prod lane decision
   belongs to user.
3. Full 4 legs: speed probe -> sanity:25 -> LCB 75 -> tau2 airline/15/seed42/
   conc2 w/ LFM user sim on non-agent GPU.
4. Comparators: Qwen3.8-27B family (LCB 0.80-0.88, tau2 0.4667-0.50) — same
   base lineage; BTL-4 Q4_K_M (agentic class, LCB 0.92, tau2 0.20-0.40).
5. Record MTP on/off as separate speed rows; identity caveat per F-arm
   (batch-invariance) if we use the kernel flags — here vendor build, note only.

---

# CLOSED (2026-10-10): Qwen3.8 Flash Next GSQ-RCO Q2_0 — V100 lane COMPLETE
# (was ACTIVE): Qwen3.8 Flash Next GSQ-RCO Q2_0

- Source: ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF (Q2_0, 2 shards, 66.4 GB total)
  - Shard 2 is the 51.2B-param n-gram table (per_layer_token_embd, IQ4_NL) — identical
    across all GSQ-RCO variants; Q2_0 picked for speed (ISTA: 3.4x prompt t/s vs IQ2_XS,
    avoids LUT formats).
- Files: /home/files/llms/flashnext-q20/ (post-HF-cache-purge, /home back to 117G free;
  purged talkie-lm trio 132G, re-downloadable from own repos).
- Arch: qwen4exp (512 experts/10 active, 48 layers, hybrid linear+full attention idx,
  PLE n-gram heads, 262K ctx). GGUF header verified via HTTP range read.
- Engine: ~/git/llama.cpp-xing4 fork (qwen4exp support confirmed in built libllama.so;
  upstream master also has it now). NOT the rmonsurate Victoria build.
- 3060: IMPOSSIBLE — smallest build 66.4 GB vs 12 GB VRAM / 37 GB RAM. Recorded blocker.
- V100 plan: probe script scripts/probe_flashnext_v100.py — config ladder cpu-moe ->
  cpu-moe-ngl48 -> n-cpu-moe-48, speed + coherence smoke. Full LCB/tau2 lane only if
  decode is workable; record blocker otherwise per house rules.

---

# CLOSED (2026-10-10): Underdog-Saluki-27B-1.0 IQ2-mix — BOTH lanes COMPLETE
# (was QUEUED/ACTIVE): Underdog-Saluki-27B-1.0 IQ2-mix (3060 + V100)

- Source tweet: https://x.com/UnderdogAI/status/2108021482983133395
- Repo: https://huggingface.co/ConwayResearch/Underdog-Saluki-27B-1.0
- File: Underdog-Saluki-27B-1.0-IQ2-mix.gguf (7.36 GB, IQ2-mix + imatrix,
  GSQ-RCO lineage, quantized_by ISTA DASLab)
- GGUF header verified (ranged read): arch qwen35 (dense 64L — NOT the qwen4exp
  Flash Next arch), 262144 ctx, quant v2. Stock llama.cpp engine.
- Claimed (their harness): BFCL-ish 88/120 vs 84 full-size; parallel tool calls
  42 vs 35; SWE-b V 30 vs 33; AIME25 79.2 vs 96.7. Tool-calling-focused tune.
- Thinking: on by default, kwarg-switchable (same as Qwen3.8 family).
- Allowlist: 'saluki' added to oai_runner.py LCB_DISABLE_THINKING list
  (2026-10-10, before first LCB leg).
- Download staged to /home/files/llms/saluki/ (mmproj NOT staged — text benches
  only; fetch if a vision leg is wanted).
- Lane plan (BOTH GPUs, per user directive):
  - 3060: full lane — speed probe -> sanity:25 -> LCB 75 thinking-off ->
    tau2 airline/15/seed42/conc2 w/ LFM user sim on V100. Fit: 7.36 GB weights
    + 65K q4_0 KV ~ 9.4 GB, fits 12 GB (same class as BTL-4 IQ2_XXS 9.78 GB).
  - V100: speed probe + LCB + tau2 (agent on V100, user sim on 3060) for the
    cross-GPU comparison row; full offload trivial.
- Comparators: Qwen3.8-27B family rows (LCB 0.80-0.88, tau2 0.4667-0.50);
  BTL-4 IQ2_XXS 3060 row (72.9 t/s, tau2 0.38) as the 2-bit-on-3060 class.
- Status: downloading (7.36 GB). Run after Flash Next lane completes (V100 busy).

---

# EXECUTED (2026-10-10): Flash Next + Saluki lanes — COMPLETE

- Flash Next GSQ-RCO Q2_0 (V100, n-cpu-moe 8, xing4 fork): 13.27 t/s, sanity 84%,
  LCB 94.7% (71/75, 0 empty, report-best LCB), tau2 0.0667 (1/15, 2 infra).
  Fit ladder recorded: full offload OOM 37.3GB / cpu-moe 1.63 / N24 5.7 / N16 8.02 /
  N8 12.21-13.27 (31.1GB VRAM). MTP head absent from quant. LCB needs
  --openai_timeout 900 (300s default aborts at 7.6 t/s decode dips; attempt-2
  failure preserved in failures[]). Verdict: elite 2-bit coder, unusable agent
  in this config. 3060 hard-blocked (66.4GB smallest build) — blocker stands.
- Saluki IQ2-mix: 3060 = 20.41 t/s, 84%, LCB 64.0%, tau2 0.3333.
  V100 = 28.69 t/s, 84%, LCB 74.7%, tau2 0.20.
  Same-quant GPU delta LCB 64->74.7 (+10.7pp) mirrors the MiniCPM5-2B
  bistable-agent variance class; tau2 0.333(3060) vs 0.20(V100) likewise
  (2 infra errors each). Vs Qwen3.8-27B family (LCB 80-88, tau2 0.47-0.50):
  2-bit IQ2-mix costs real codegen + agent depth; tool-calling tune (vendor
  BFCL claim) does not transfer to tau2 conversational loops.
- LCB aliases registered: local/flashnext-qwen38-q20, local/saluki-27b-iq2mix,
  local/saluki-27b-iq2mix-v100. Allowlist tokens: flash-next/flashnext/saluki.
