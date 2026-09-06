# Fuse-2-MoE (Akahsizrr) Benchmark - 2026-09-02

**Verdict: MODEL BROKEN - do not deploy. Both BF16 and Q4_K_M are incoherent.**

## Setup
- Patched llama.cpp build (Fuse4 `ffn_moe_norm` + `build_layer_ffn_fuse4`): `~/git/llama.cpp-fuse4` (worktree off upstream 62bf73d25, CUDA sm_70;86)
- Downloads: `~/models/fuse2/Fuse-2-MoE-{BF16,Q4_K_M}.gguf` (17.75GB / 5.72GB)
- LCB registered: `local/fuse2-bf16`, `local/fuse2-q4`; "fuse" added to oai_runner thinking pattern list

## Speed (patched build, -ngl 99, -fa 1, 3 runs)

| Config | GPU | pp512 t/s | tg128 t/s | chat decode t/s | VRAM |
|--------|-----|-----------|-----------|-----------------|------|
| BF16 | V100 32GB | 625.5 | 60.3 | 49.2 (32K ctx) | ~19GB |
| Q4_K_M | 3060 12GB | 1952.5 | 67.8 | 64.7 | ~5.5GB |

Both configs are bandwidth-bound as expected. Load is clean (no missing tensors) - the patch works.

## Quality: BROKEN in both quants
- Chat completions: `[Start thinking]` channel produces looped nonsense ("The function is a to add a function to.."), `content` empty or looped.
- Raw completions (template bypassed): same loop-collapse within 1-2 sentences ("The capital of France is located in Paris, which is in the north of the region?" repeated).
- Sampling rescue (repeat_penalty 1.15-1.3, min_p, temp 0.7-0.8): degrades to word salad, no recovery.
- Knowledge fragments survive (Paris/capitals, Python shapes) - weights are not random; generation dynamics are broken.

## Root-cause chain
1. BF16 (direct conversion, no quant error) fails IDENTICALLY to Q4 -> damage is upstream of quantization.
2. Author's README: GGUF converted from base `fuse-2-boosted` checkpoint; **the fine-tune was lost**.
3. README also documents 4 conversion approximations: softmax routing instead of sqrtsoftplus (-2.0 bias + weight norm), no 10% expert-delta residual clamp, no expert-output scale matching, static 0.0184 gate*scale constant. Broken routing dynamics over an untrained fusion = observed behavior.

## Suite decision
LCB/tau2/HumanEval NOT RUN - protocol skip (model cannot produce parseable output; suites would burn 1.5-2h each scoring ~0). Registrations kept in place in case the author ships a fixed GGUF/fine-tune.

## Artifacts
- Server/probe logs: `/tmp/fuse2_bf16_server.log`, `/tmp/fuse2_q4_server.log`, `/tmp/fuse2_bench_*.log`
- Orchestrator (ready for a fixed model): `~/llm-benchmarks/scripts/run_fuse2.py`
