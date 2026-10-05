# borg lane results — Radeon AI PRO R9700 + Ryzen AI MAX+ 395 (Strix Halo)

Reference run of `scripts/bench_borg_lanes.py` on a single host with two AMD
devices: a R9700 (32 GB GDDR6, gfx1201) and a Ryzen AI MAX+ 395 / Strix Halo
(128 GB unified, gfx1151), ROCm 7.2.1. Measured 2026-10-05.

## Why this lane exists

The other lanes in this repo assume CUDA device names and this repo's own
`llama-server` builds. On this host the models run on several different engines,
so one registry covers them all and the engine is a recorded column, which keeps
the comparison like-for-like: same prompts, same sampling, same 2-runs-per-prompt
protocol, one row per model.

| Engine | Models | Notes |
|---|---|---|
| luce_server (Lucebox) | Qwen 3.8 27B, Qwen 3.8 27B Vision, Laguna XS 2.1 33B, DeepSeek V4 Flash, DeepSeek V4.1 Flash | HIP build; OpenAI + Anthropic APIs, no `/completion` route |
| paoai-strix-engine (llama.cpp fork, Vulkan) | GLM-5.3-Flash (ROCmFP4 Strix quant) | glm5next architecture; must pin `GGML_VK_VISIBLE_DEVICES=1` on a 2-GPU host |
| Strata (HIP) | Qwen 3.8 Flash Next (IQ3_S) | reasoning must be disabled per request for throughput |

## How to run

```bash
python3 scripts/bench_borg_lanes.py --models all      # measure + print the table
python3 scripts/bench_borg_lanes.py --table           # print the table from the JSON
python3 scripts/bench_borg_lanes.py --models qwen38-27b,laguna-xs21
```

Results are written to `/root/bench/bench_results_borg.json`. Launcher scripts
live in `/root/bench/serve_*.sh` on the host; the launcher path is part of the
model registry, so a new engine is added by adding one registry row.

**Run only one lane at a time.** Each lane stops any server it finds before it
starts its own, so two concurrent lane processes kill each other's servers
(observed: a run left a defunct server and burned its full health timeout).

## Columns

| Column | Meaning |
|---|---|
| prefill cold | prompt processing, cold (no prefix cache), short prompt |
| prefill 2.5K | prompt processing, cold, ~2.5K-token prompt (the fair prefill number) |
| decode sampled | harness default, temperature 0.3 |
| spec (sampled) | did the engine run speculative decode at temperature 0.3 |
| decode greedy | temperature 0 — the production path, and where drafters fire |
| spec (greedy) | drafter state on the greedy path |
| accept | mean draft acceptance on the greedy path |
| boot | model load (seconds) |

## Results

| model | engine/server | prefill cold | prefill 2.5K | decode sampled | spec | decode greedy | spec | accept | boot |
|---|---|---|---|---|---|---|---|---|---|
| qwen38-27b | luce_server | 491.22 | 1114.82 | 33.05 | no | **136.57** | yes | 0.504 | 6.0 |
| qwen38-27b-vision | luce_server | 469.32 | 1173.88 | 36.03 | no | **138.10** | yes | 0.472 | 8.0 |
| laguna-xs21 | luce_server | 1128.73 | **2054.86** | 78.53 | no | **142.51** | yes | 0.673 | 6.0 |
| ds4-flash | luce_server | 177.88 | 366.55 | 48.97 | yes | 51.72 | yes | 0.732 | 42.0 |
| ds41-flash | luce_server | 52.58 | 67.27 | 23.86 | yes | 26.56 | yes | 0.812 | 46.1 |
| glm53-flash | paoai-strix-engine | 63.50 | 66.59 | 15.33 | no | 15.07 | no | 0.0 | 24.1 |
| flashnext | Strata | 184.14 | 446.93 | 75.78 | n/a | **88.71** | n/a | n/a | 42.1 |

All values are tokens/second.

## Reading the table

1. **Speculative decoding is per-engine, not global.** Qwen's DFlash2 only
   engages on greedy requests (plain autoregressive at temperature 0.3, ~4x
   slower); DeepSeek's DSpark speculates at sampled temperatures too. So the
   sampled column is not "the engine at its best" for every model.
2. **Acceptance, not just the switch, moves the number.** Qwen 3.8 27B greedy is
   136.6 t/s on these prompts (accept 0.504) but 228.7 t/s on greedy HumanEval
   code at accept 0.824. Prompt content changes the drafter's hit rate by ~1.7x.
3. **Prefill depends on prompt length.** The same Qwen 3.8 27B config measures
   491 t/s at ~500 tokens and 1115 t/s at ~2.5K; fixed per-request overhead
   dominates short prompts. Use the 2.5K column for comparisons.
4. **Spec column for non-lucebox engines** reports what the API exposes.
   Strata and llama.cpp-based servers do not report `spec_decode_ran`, so `n/a`
   means "not reported", not "not running" (Strata runs MTP internally).
5. **20-30 tok/s floor.** Two rows sit below it (`ds41-flash` 26.6,
   `glm53-flash` 15.1). They remain servable but are not benchmark-suite
   candidates on this host.
6. **Flash Next requires `reasoning_effort=off`.** With reasoning on it measures
   68 t/s; the lane sends the flag per model and records the extras used.

## Pending lanes

- **Kolibri-1** (Aleph Alpha, 78B MoE / 3.46B active): community Q4_K_M GGUF plus
  a patched llama.cpp (`kolibri1` architecture) built for gfx1201+gfx1151.
  Community CPU-only reference is 12-15 t/s decode; the GPU result is untested
  upstream.
- **GLM-5.3-Flash EXL3 (Kyojin)**: ROCm/ExLlamaV3 engine for gfx1151 with a
  99.7 GB EXL3 pack. The engine authors publish 26-30 t/s decode and ~580 t/s
  prefill at that pack; the same model measures 15.1 t/s through the
  paoai/Vulkan lane above, so this is the direct engine comparison.
