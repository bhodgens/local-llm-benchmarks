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
| prefill 1st req | prompt processing on the FIRST request after load — includes kernel/buffer warmup, so it is systematically low; recorded to make that visible, not for comparison |
| prefill cold | prompt processing, uncached, short prompt, after one discarded warmup request |
| prefill 2.5K | prompt processing, uncached, ~2.5K-token prompt (the fair prefill number) |
| decode sampled | harness default, temperature 0.3 |
| spec (sampled) | did the engine run speculative decode at temperature 0.3 |
| decode greedy | temperature 0 — the production path, and where drafters fire |
| spec (greedy) | drafter state on the greedy path |
| accept | mean draft acceptance on the greedy path |
| boot | model load (seconds) |

## Results

| model | engine/server | prefill 1st req | prefill cold | prefill 2.5K | decode sampled | spec | decode greedy | spec | accept | boot |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen38-27b | luce_server | 406.79 | 838.89 | 1081.84 | 32.73 | no | **138.63** | yes | 0.517 | 8.0 |
| qwen38-27b-vision | luce_server | 444.97 | 980.92 | 1149.45 | 36.06 | no | **127.70** | yes | 0.433 | 8.0 |
| laguna-xs21 | luce_server | 1451.66 | 2139.81 | **2237.28** | 78.34 | no | **129.69** | yes | 0.629 | 6.0 |
| ds4-flash | luce_server | 214.81 | 256.44 | 372.80 | 50.24 | yes | 53.51 | yes | 0.696 | 54.5 |
| ds41-flash | luce_server | n/a | 52.58 | 67.27 | 23.86 | yes | 26.56 | yes | 0.812 | 46.1 |
| glm53-flash | paoai-strix-engine | 67.46 | 73.35 | 75.19 | 15.57 | no | 14.25 | no | 0.0 | 26.1 |
| flashnext | Strata | 244.95 | 288.61 | 441.60 | 77.05 | n/a | **86.31** | n/a | n/a | 22.0 |
| kolibri-1 | llama.cpp-kolibri (patched) | 248.76 | 282.58 | 438.51 | 54.62 | no | **54.32** | no | n/a | 16.0 |

All values are tokens/second.

**The warmup column is not noise.** The first request after a load measures
2.1x lower prefill on Qwen 3.8 27B (407 -> 839), 2.2x on the Vision variant
(445 -> 981) and 1.5x on Laguna (1452 -> 2140). A lane that measures only the
first request reports roughly half the engine's real prefill rate. The lane
therefore discards one warmup request before measuring.

### Separate lanes with their own harness (not the registry above)

| model | engine | prefill | decode | notes |
|---|---|---|---|---|
| glm-5.3-flash EXL3 | Kyojin (ExLlamaV3 ROCm, gfx1151) | 409.60 @4.5K | 33.73 | `scripts/bench_kyojin_glm.py`; first run measured 126 prefill / 27 decode because the engine builds a dense-GEMM tune cache for ~13 min on first start — re-run warm is the number above |

### Kolibri-1 expert placement (`--n-cpu-moe`)

`--n-cpu-moe N` keeps the MoE tensors of N layers on the CPU; a smaller N moves
more experts onto the GPU. The model is 47.5 GB against 32 GB of VRAM, so only
part of it fits. Decode and prefill both improve as experts move onto the card:

| `--n-cpu-moe` | decode t/s | prefill t/s @2.6K | VRAM used |
|---|---|---|---|
| 999 (all experts on CPU) | 39.8 | 235.0 | — |
| 40 | 42.1 | 276.8 | — |
| 28 | 49.9 | 370.4 | — |
| 26 | 50.9 | — | 25.4 GB |
| 24 | 52.6 | — | 27.1 GB |
| 22 (chosen default) | **54.1** | — | 28.9 GB |

+36% decode over the all-CPU default, with ~3 GB of headroom left. Lower values
risk an allocation failure: an earlier `--n-cpu-moe 16` arm produced no
measurement.

### Prompt chunking (Strata, Flash Next) — prefill gain was an artifact, decode cost is real

An initial sweep suggested `STRATA_PREFILL_AUTO_MAX=32768` bought +59% prefill at
~28K prompts. A controlled A/B on the same server, same session, says otherwise:

| | decode t/s (mean of 3) | prefill @2.4K | prefill @27.7K |
|---|---|---|---|
| chunk cap 8192 (engine default) | **~65** | 454.3 | 1470.3 |
| chunk cap 32768 | ~37 | 455.7 | 1469.2 |

Prefill is unchanged at both sizes (the earlier "+59%" came from arm ordering —
the first arm ran against a cold engine). Decode drops **43%**, because larger
chunks borrow memory from the expert cache. **Keep the engine default.** The
launcher's `STRATA_PREFILL_AUTO_MAX` override was reverted.

### Separately: Strata's n-gram table I/O mode (real win)

| mode | prefill @29K | prefill @116K |
|---|---|---|
| `ram` (table locked in RAM) | **1266.8** | **1129.8** |
| `direct` (engine default, unbuffered SSD) | 1240.2 | 966.5 |
| `mmap` | 891.0 | 986.9 |

`--ple-io ram` is set in that host's Strata config: +17% prefill at 116K.

## Which models are worth running on this hardware

Ranked by value for coding and planning work, using the measured numbers above
plus the deployment facts that matter (boot time, license, memory footprint).

### Tier 1 — run these

| Rank | Model | Why | Numbers |
|---|---|---|---|
| 1 | **Qwen 3.8 27B** (Vision variant if images) | Highest decode + prefill combination, 6-8 s boot, fits entirely in the 32 GB card, quality matches an 8-bit reference on HumanEval (151/164) and GSM8K (177/200) at the quant we run | 138.6 decode, 1082 prefill @2.5K, 131K ctx |
| 2 | **Laguna XS 2.1 33B** | Best prefill of anything measured here, 6 s boot, coding-focused lineage, long-context design | 129.7 decode, **2237** prefill @2.5K |
| 3 | **Kolibri-1** | Apache-2.0 (cleanest license of the set), 16 s boot, 1M-token context design, tool calling, German+English; 54 t/s is plenty for agent loops | 54.3 decode, 439 prefill @2.5K |

### Tier 2 — run for their specific strengths

| Model | Use it for | Numbers | Cost |
|---|---|---|---|
| **Qwen 3.8 Flash Next** (Strata) | Long-context agent work: 131K ctx (262K capable), vision, tools, strongest tool-calling scores in the community benchmark we reviewed | 86.3 decode, 442 prefill @2.5K | 84 GB, 22 s boot |
| **DeepSeek V4 Flash** | Deep reasoning / hardest planning steps; uses both devices | 53.5 decode, 373 prefill | 98 GB, 55 s boot |
| **GLM-5.3-Flash via Kyojin EXL3** | Only if GLM specifically is wanted; the EXL3 engine is worth 2.2x over the GGUF route on the same model | 33.7 decode, 410 prefill @4.5K | 93 GB pack, ~13 min first-run tune |

### Tier 3 — keep the weights, do not build on them here

- **DS4.1 Flash**: 26.6 t/s. Its own published reference is 25.4, so this is the
  model's ceiling on this class of hardware, not a tuning miss.
- **GLM-5.3-Flash (ROCmFP4 GGUF / Vulkan)**: 14.25 t/s. Use the Kyojin EXL3 route
  instead. It also shows no speculative decoding through that engine (accept 0.0)
  even with MTP enabled in the launch.
- **GLM-5.3 proper (753B)**: gate-failed. At ~8.3 GB of expert reads per token
  against 3.9 GB/s measured NVMe, it projects sub-1 t/s.

## How to get the best out of the box

1. **Treat it as a two-tier machine.** The R9700 (32 GB GDDR6, PCIe Gen5 x16) is
   the fast tier; the Ryzen AI MAX+ 395 (124 GB unified) is the capacity tier.
   Everything above 32 GB spills into the slow tier, so pick models whose hot set
   fits the card.
2. **Run one large model at a time.** Two big models fail to allocate — verified
   twice, including `cudaMalloc failed: out of memory` when a second engine held
   the card. RAM is shared between the iGPU's model and the host.
3. **Swap by workload, not by wish.** Boot times are 6-26 s for the Tier 1-2
   models except DS4 (55 s), so a router that loads on demand costs seconds, not
   minutes. Keep Qwen 27B (8 s) and Laguna (6 s) as the hot pair.
4. **Match the engine to the model, not the other way round.** The same GLM
   weights run 2.2x faster under Kyojin/EXL3 than under the GGUF/Vulkan engine;
   the same Flash Next weights run 2.7x faster under Strata than the best
   llama.cpp path reported elsewhere. Engine choice moved results more than any
   sampling or flag change measured here.
5. **Enable speculative decoding and verify it engaged.** Qwen's DFlash2 only
   fires on greedy requests (33 -> 139 t/s, 4.2x); DeepSeek's DSpark fires at
   sampled temperatures too. On any new engine, check `spec_decode_ran` and the
   accept rate rather than trusting the launch flags.
6. **Budget the capacity tier deliberately.** Flash Next 84 GB, GLM EXL3 93 GB,
   DS4 98 GB — each fills most of the 124 GB. Keep ~20 GB free for KV cache and
   page cache, or prefill and expert streaming will thrash the SSD.
7. **Skip the below-floor models in production.** They are servable, but two of
   the three engines behind them are slower than the hardware is capable of.

### Suggested production layout

```
R9700 (fast tier)                     Strix Halo (capacity tier)
  Qwen 3.8 27B [+Vision]   138 t/s      Qwen 3.8 Flash Next   86 t/s   (on demand)
  Laguna XS 2.1            130 t/s      DeepSeek V4 Flash     54 t/s   (on demand)
  Kolibri-1                 54 t/s      GLM-5.3-Flash EXL3    34 t/s   (on demand)
```
One resident fast model plus one on-demand capacity model; swap the capacity
model when the workload changes.

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

Both previously-pending lanes have now run (see the two tables above).

Notes for whoever repeats them:

- **Kolibri-1** (Aleph Alpha, 78B MoE / 3.46B active): needs a patched llama.cpp
  for the `kolibri1` architecture. Apply the community patch with
  `git apply --3way` — `git am` fails without a committer identity, and
  `git fetch --depth 1 origin <sha>` cannot fetch an arbitrary commit. Verify the
  patch with `strings build/bin/libllama.so | grep -c kolibri1`, NOT the
  `llama-server` executable: that is a ~17 KB dispatcher and will show 0 either
  way. The 47.5 GB Q4_K_M does not fit one 32 GB card, so experts are offloaded
  (`--n-cpu-moe`); measured 39.2 t/s decode against a community CPU-only
  reference of 12-15 t/s.
- **GLM-5.3-Flash EXL3 (Kyojin)**: the engine's MTP path needs an unquantized
  `eh_proj` sidecar that the EXL3 pack does not contain (it ships the tensor
  quantized as suh/svh/mul1/trellis). Extract it with the repo's own
  `tools/glm/mtp_eh_sidecar.py` from the source checkpoint; the tensor lives in
  `model-00001-of-00062.safetensors` (5 GiB), so only that shard is needed, not
  the 642 GB checkpoint. On a two-GPU host also set `HIP_VISIBLE_DEVICES=1` and
  `EXL3_HSA_LIB` to the ROCm SDK's own `libhsa-runtime64.so.1` — the system ROCm
  7.2 HSA is too old for the 7.13 nightly torch and fails with
  `hsa_ext_image_create_v2` undefined.
