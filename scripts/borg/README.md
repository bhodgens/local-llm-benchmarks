# scripts/borg — launchers and probes for the borg host (R9700 + Strix Halo)

These are the exact scripts used to produce the numbers in `BORG_RESULTS.md`.
They live here so the lane runs are reproducible; `scripts/bench_borg_lanes.py`
is the entry point and reads its model registry from its own `MODELS` dict.

## Layout on the host

| Path | What |
|---|---|
| `/root/bench/` | everything below (launchers, probes, results) |
| `/root/models/` | model weights by family |
| `/root/lucebox/` | Lucebox server, HIP build (`server/build-hip/luce_server`) |
| `/root/llama.cpp-kolibri/` | patched llama.cpp for the `kolibri1` architecture |
| `/root/kyojin/` | Kyojin (ExLlamaV3 ROCm) engine + venv |
| `/root/strata/` | Strata engine for Qwen 3.8 Flash Next |

## Launchers (one per model, all start a server and print nothing until healthy)

| Script | Model | Engine | Port |
|---|---|---|---|
| `serve_qwen3827b.sh [kv] [block]` | Qwen 3.8 27B | luce_server | 8901 |
| `serve_qwen3827b_kvflash.sh <ctx> <kvflash>` | Qwen 3.8 27B | luce_server | 8901 |
| `serve_qwen38vision.sh [kv] [block]` | Qwen 3.8 27B Vision | luce_server | 8902 |
| `serve_laguna.sh` | Laguna XS 2.1 33B | luce_server | 8902 |
| `serve_ds4flash.sh [chunk]` | DeepSeek V4 Flash | luce_server | 8902 |
| `serve_ds4flash_f16kv.sh` | DeepSeek V4 Flash | luce_server | 8902 |
| `serve_ds41flash.sh` | DeepSeek V4.1 Flash | luce_server | 8902 |
| `serve_glm53flash.sh` | GLM-5.3-Flash (ROCmFP4 GGUF) | paoai-strix-engine (llama.cpp/Vulkan) | 8902 |
| `serve_kolibri.sh [n_cpu_moe]` | Kolibri-1 (78B MoE) | patched llama.cpp | 8902 |
| `/root/strata/run-iq3_s.sh` | Qwen 3.8 Flash Next | Strata | 8080 |

## Probes

| Script | Purpose |
|---|---|
| `gen_probe.py` | one request with unique random text; reports prefill and decode t/s from the engine's own timers |
| `prefill_probe.py` | cold-prefill probe for Strata (unique nonce text, no prefix cache) |
| `parity.py` | Class A parity harness: 10 HumanEval prompts, greedy, matches the lucebox blog protocol |
| `probe_suite.py` | harness-protocol probes against an already-running server |
| `dlrate.sh` | measures in-flight download rates (a stalled HF download cost hours once) |

## Sweeps and repeated experiments

| Script | What it tested |
|---|---|
| `rerun_all_lanes.sh` | re-runs every registry model through the lane, serialized with a lock |
| `run_remaining_lanes.sh` | same idea for a subset; keeps one lane at a time |
| `strata_prefill_sweep.sh` | Strata prompt-chunk cap (8192 vs 32768) across four prompt sizes |
| `ple_experiment.sh` | Strata n-gram table I/O mode: `direct` vs `mmap` vs `ram` |
| `qwen_kvflash_arms.sh`, `qwen_kvflash_ctl.sh` | KVFlash on/off and pool size on Qwen 3.8 27B |
| `ds4_prefill_sweep.sh`, `ds4_prefill_modes.sh` | DeepSeek V4 chunk size and `sparse` vs `exact` prefill |
| `kolibri_setup.sh`, `kolibri_patchfix.sh` | build the patched llama.cpp for `kolibri1` |
| `kolibri_ncpumoe_sweep.sh`, `kolibri_ncm_limit.sh` | `--n-cpu-moe` expert placement and its VRAM ceiling |

## Hard-won rules on this host (each one cost real time)

1. **One big model at a time.** 124 GB of unified memory is shared between the
   iGPU's model and anything on the CPU; two large models fail to allocate
   (observed: `cudaMalloc failed: out of memory`).
2. **Run one lane at a time.** Each lane stops any server it finds before
   starting its own, so two concurrent lanes kill each other's servers. Use the
   flock'd runner scripts.
3. **A lane must match server BINARY PATHS, never bare model names.** The lane's
   own command line contains strings like `kolibri` or `luce_server`, so a
   name-based `pkill`/pattern makes the lane kill its own session.
4. **Some engines need a warmup before measurement.** The first request after a
   load pays kernel/buffer warmup (Strata: 184 -> 218 t/s prefill at 500 tokens).
   Kyojin is extreme: its first start builds a dense-GEMM tune cache for ~13
   minutes, and measuring during that reads as a 5x prefill regression.
5. **Never reuse probe text.** Engines keep a prefix cache; identical prompts
   return cache hits (seen as 12,879 t/s "prefill"). Every probe must use unique
   text, and "warm" is a deliberate repeat of the same text.
6. **On a two-GPU host, pin the engine to the GPU it was built for.** Vulkan
   enumeration order differs from HIP's: GLM-5.3 on the Halo needs
   `GGML_VK_VISIBLE_DEVICES=1`, or it lands on the 32 GB card and pages over
   PCIe at 0.5 t/s. Kyojin needs `HIP_VISIBLE_DEVICES=1` plus `EXL3_HSA_LIB`
   pointing at the ROCm SDK's own `libhsa-runtime64.so.1` (the system ROCm 7.2
   HSA is too old and fails with `hsa_ext_image_create_v2` undefined).
7. **HF downloads can die silently mid-file.** Check the process, not just the
   log; `hf download` resumes, and a stale `.incomplete` can be removed to
   reclaim disk.
