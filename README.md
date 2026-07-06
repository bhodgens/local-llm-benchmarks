# LLM Benchmarks

Benchmarking suite for local llama.cpp models on dual-GPU system
(Tesla V100 32GB + RTX 3060 12GB), llama.cpp v9836 (upstream-dflash branch).

All tests: 8K context, flash attention on, 2 decode runs averaged.

## Benchmark Results (2026-07-06)

### V100 (CUDA0) - 27B Dense Models

| Model                              | Config   | Decode (tok/s) | Prompt (tok/s) | MTP Accept |
|------------------------------------|----------|---------------|----------------|------------|
| Qwopus3.6-27B-Coder Q5_K_S         | no-spec  |          30.6 |          555   | -          |
| Qwopus3.6-27B-Coder Q5_K_S         | MTP n=3  |          36.9 |            -   | 70%        |
| Qwopus3.6-27B-Coder Q5_K_S         | MTP n=5  |          34.4 |            -   | 45%        |
| Qwopus3.6-27B-Coder-Compat Q4_K_M  | no-spec  |          31.2 |          527   | -          |
| Qwopus3.6-27B-Coder-Compat Q4_K_M  | MTP n=3  |          40.3 |            -   | 73%        |
| Qwopus3.6-27B-Coder-Compat Q4_K_M  | MTP n=5  |          40.3 |            -   | 53%        |
| Qwopus3.6-27B-v2-MTP Q4_K_M        | no-spec  |          31.2 |          480   | -          |
| Qwopus3.6-27B-v2-MTP Q4_K_M        | MTP n=3  |          36.9 |            -   | 63%        |
| Qwopus3.6-27B-v2-MTP Q4_K_M        | MTP n=5  |          31.3 |            -   | 44%        |
| Qwopus3.5-27B-v3 Q4_K_M            | no-spec  |          31.2 |          490   | -          |

### RTX 3060 (CUDA1) - 35B MoE Models (--cpu-moe, ngl=99)

| Model                              | Config   | Decode (tok/s) | Prompt (tok/s) | MTP Accept |
|------------------------------------|----------|---------------|----------------|------------|
| Qwen3.6-35B-A3B-Abliterated Q4_K_M | no-spec  |          25.4 |          110   | -          |
| unsloth-qwen3.6-35B-A3B-UD Q4_K_M  | no-spec  |          24.1 |          110   | -          |
| Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M | no-spec  |          25.5 |           88   | -          |
| Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M | MTP n=3  |          16.9 |            -   | 69%        |
| Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M | MTP n=5  |          12.0 |            -   | 63%        |
| RavenX-OpenFable-Holo3 Q4_K_M      | no-spec  |          25.7 |          114   | -          |

## Key Findings

### V100 (27B Dense): MTP is a clear win

- Best config: Qwopus3.6-27B-Coder-Compat-MTP at MTP n=3 = 40.3 tok/s (29% faster than no-spec)
- MTP n=3 is the sweet spot across all V100 models (36.9-40.3 tok/s, 63-73% acceptance)
- MTP n=5 is inconsistent: helps Coder-Compat (40.3 tok/s) but hurts v2-MTP (31.3, no improvement over no-spec)
- All Q4_K_M models decode at ~31.2 tok/s baseline; Q5_K_S at 30.6 (slightly larger, slightly slower)
- The original Q5_K_S Qwopus Coder and the new Q4_K_M Coder-Compat are the top MTP performers

### 3060 (35B MoE): MTP hurts, cpu-moe is the equalizer

- MTP makes the MoE models SLOWER: 25.5 -> 16.9 tok/s at n=3. The MTP draft computation
  competes with the main model for GPU resources, and the overhead exceeds the speculative benefit
- RavenX lacks nextn_predict_layers metadata entirely (MTP not possible)
- All four MoE quants perform similarly at no-spec: 24.1-25.7 tok/s
- cpu-moe is essential: it puts the 256 expert FFN weights on CPU (only 8 active per token)
  while keeping attention/shared experts on GPU

### Recommendations

- V100 service: Use Qwopus3.6-27B-Coder-Compat-MTP Q4_K_M with MTP n=3 (40.3 tok/s, best speed)
- 3060 service: Use any MoE variant no-spec (~25 tok/s). MTP is counterproductive on this GPU.
- DFlash remains non-functional in mainline llama.cpp (zero drafts produced in all prior tests)

## Directory Structure

```
llm-benchmarks/
  README.md                     This file
  bench_results.json            Raw benchmark data
  scripts/
    bench_orchestrator.py       Master benchmark runner (all models, incremental)
    bench_model.py              Single-model manual benchmark tool
    bench_watcher.sh            Polling watcher (runs orchestrator every 5 min)
    download_models.sh          Sequential HF download script
```

## Usage

### Run all benchmarks (incremental, skips completed)

```bash
python3 scripts/bench_orchestrator.py
```

Results save to `bench_results.json` after each model. Re-running skips
already-benchmarked configs.

### Benchmark a single model

```bash
python3 scripts/bench_model.py \
    --model /home/files/llms/MODEL.gguf \
    --port 18099 \
    --device CUDA0 \
    [--mtp --n-max 3]
```

## Configuration

The orchestrator defines model lists at the top of `scripts/bench_orchestrator.py`:

- `V100_MODELS` - Dense models tested on the V100 (CUDA0)
- `M3060_MODELS` - MoE models tested on the 3060 (CUDA1) with `--cpu-moe`

Each entry is `(filename, device, is_moe, has_mtp)`.

All tests run at 8K context with flash attention to isolate model+arch performance.
Production context sizes (128K-200K) are configured separately in systemd services.
