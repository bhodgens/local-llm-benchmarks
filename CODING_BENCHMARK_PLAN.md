# Coding Benchmark Suite - Implementation Plan

> **For Hermes:** Execute sequentially. One model at a time on the 3060. Save progress to JSON after each model. Generate HTML report at the end.

**Goal:** Benchmark 8 models on HumanEval, LiveCodeBench, and Aider Polyglot (top 3) to find the best coding model for the RTX 3060.

**Hardware:** RTX 3060 12GB (CUDA1), one model loaded at a time, V100 service (port 8081) stays running throughout.

**Estimated total runtime:** 16-24 hours (see per-model breakdown below)

---

## Models (8 total)

| # | Model | Type | Decode tok/s | Est. Time |
|---|-------|------|-------------|-----------|
| 1 | Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M | MoE | 26 | 3.2h |
| 2 | LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M | Small | 181 | 0.7h |
| 3 | LFM2.5-8B-A1B base Q4_K_M | Small | 181 | 0.7h |
| 4 | LFM2.5-8B-A1B Q6_K | Small | 170 | 0.7h |
| 5 | Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M | MoE | 26 | 3.2h |
| 6 | RavenX-OpenFable-Holo3 Q4_K_M | MoE | 26 | 3.2h |
| 7 | Qwen3.6-35B-A3B IQ3_K_R4 | MoE | 30 | 2.8h |
| 8 | qwen2.5-coder-14b-instruct Q4_K_M | Medium | 70 | 1.4h |

**Execution order:** LFM and Qwen-14b first (fast, 2.1h total), then MoE models (12.4h). This gives early results to validate the pipeline before the long runs.

---

## Benchmarks

### 1. HumanEval (ALL models)
- **Framework:** lm-evaluation-harness (EleutherAI) with `humaneval` task
- **Problems:** 164 (HumanEval) or use EvalPlus (HumanEval+ with more tests)
- **Settings:** pass@1, temperature=0.0, n_gen=1, max_tokens=1024
- **Gate:** If decode speed <24 tok/s during HumanEval, skip LiveCodeBench and Aider Polyglot for this model
- **Duration:** 5 min (LFM) to 26 min (MoE) per model

### 2. LiveCodeBench (ALL models passing gate)
- **Framework:** LiveCodeBench (https://github.com/LiveCodeBench/LiveCodeBench)
- **Problems:** 50-100 from latest release
- **Settings:** pass@1, temperature=0.0, max_tokens=4096, mixed languages (Python/Java/C++)
- **Release:** Latest available
- **Duration:** 35 min (LFM) to 2.7h (MoE) per model

### 3. Aider Polyglot (TOP 3 models only)
- **Framework:** Aider benchmark (https://github.com/Aider-AI/aider/blob/main/benchmark.md)
- **Exercises:** ~133 across 8 languages
- **Settings:** Edit format=whole, max_tokens=4096
- **Duration:** 20-60 min per model depending on speed
- **Selection:** Top 3 by combined HumanEval + LiveCodeBench score

---

## Model Configurations

Each model gets specific tuning based on our benchmark findings:

### MoE models (Qwopus3.6-35B variants + RavenX + IQ3_K_R4)
```
--device CUDA0
--gpu-layers 99
--cpu-moe
--flash-attn on
--ctx-size 262144        (max context)
--ubatch-size 512         (best prompt eval speed)
--threads 6               (best decode speed on 2600X)
--threads-batch 6
--cache-type-k q8_0
--cache-type-v q8_0
--parallel 1
--host 127.0.0.1
--port 18099
```

### LFM2.5 models (Q4_K_M and Q6_K)
```
--device CUDA0
--gpu-layers 99           (full GPU offload)
--flash-attn on
--ctx-size 65536          (model max)
--ubatch-size 512
--threads 4
--threads-batch 4
--cache-type-k q8_0
--cache-type-v q8_0
--parallel 1
--host 127.0.0.1
--port 18099
```

### qwen2.5-coder-14b-instruct
```
--device CUDA0
--gpu-layers 99
--flash-attn on
--ctx-size 32768          (model max)
--ubatch-size 512
--threads 4
--threads-batch 4
--cache-type-k q8_0
--cache-type-v q8_0
--parallel 1
--host 127.0.0.1
--port 18099
```

---

## Infrastructure Setup (Tasks 1-4)

### Task 1: Install lm-evaluation-harness
```bash
cd /home/caimlas/git
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
pip install -e ".[humaneval]"
```
Verify: `python -m lm_eval --tasks list | grep humaneval`

### Task 2: Install LiveCodeBench
```bash
cd /home/caimlas/git
git clone https://github.com/LiveCodeBench/LiveCodeBench.git
cd LiveCodeBench
pip install -e .
```
Verify: `python -m livecodebench --help`

### Task 3: Install Aider
```bash
pip install aider-chat
```
Verify: `aider --version`

### Task 4: Create scratch directory and progress tracking
```
mkdir -p /tmp/coding-bench/{results,logs,html}
```
Progress file: `/tmp/coding-bench/progress.json`

---

## Execution Tasks (Tasks 5-28)

### Per-Model Workflow (repeated for each model)

For each model, the workflow is:

**Step A: Load model on 3060**
1. Stop caimlas-llama service (frees 3060)
2. Start model with optimized settings on port 18099
3. Wait for health check
4. Record VRAM usage

**Step B: Throughput probe**
1. Send a short coding prompt
2. Measure decode tok/s
3. If <24 tok/s, record and skip remaining benchmarks for this model

**Step C: HumanEval**
1. Run lm-eval with humaneval task against localhost:18099
2. Save raw results + pass@1 score + tok/s
3. Update progress.json

**Step D: LiveCodeBench**
1. Run LiveCodeBench evaluation against localhost:18099
2. Save raw results + pass@1 score + tok/s
3. Update progress.json

**Step E: Teardown**
1. Kill model server
2. Move to next model

### Execution Order (fast models first to validate pipeline):

**Phase 1 - Quick validation (3 models, ~2.1h):**
- Task 5: LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M
- Task 6: LFM2.5-8B-A1B base Q4_K_M
- Task 7: LFM2.5-8B-A1B Q6_K

**Phase 2 - Medium model (~1.4h):**
- Task 8: qwen2.5-coder-14b-instruct Q4_K_M

**Phase 3 - MoE models (4 models, ~12.4h):**
- Task 9: Qwopus3.6-35B-A3B-Coder-MTP Q4_K_M (current production model)
- Task 10: Qwen3.6-35B-A3B IQ3_K_R4 (lower quant, might be faster)
- Task 11: Qwen3.6-35B-A3B-Abliterated-Heretic Q4_K_M
- Task 12: RavenX-OpenFable-Holo3 Q4_K_M

**Phase 4 - Aider Polyglot on top 3 (2-4h):**
- Task 13: Select top 3 models by combined score
- Task 14-16: Run Aider Polyglot on each

**Phase 5 - Report:**
- Task 17: Generate HTML results page

### Task 17+: Restore production service
1. Restart caimlas-llama with Qwopus3.6-35B-A3B-Coder-MTP production config
2. Verify both services healthy

---

## Progress Tracking Format

`/tmp/coding-bench/progress.json`:
```json
{
  "start_time": "2026-07-06T20:00:00Z",
  "models": [
    {
      "name": "LFM2.5-8B-A1B-Clean-RealWorld-v2 Q4_K_M",
      "status": "completed",
      "decode_tps": 181.2,
      "vram_mib": 5200,
      "humaneval": {
        "pass_at_1": 0.35,
        "completed": 164,
        "total_tokens": 32800,
        "avg_decode_tps": 179.5,
        "wall_time_s": 320
      },
      "livecodebench": {
        "pass_at_1": 0.12,
        "completed": 75,
        "total_tokens": 225000,
        "avg_decode_tps": 178.1,
        "wall_time_s": 2100
      },
      "aider_polyglot": null
    }
  ]
}
```

---

## HTML Report

Single-page report at `/tmp/coding-bench/results/report.html` with:
- Summary table: model, decode tok/s, HumanEval pass@1, LiveCodeBench pass@1, Aider Polyglot %, wall time
- Sorted by combined score
- Per-model detail cards with VRAM usage and throughput graphs
- Bar charts comparing pass@1 across models
- Clean dark theme, suitable for terminal/CLI viewing

---

## Time Estimate Summary

| Phase | Models | Duration |
|-------|--------|----------|
| Infrastructure setup | - | 30 min |
| Phase 1: LFM models (3) | Fast validation | 2.1h |
| Phase 2: Qwen 14B (1) | Medium | 1.4h |
| Phase 3: MoE models (4) | Slow | 12.4h |
| Phase 4: Aider Polyglot (top 3) | Selected | 2-4h |
| Phase 5: Report + restore | - | 30 min |
| **TOTAL** | | **19-21 hours** |

With the >24 tok/s throughput gate, all 8 models qualify (slowest is MoE at 26 tok/s). If the gate were stricter (>30 tok/s), the 3 MoE Q4_K_M models would be skipped, saving ~9.5h.

**Recommendation:** Run Phase 1-2 first (3.5h) to validate the pipeline and get quick results. Then let Phase 3 run overnight (12.4h). Aider Polyglot and report generation the next morning.

---

## Notes

- All benchmarks run against the OpenAI-compatible API at localhost:18099. Each model uses the same endpoint, just swapped sequentially.

- The V100 service (port 8081) stays running throughout. Only the 3060 is repurposed for benchmarking.

- LiveCodeBench requires code execution for test validation. Python/Java/C++ runtimes must be installed. Check: `python3 --version`, `javac -version`, `gcc --version`.

- Temperature 0.0 means greedy decoding. This is deterministic -- reruns would produce identical results. Good for reproducibility.
