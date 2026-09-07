# V100 ECC-off tok/s re-tests (2026-09-06)

Hardware change under test: ECC disabled on V100 (GPU0). 3060 (GPU1) unchanged = control lane.
Protocol notes: llama-bench tg128 = pp512/tg128 -r 3 -fa on -q8_0 KV (matches original speed_norm).
Chat probe = 256-token decode at 8K ctx, best of 2. Raw data: bench_results.json `[ecc-off*]` keys.

## Clean comparisons (same config, same measurement)

| Model | Lane | Metric | Prior | ECC-off | Delta |
|---|---|---|---|---|---|
| BTL-4 Q4_K_M | V100 | llama-bench tg128 | 88.85 | 93.69 | **+5.4%** |
| BTL-4 Q4_K_M | V100 | chat probe | 88.85 | 88.81 | 0.0% |
| Qwopus3.6-27B-v2-MTP | V100 | chat probe | 30.4 | 32.03 | **+5.4%** |
| Carnice-V3 Q4_K_M | V100 | chat probe | 31.6 | 31.24 | -1.1% |
| gemma-4-12B-QAT | 3060 (control) | chat probe | 38.2 | 39.81 | +4.2% |

**Verdict: ECC-off effect on V100 decode is +0% to +5.4% - small, within the
noise band established by the control lane (+4.2% with no hardware change).**
No dramatic tok/s impact from disabling ECC.

## Not comparable (config changed between measurements) - do NOT read as ECC deltas

| Model | Why |
|---|---|
| Nail [Sharp] 85.3 / Ornith [Sharp] 92.1 (vs prior 24.3/25.1) | Prior: 262K ctx + cpu-moe. Retest: 8K ctx full offload. Config artifact, ~3.6x. |
| LFM2.5-Clean 205.9 (vs prior 105.7) | CONTROL lane moved 2x -> probe config drift (threads/parallel), proves cross-config deltas swamp ECC effects. |
| Qwythos-27B-MTP mtp3 40.6 (vs prior 42.9) | Ctx differed (8K vs 16K). Weak comparability, -5%. |
| pp512 columns (e.g. BTL-4 150.9 -> 753.9) | Original pp512 measured under different batch/prompt conditions. Not comparable. |

## Blocked / open items

- **Nail + Ornith original-config retests**: current llama.cpp build fails
  `create_context` at 262K + cpu-moe for both models (worked in the original
  runs). Needs a build bisect or the original binary to reproduce.
- **Ternary-Bonsai dspark retest**: Q2_0_g128 requires PrismML fork; upstream
  cannot load it. PrismML server with draft n=4 hit VRAM fit conflicts
  (-ngl 99 + draft) and then client-timeout at fit-friendly settings. Aborted
  per low-value rule; can revisit with -fit off and reduced ctx.
- **DeepSeek-Coder-V2-Lite**: GGUF deleted, skipped (was first-measurement want).
- **Qwythos-27B-v1**: only MTP GGUF remains; plain-mode bench recorded on it
  (31.6 tg128) - usable as the v1 proxy (same weights + MTP head).
