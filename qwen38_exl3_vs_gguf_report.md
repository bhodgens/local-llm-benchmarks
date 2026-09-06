# Qwen3.8-27B on V100: EXL3 request -> GGUF fallback benchmark

Date: 2026-09-04
Request: benchmark Mia-AiLab 3.5bpw EXL3, turboderp EXL3, darkbit1001 4.5bpw EXL3 on the V100 (and unload whatever was on it).

## 1. GPU state

- ComfyUI (PID 872176, port 8188, 8 GB) killed -> V100 at 0 MiB before any work.

## 2. Downloads completed (all EXIT:0)

| Repo | Local dir | Size |
|---|---|---|
| Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw | /home/files/llms/Qwen3.8-27B-EXL3-3.5bpw-Mia | 15.3 GB |
| turboderp/Qwen3.8-27B-exl3 @ 4.00bpw branch | /home/files/llms/Qwen3.8-27B-exl3-4.00bpw-turboderp | 16.9 GB |
| darkbit1001/Qwen-3.8-27B-exl3-4.500bpw-hb6 | /home/files/llms/Qwen-3.8-27B-exl3-4.500bpw-hb6 | 18.4 GB |
| unsloth/Qwen3.8-27B-GGUF UD-IQ3_S (3.4375 bpw) | /home/files/llms/Qwen3.8-27B-UD-IQ3_S.gguf | 12.0 GB |
| unsloth/Qwen3.8-27B-GGUF UD-Q4_K_S (4.49 bpw) | /home/files/llms/Qwen3.8-27B-UD-Q4_K_S.gguf | 15.4 GB |

Notes: turboderp main rev is measurement-only (recipes/KLD/qbench); weights live in
bpw branches - chose 4.00bpw to bracket the requested 3.5/4.5 range. All are the
same base model (Qwen/Qwen3.8-27B, Qwen3_5ForConditionalGeneration, hybrid
linear-attn + full-attn, 248k vocab).

## 3. EXL3 on V100: hard blocker (verified, not a config issue)

Evidence chain, four independent levels:

1. **Wheel arch coverage**: official exllamav3 wheels are built with
   `cudaarch = "8.0 8.6 8.9 9.0 10.0 12.0+PTX"` (.github/workflows/build.yml).
   No sm_70 SASS, and PTX from sm_80 cannot JIT *down* to sm_70.
2. **Kernel source**: EXL3 GEMM/GEMV unconditionally use `cp.async` and
   `mma.sync.aligned.m16n8k16` (ptx.cuh, exl3_gemm_inner.cuh, exl3_gemv_kernel.cuh).
   No `__CUDA_ARCH__` guards around them; the CC_OLD dispatch constant exists but
   no sm_70 kernel path does.
3. **PTX compile probe**: the exact PTX statements compiled with
   `nvcc -arch=sm_70` (toolkit 12.0) -> ptxas:
   `Feature 'cp.async' requires .target sm_80 or higher`
   `Feature '.m16n8k16' requires .target sm_80 or higher`
   (sm_80 control compiles clean). Source build is impossible without a kernel
   rewrite.
4. **Runtime probe**: bench-venv (exllamav3 1.0.0+cu128.torch2.11) on V100:
   even `torch.ones(8, device="cuda")+1` -> `cudaErrorNoKernelImageForDevice`.

Alternative engines: none. vLLM/SGLang have no EXL3 support (open feature request
vllm#19896); GPTQModel's EXL3 runtime is sm_75+; exllamav2/tabbyAPI do not read
EXL3. EXL3 = QTIP trellis decode built on the Ampere ISA.

Conclusion: the requested EXL3 benchmark on V100 is impossible. This is an
instruction-set wall, fixable only upstream (a pre-Ampere kernel path).

## 4. Fallback: bpw-matched GGUF (same base model, engine that supports sm_70)

| Requested EXL3 | GGUF stand-in | bpw match |
|---|---|---|
| Mia 3.5bpw | UD-IQ3_S | 3.4375 actual |
| darkbit 4.5bpw | UD-Q4_K_S | 4.49 actual |

(RESULTS TABLES BELOW - filled by run)

### Speed (llama-bench, V100, fa=on, KV q8_0, 8K depth, 2 reps)

| Quant | bpw | pp512 | tg128 | weights |
|---|---|---|---|---|
| UD-IQ3_S | 3.44 | 578.8 t/s | 29.05 t/s | 11.2 GiB |
| UD-Q4_K_S | 4.49 | 649.8 t/s | 33.11 t/s | 14.3 GiB |

### Quality

| Quant | BenchKit sanity:25 | LCB release_latest pass@1 (75) |
|---|---|---|
| UD-IQ3_S | 84.0% (21/25), 13 min | 0.707 |
| UD-Q4_K_S | 84.0% (21/25), ~10 min | 0.693 |

## 6. Verdict

- **Speed**: UD-Q4_K_S wins on both axes (+12% pp, +14% tg) despite more weights -
  K-quants have a more regular layout than i-quants on Volta's tensor cores.
- **Quality**: sanity identical (84%); LCB 0.707 vs 0.693 is inside the n=1 noise
  band for this model family (same-base prior runs spanned 0.693-0.760). No
  measurable quality difference at these bitrates on this suite.
- **Practical**: at ~4.5bpw-equivalent you gain nothing on LCB but ~14% speed over
  the 3.4bpw point; IQ3_S saves 3.1 GiB if VRAM is the constraint (plenty of
  headroom either way at 32GB).
- **Notable**: both UD quants scored below the previously-benched plain
  Q4_K_S (0.747) on the same suite - unsloth dynamic quants were not better here.
- **EXL3 quality ceiling foregone** (turboderp KLD ladder): 3bpw 0.000399 ->
  4bpw 0.000124 -> 5bpw 0.000038 weighted KLD vs BF16 - EXL3's ~3.2x-per-bpw
  fidelity curve remains inaccessible until an Ampere-only GPU is available
  (the 3060 12GB cannot hold these 15-18GB quants).

Prior V100 reference points (same base, progress.json): plain Q4_K_S LCB 0.747 @
33.6 t/s; plain Q4_K_M LCB 0.760 (MTP n3) / 0.747; Heretic Q4_K_M 0.733; AEON
0.693. tau2 refs: Q4_K_M 0.50, Q4_K_S 0.286.

### Publisher KLD ladder (turboderp calibration vs BF16 base, weighted)

| bits | weighted KLD |
|---|---|
| 3 | 0.000399 |
| 4 | 0.000124 |
| 5 | 0.000038 |

Each +1 bpw ~= 3.2x fidelity improvement; 4.5bpw sits ~2x closer to BF16 than 4.0.

## 5. Artifacts

- **Sortable HTML report (regenerated 2026-09-04): ~/llm-benchmarks/report.html** - includes the two UD-GGUF rows, the three EXL3 models as documented failures (engine badge: EXL3), an "EXL3 on V100: Instruction-Set Wall" findings card, and the failure table with the full root-cause chain. Generator: scripts/generate_report.py (progress.json normalized to its schema).
- Orchestrator: ~/llm-benchmarks/scripts/run_qwen38_v100_gguf.py (resumable, caches per stage)
- Logs: /tmp/coding-bench/logs_qwen38gguf/
- Results: /tmp/coding-bench/progress.json (qwen38 entries)
- New LCB registrations: local/qwen38-27b-ud-iq3s, local/qwen38-27b-ud-q4ks
- Skill saved: mlops/engine-gpu-compat-gate (pre-download arch gate; would have caught this in 5 min)
