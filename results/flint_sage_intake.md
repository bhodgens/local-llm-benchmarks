# Flint + Sage-A1.5 intake (2026-10-22, V100, upstream llama.cpp 970b51c)

## Engine
- Upstream ggml-org/llama.cpp master `970b51c` (2026-10-22), built sm_70-only:
  `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DCMAKE_BUILD_TYPE=Release`
  binaries in `/tmp/upstream-master/build/bin/`
- Both archs verified in-tree: `flint` (LLM_ARCH_FLINT) and `qwen35moe`

## FLINT (Recursive, 39.8B MoE / ~2.5B active, Q4_K_XL 23.33GiB)
| metric | value |
|---|---|
| decode (probe) | 85.9 t/s |
| decode (bench tg128) | 89.1 t/s |
| prefill pp512 | 758.9 t/s |
| prefill pp4096 | 746.8 t/s |
| prompt_tps | 129.1 |
| VRAM | 23.9 GiB (fits V100, ~8 GiB headroom) |
- Sanity: coherent coding output. No MTP head in file (none in header).
- Sliding-window 513 keeps KV tiny; no KV quant needed at 32K ctx.

## SAGE-A1.5 (Shanghai AI Lab, 35B MoE / ~3B active, Q4_K_M 20.21GiB)
| metric | value |
|---|---|
| decode (probe) | 109.0 t/s |
| decode (bench tg128) | 115.3 t/s |
| prefill pp512 | 873.7 t/s |
| prefill pp4096 | 846.0 t/s |
| +MTP decode | 146.1 t/s (acc 149/211 = 70.6%) |
| VRAM | ~20.2 GiB (comfortable fit) |
- Sanity: coherent coding output. MTP head present (nextn x1) — draft-mtp n_max 2 gives +34% decode.
- Multimodal (mmproj verified); text lane only here.

## Notes
- Both fit the V100 32GB at Q4 with headroom; neither fits the 3060 12GB.
- Sage is the faster of the two (115 vs 89 t/s decode) and has working MTP (+34%).
- Flint is a coding-agent specialist (SWE-bench-V 74.2 claimed) — worth an LCB/tau2 lane if you want quality numbers, not just speed.
- Gotcha: llama-bench has no `-mmp` flag (mmproj not supported in bench) — use the server for multimodal; text bench runs fine without it.
