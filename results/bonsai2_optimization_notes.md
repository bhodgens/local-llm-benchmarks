# Bonsai-2 optimization campaign — reproduction notes (2026-10-22)

## Engine
- PrismML fork HEAD `73bfe6c` (prism-b10787), built sm_70+sm_86:
  `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="70;86" -DCMAKE_BUILD_TYPE=Release`
  binaries in `/tmp/prism-head/build/bin/`
- ADA-patched variant: `/tmp/ada-port` (patch1+patch2 --3way onto 73bfe6c; +5-13% prefill, PPL parity OK)

## Best config (both cards)
```
llama-server -m Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  --spec-draft-type-k q4_0 --spec-draft-type-v q4_0 \
  -fa 1 -ctk q4_0 -ctv q4_0 -ub 2048 -ngl 99
```
- PTQ1_0 beats PQ2_0: +13.6% decode (V100), +26% (3060), -1.17 GiB weights
- MTP graft (built-in head) is the biggest decode lever: +59% V100, +58% 3060
- Larger context is FREE on the 3060 (hybrid GDN attention = tiny KV): decode flat ~71 t/s 32K->64K
- q4_0 KV saves ~1.1 GiB at 64K ctx on the 3060 (9956 vs 11044 MiB), no decode cost
- ubatch 2048 helps prefill only at >4K ctx; ub512 fine below

## Measured (prism HEAD 73bfe6c, fa on)
| config | V100 | 3060 |
|---|---|---|
| PTQ1_0 tg128 | 55.9 t/s | 45.2 t/s |
| PTQ1_0 pp4096@ub2048 | 972.6 t/s | 589.9 t/s |
| +MTP decode | 87.0 t/s (acc 78.5%) | 61.7 t/s (acc 58%) |
| +MTP 64K ctx | — | 71.1 t/s (acc 77%, q4_0 KV) |
| Sept baseline (prebuilt) | 44.35 t/s | 26.7 t/s |

## PPL parity
PQ2_0 == PTQ1_0 bit-identical: 8.7508 (2-chunk), 9.2476 (full wikitext-2-raw).
ADA port vs stock: delta 0.0001 (PASS, gate <= 0.005).

## ft5 DFlash-2 drafter (leg 4)
- Loads natively via `--spec-type draft-dflash` (loader reads GGUF lineage; no rename needed)
- Acceptance 59.7%, decode 73.5 t/s (+34.5%), prefill penalty -45%
- MTP graft still wins (87.0 t/s, 78.5%) — use MTP when the head is in the file

## Gotchas
- llama-server from python MUST get `CUDA_VISIBLE_DEVICES=0` (spills to 3060 OOM otherwise)
- `llama-bench -o json` broken in this tree — use `-o md`
- llama-bench has NO `-c` context flag in this build (use `--fit-ctx`/server for ctx sweeps)
- Report prefill column matches by GGUF filename — 3060 rows inherit V100 prefill value (cosmetic; decode tok/s is per-row correct)
