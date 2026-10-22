# Bonsai-2 perf: ranked test plan (from deleg_6093f46f repo deep-read, 2026-10-22)

## Context
- ~/git/llama.cpp-prismml = July b9591 = Bonsai-1 era (no Bonsai-2 refs). Sept Bonsai-2 bench used vendor PREBUILT b10685.
- Vendor alive: HEAD 73bfe6c (b10787) + Oct branches (cuda/yukon-gdn-verify-20261017, feat/tq1_0-cuda-v7, dspark-layout-autodetect-v7).
- ADA-quart/bonsai2-27b-v100-optimizations = tuning kit for prism branch (base 842b18804 = prism PR #248, NOT vanilla).
  All their numbers: WDDM V100-16GB Windows. No Linux/32GB numbers published.
- Baselines ours (bench-0918, thinking-off): Bonsai2 PQ2_0 44.35 t/s; PTQ1_0 43.4 t/s; PTQ1_0+MTP-n1 54.4 t/s (V100).
- Local assets: /var/tmp/llms/bench-0918/bonsai2/{PQ2_0, PTQ1_0, PTQ1_0-mtp, qwen38-27b-nextn-head}.gguf

## Ranked experiments (all STAGED; user gate before GPU runs)
0. Build prism HEAD 73bfe6c sm_70+86 (bg proc; /tmp/prism-head) — unlocks everything
1. V100 PTQ1_0 vs PQ2_0 A/B (llama-bench -p 16384 -n 128 -fa 1 -ctk/v q8_0): expect tg +6.4%, -1.17GiB; PPL parity check (2-chunk wikitext)
2. V100 MTP: -m PTQ1_0-mtp.gguf --spec-type draft-mtp --spec-draft-n-max 2 --spec-draft-type-k/v q8_0: expect +22-38% decode; check draft acceptance (our Sept dspark sidecar had 1% acceptance — MTP-grafted head should be 52-88%)
3. V100 ubatch sweep: -ub 1024 vs 2048 @ 32K/128K (32GB may allow ub2048@128K where their 16GB could not): expect pp +5-7%, tg +12-25% short
4. ft5 DFlash-2 drafter (Q8_0 + Q4_K_M downloading to /home/files/llms/bonsai2-drafter/): try -md draft loader on prism HEAD (native dflash; dspark converter NOT applicable); Metal-only numbers so far (acc +8-10%, decode +5-15%)
5. ADA patch port onto prism HEAD (HIGH effort; 7300 lines, conflicts mmvq/gdn/speculative/qwen35): q8_0-KV FA direct read + D256 prefill (DIVERGES greedy - PPL-validate) + MTP microbatch fix. Only if 1-4 leave prefill-on-long-ctx still weak.
6. 3060 lane: prism HEAD + PTQ1_0 + MTP (third party claims 26.3->40.5 t/s decode via PTQ1_0 small-batch matvec, vendor PR #218 - check if in HEAD). bonsai-turbo = Bonsai-1 only, dead end.

## Gotchas
- ADA patch chain: 842b18804 -> 9b98d9dfd (patch1) -> 2b6c36f69 (patch2). Apply with --3way, expect fuzz.
- D256 kernel needs CUTLASS@62750a2b include/cute+cutlass in ggml/src/ggml-cuda/sm70-vendor/ (kit ships flash/ headers).
- Kill switches: LLAMA_SM70_D256=0 runtime.
- PTQ1_0-quantized MTP head = rejected by ADA (acceptance 70->10%). Use f16/q8_0 head.
- Their model-as-agent warning: tool-call infinite loops (90+ identical calls); ship jinja tool-loop guard if used for agents.
