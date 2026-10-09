BORG GLM-5.3-FLASH MULTI-DEVICE HANDOFF
========================================
Date: 2026-10-08. Written after live investigation on borg.
Full research set: ~/git/glm-5.3-optimization/docs/research/ (start: 50-synthesis.md).
Probe sources: ~/git/glm-5.3-optimization/probes/hybrid_p2p_probe.cc, hybrid_p2p_probe2.cc

HARDWARE (verified on borg via SSH)
-----------------------------------
- borg: AMD Ryzen AI Max+ 395, 32 threads, 124 GiB LPDDR5X, kernel 6.17.0-1032-oem,
  ROCm 7.2.1 (/opt/rocm -> /opt/rocm-7.2.1). Kernel cmdline: no iommu parameter (default).
- dev0 = Radeon AI PRO R9700 (gfx1201 RDNA4): 32 CU, 31.9 GiB GDDR6, 640 GB/s.
- dev1 = Radeon 8060S iGPU (gfx1151 RDNA3.5): 20 CU visible to HIP, 118 GiB (GTT-backed
  unified), ~256 GB/s shared.
- Target model: zai-org/GLM-5.3-Flash (glm5_next: 320B-A18B MoE, 34 KDA linear layers +
  11 DSA layers, mHC, native MTP layer).
- Benchmarks to beat (owner-verified): Kyojin/EXL3 single-iGPU 26-32 t/s decode;
  hybrid Q2 iGPU+dGPU 34-36 t/s.

INVESTIGATION RESULT: IGPU<->R9700 PEER-COPY CORRUPTION (the blocker for multi-device)
--------------------------------------------------------------------------------------
Probe: probes/hybrid_p2p_probe2.cc (64 MiB pattern, hipMemcpyPeerAsync, round-trip memcmp).
Run on borg 2026-10-08, ROCm 7.2.1.

1. The HIP API LIES about peers on this box:
   hipDeviceCanAccessPeer = 1 BOTH directions; hipDeviceEnablePeerAccess = hipSuccess BOTH.
   (On the Windows rig that first found this bug, the pair reported non-peers; on Linux the
   API claims full peer capability. Do not trust the API here.)
2. Despite that, hipMemcpyPeerAsync dev0(R9700) -> dev1(iGPU) SILENTLY TRANSFERS GARBAGE
   (pattern corrupt, with and without enablePeerAccess). Direction dev1(iGPU) -> dev0(R9700)
   is CLEAN. One-directional silent corruption = worst failure class: timings look normal,
   output is soup.
3. Explicit staging through host memory (D2H then H2D) is CORRECT both directions.
   Measured (pageable host buffers, sync): 0.25 GiB round trip in 44.3 ms ~= 6.1 GB/s
   effective. Pinned (hipHostMalloc) staging was NOT yet measured and will be much faster.
   Layer-boundary tensors at batch 1 are KB-MB scale, so even the pessimistic rate costs
   microseconds per layer - staging is NOT a blocker for layer-split.
4. Repro one-liner on borg:
     cd /tmp && hipcc -O2 -o p2 hybrid_p2p_probe2.cc && ./p2 --full
   Expected output: 0->1 raw CORRUPT / 1->0 raw OK / 0->1 after enablePeer CORRUPT /
   1->0 after enablePeer OK / staged roundtrip integrity OK.

RULES THAT FOLLOW (non-negotiable for any multi-device build on borg)
---------------------------------------------------------------------
R1. Build llama.cpp with -DGGML_CUDA_NO_PEER_COPY=ON. It is COMPILE-TIME (CMake), not an
    env var (setting the env var does nothing - verified in-tree ggml-cuda.cu @71ad0590).
R2. Keep GGML_CUDA_P2P unset unless a measured win exists (upstream docs: may corrupt when
    IOMMU enabled).
R3. Every multi-device benchmark must pair a CORRECTNESS GATE with the speed number.
    Gate: temp-0 short factual prompt, exact-match expected text (e.g. "capital of France"
    -> "Paris"). A speed-only benchmark happily records numbers for garbage output
    (measured on the Windows rig: plausible timings + token soup).
R4. If writing custom HIP code that moves tensors across devices, stage via host memory
    explicitly; never hipMemcpyPeerAsync in the R9700->iGPU direction.
R5. Untested variable that could change everything: boot with amd_iommu=off (also worth
    +1-7% prefill per strix-halo-llamacpp measurements) and rerun the probe. If corruption
    disappears, re-evaluate R1-R4 - but keep the correctness gate forever.

NEXT STEPS: GET GLM-5.3-FLASH RUNNING ON BOTH DEVICES (llama.cpp path)
----------------------------------------------------------------------
Step 0 - Free the box. Currently ~1.1 GiB free: root runs
  /root/strata/build-hip-gfx1151/strata --serve --pack (~69 GiB RSS, 96% CPU)
  and /root/lmeval/bin/python3 serve/server.py. Stop or trim before model loads
  (owner approval needed - these may be live services).

Step 1 - Build dual-arch llama.cpp (>= b11476, which merged GLM MTP #29928; >= b11512 ideal):
    git clone https://github.com/ggml-org/llama.cpp && cd llama.cpp
    cmake -B build-hip \
      -DGGML_HIP=ON \
      -DAMDGPU_TARGETS="gfx1151;gfx1201" \
      -DGGML_CUDA_NO_PEER_COPY=ON \
      -DCMAKE_HIP_ARCHITECTURES="gfx1151;gfx1201"
    cmake --build build-hip --config Release -j --target llama-cli llama-server llama-bench
  Sanity: llama-bench must list both devices; single-device runs must pass R3 gate.

Step 2 - Model file: unsloth GGUF UD-Q2_K_XL (108.7 GB; fits 124 GiB with ~8-10 GB KV
  headroom at 64K) or UD-IQ3_XXS (120.4 GB, needs 2-4 CPU-offloaded expert layers via
  --n-cpu-moe). Quant must carry NextN (MTP) tensors - unsloth quants do. Verify arch
  string is "glm5-next" (hyphenated; older "glm5next" GGUFs fail on master).

Step 3 - A/B test matrix (each row: correctness gate FIRST, then llama-bench):
    a) iGPU only:        -ngl 99 -sm row -dev none (baseline, expect ~13-17 t/s no MTP)
    b) R9700 only:       -dev ROCM0 (32 GB won't fit Q2_K_XL; sanity only)
    c) layer split:      -sm layer -ts 96,24   (tune ratio; try 64,64 / 96,32 / 80,48)
    d) expert-tier:      keep attention+KV on R9700 via -ot exps=dev1 style override +
                         --moe-cache-mib <N> (expert LRU on the dGPU; merged 2026-10-07,
                         zero published numbers - this is the novel measurement)
    e) add MTP to best of c/d: --spec-type draft-mtp --spec-draft-n-max 2 -np 1
       (n-max 2..4 only; MoE verify cost grows with expert-union size, n>=8 = net loss)
  Watch: multi-seq on glm5-next needs --kv-unified; greedy spec output is not
  byte-reproducible on this arch (batch-decomposition numerics) - gate on semantic match.

Step 4 - Measure and record: t/s at ctx 0 / 4K / 32K / 64K (decode decays with depth),
  prefill pp512/pp4096, TTFT, VRAM/GTT per device, temp drift over 30 min.
  Success line: hybrid beat 36 t/s with correct output, or a documented reason it can't.

Step 5 - Only if Step 4 shows overlap gains: expert-TIER placement patch (R9700 = attention/
  KDA/DSA+indexer/KV/MTP head/hot experts; iGPU = 288-expert tail). Prior art says this is
  the big one (qwen4exp sibling-arch split 3.7-4.4x on this GPU pair; Lucebox R9700 hot
  experts +2x; cache sweet spot ~2x active experts). Details:
  docs/research/engines/20-hybrid-placement-axis.md sections 2-4.

FALLBACK / PARALLEL PATHS
-------------------------
- Best-known-good today (no multi-device work): Kyojin + turboderp EXL3 td205 pack
  (85.2 GB) = published 32.0-35.7 t/s, 38.9 @128K, MTP-2. Provenance/quality:
  docs/research/engines/19-engine-kyojin-exl3.md. Kyojin itself is single-device for GLM;
  its multi-device code paths are untested AND unaudited for the R4 hazard.
- Strata GLM port (PR #1315, HIP) - if merged, its Qwen3.8 anchor on this chip is 59.7 t/s;
  verify its cross-device behavior against R4 before trusting hybrid modes.
- antirez/ds4 - native gfx1151 ROCm backend, official GLM support, MIT; single-device
  benchmark target, not yet tested on borg.

ENVIRONMENT NOTES
-----------------
- borg GTT gives the iGPU a 118 GiB window; BIOS UMA carveout setting unverified (check in
  Phase 1; likely minimal carveout since HIP reports 118 GiB visible).
- amd-smi / rocm-smi work. Device order: dev0=R9700 (PCIe), dev1=8060S (iGPU).
- Probe + gate harness live in ~/git/glm-5.3-optimization/probes/.
- Research docs with full sourcing: ~/git/glm-5.3-optimization/docs/research/
  00-research-plan.md (plan), 50-synthesis.md (ranked actions), engines/10-20 (per-engine),
  40-arxiv-emerging-techniques.md (research-grade levers).
