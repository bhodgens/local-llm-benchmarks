#!/usr/bin/env python3
"""Update BORG_RESULTS.md's multi-device findings with the exllamav3 probe
results, sync the borg-2026-10 evidence dir, then regenerate both reports."""
import json

BORG = '/private/tmp/local-llm-benchmarks/BORG_RESULTS.md'
s = open(BORG).read()

anchor = '## Pending lanes'
block = '''### exllamav3 (Kyojin) cross-device probe (2026-10-09)

The hybrid GLM question was also tested on Kyojin's engine (exllamav3), which
has real multi-device machinery (--gpu_split / autosplit load / tensor-parallel).
Result: **blocked by per-arch ROCm wheel packaging**, verified with a 4-attempt
escalation (all in benchmarks/borg-2026-10/glm-exl3-autosplit-blocker.json):

1. Kyojin venv as-is: gfx1151-only wheel ships no gfx1201 TensileLibrary -
   rocBLAS cannot serve the R9700.
2. env.sh source path: 7.14 devel libhsa has undefined symbols vs the 7.13
   runtime the torch wheel needs.
3. Copying 353 gfx1201 kernel/table files from system ROCm into the wheel:
   Tensile resolves, but hipblasSgemm then fails HIPBLAS_STATUS_INTERNAL_ERROR
   (7.2.1 code objects vs 7.13 wheel API mismatch).
4. Redirecting the runtime at system /opt/rocm/lib: torch 7.13 device kernels
   are invalid against the 7.2.1 runtime (hipErrorInvalidImage).

Root cause: the torch wheel is rocm7.13-based; no install on the box has both
gfx1151 and gfx1201 with a torch-compatible runtime. Cross-device exllamav3 GLM
needs a unified ROCm >=7.13 install or multi-day source builds of torch +
exllamav3 against system ROCm 7.2.1 - and would still face the peer-copy
corruption below in a layer split.

Also verified: borg's system ROCm 7.2.1 DOES ship both archs' libraries (96
gfx1151 + 56 gfx1201 rocBLAS entries, 3027 hipBLASLt files), and a proper
gfx1151 hipBLASLt tuning table now exists (92 shapes tuned on-device,
benchmarks/borg-2026-10/../tools note: /root/strata/tools/hip/
gfx1151-hipblaslt-100202.txt) - up to 21.5x over untuned hipBLAS on the
largest shape. The pieces are all there except a unified ROCm/torch install.

''' + anchor
if 'exllamav3 (Kyojin) cross-device probe' not in s:
    s = s.replace(anchor, block, 1)
    open(BORG, 'w').write(s)
    print('BORG_RESULTS.md updated')
else:
    print('already present')
