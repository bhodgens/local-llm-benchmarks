#!/usr/bin/env python3
"""Probe v5: use the SYSTEM /opt/rocm 7.2.1 BLAS libraries (which have both
gfx1151 AND gfx1201 kernels, 1778 Tensile files + 3027 hipBLASLt entries)
instead of the per-arch wheel libraries. Point LD_LIBRARY_PATH at /opt/rocm/lib
so torch's rocBLAS/hipBLASLt resolve there, keeping the wheel's python side.

This is the cleanest unification: one ROCm 7.2.1 BLAS install serves both archs.
Risk: torch 2.12+rocm7.13 wheel may require newer symbols than 7.2.1 ships -
if so this fails at load with a symbol error and we document it.
"""
import os, sys, time

HSA = '/opt/rocm/lib/libhsa-runtime64.so.1'
os.environ['LD_PRELOAD'] = HSA
os.environ['LD_LIBRARY_PATH'] = '/opt/rocm/lib:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ.setdefault('PYTHONNOUSERSITE', '1')
sys.path.insert(0, '/root/kyojin')

import torch
print('torch:', torch.__version__, '| hip:', torch.version.hip, flush=True)
print('devices:', torch.cuda.device_count(), flush=True)

# minimal BLAS sanity on BOTH devices first
for dev in ('cuda:0', 'cuda:1'):
    a = torch.randn(512, 512, device=dev)
    b = a @ a
    torch.cuda.synchronize(dev)
    print(f'  {dev}: GEMM ok, checksum {float(b.sum()):.1f}', flush=True)

from exllamav3 import Config, Model, Cache

config = Config.from_directory('/root/models/glm-exl3-yamz')
model = Model.from_config(config)
t0 = time.time()
model.load(progressbar=False, use_per_device=None,
           reserve_per_device=[6.0, 6.0], autosplit_no_forward=True)
print(f'load: {time.time()-t0:.0f}s', flush=True)

from collections import Counter
c = Counter(str(getattr(m, "device", None)) for m in model.modules)
print('module device distribution:', dict(c), flush=True)

cache = Cache(model, max_num_tokens=2048, max_history=1)
from exllamav3 import Tokenizer, Generator
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok, num_draft_tokens=0)
ids = tok.encode("The capital of France is")
out = gen.generate_simple(ids, 0, 8, 0)
print('forward OK:', tok.decode(out)[:60], flush=True)
print('PROBE_V5_DONE', flush=True)
