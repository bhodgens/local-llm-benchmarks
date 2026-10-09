#!/usr/bin/env python3
"""Autosplit probe v4: same as v3 but with the gfx1201 rocBLAS kernels now
copied from /opt/rocm into the gfx1151 wheel's library dir (they are
version-compatible code objects; the wheel simply shipped without gfx1201).
Purpose: prove or refute that both devices can resolve BLAS in one process.
"""
import os, sys, time, json

HSA = '/root/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1'
os.environ['LD_PRELOAD'] = HSA
os.environ.setdefault('EXL3_HSA_LIB', HSA)
os.environ.setdefault('PYTHONNOUSERSITE', '1')
sys.path.insert(0, '/root/kyojin')

import torch
print('devices:', torch.cuda.device_count(), flush=True)

from exllamav3 import Config, Model, Cache

MODEL = '/root/models/glm-exl3-yamz'
config = Config.from_directory(MODEL)
model = Model.from_config(config)

t0 = time.time()
model.load(progressbar=False, use_per_device=None,
           reserve_per_device=[6.0, 6.0], autosplit_no_forward=True)
print(f'load: {time.time()-t0:.0f}s', flush=True)

from collections import Counter
c = Counter()
for mod in model.modules:
    c[str(getattr(mod, "device", None))] += 1
print('module device distribution:', dict(c), flush=True)

cache = Cache(model, max_num_tokens=2048, max_history=1)
print('cache ok', flush=True)

from exllamav3 import Tokenizer, Generator
tok = Tokenizer.from_config(config)
gen = Generator(model=model, cache=cache, tokenizer=tok, num_draft_tokens=0)
ids = tok.encode("The capital of France is")
out = gen.generate_simple(ids, 0, 8, 0)
print('forward OK:', tok.decode(out)[:60], flush=True)
print('PROBE_V4_DONE', flush=True)
