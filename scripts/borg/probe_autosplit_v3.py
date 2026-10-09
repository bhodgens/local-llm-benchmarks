#!/usr/bin/env python3
"""Autosplit probe v3: replicate the working serve.py env exactly (LD_PRELOAD of
the rocm-sdk-devel HSA, as tools/strix_halo/env.sh does for single-device), then
attempt the cross-device autosplit load of the GLM EXL3 model.

Key insight being tested: serve.py works single-device today, so its env CAN
drive both torch and the HSA runtime. The question is purely whether exllamav3's
autosplit load can place tensors on both HIP devices in one process.
"""
import os, sys, json, time

HSA = '/root/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1'
os.environ['LD_PRELOAD'] = HSA
os.environ.setdefault('EXL3_HSA_LIB', HSA)
os.environ.setdefault('PYTHONNOUSERSITE', '1')
sys.path.insert(0, '/root/kyojin')

import torch
print('devices:', torch.cuda.device_count(), flush=True)
for i in range(torch.cuda.device_count()):
    print(' ', i, torch.cuda.get_device_name(i), flush=True)

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
bytes_by_dev = Counter()
for mod in model.modules:
    dev = getattr(mod, 'device', None)
    dev_s = str(dev)
    c[dev_s] += 1
print('module device distribution:', dict(c), flush=True)

# Try a tiny forward on cuda:0 input to check it can route across devices
cache = Cache(model, max_num_tokens=2048, max_history=1)
print('cache ok', flush=True)
try:
    from exllamav3 import Tokenizer, Generator
    tok = Tokenizer.from_config(config)
    gen = Generator(model=model, cache=cache, tokenizer=tok, num_draft_tokens=0)
    ids = tok.encode("The capital of France is")
    out = gen.generate_simple(ids, tok.eos_token_id if hasattr(tok, 'eos_token_id') else 0, 8, 0)
    print('forward OK:', tok.decode(out)[:60], flush=True)
except Exception as e:
    print('forward FAILED:', str(e)[:200], flush=True)
print('PROBE_V3_DONE', flush=True)
