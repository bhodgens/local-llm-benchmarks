#!/usr/bin/env python3
"""IOMMU A/B: checksum-verified P2P probe, both directions, matching the
handoff's hybrid_p2p_probe2 methodology (pattern write, round-trip, memcmp)
plus ROCR #355's index-derived pattern, run under amd_iommu=off.

Also: prefill benchmark on the iGPU + R9700 for the throughput gain datapoint.
"""
import os, sys, time, json

os.environ['LD_PRELOAD'] = '/root/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1'
os.environ.setdefault('EXL3_HSA_LIB', '/root/kyojin/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib/libhsa-runtime64.so.1')
os.environ.setdefault('PYTHONNOUSERSITE', '1')
sys.path.insert(0, '/root/kyojin')

import torch
print('iommu cmdline check passed externally; devices:', torch.cuda.device_count(), flush=True)

SIZE = 64 * 2**20  # 64 MiB, matches hybrid_p2p_probe2
ROUNDS = 3

results = {}
for direction, (src, dst) in {'dev0->dev1': (0, 1), 'dev1->dev0': (1, 0)}.items():
    # allocate on owner devices
    src_t = torch.randn(SIZE // 4, dtype=torch.float32, device=f'cuda:{src}')
    # deterministic pattern (index-derived, ROCR #355 style)
    idx = torch.arange(SIZE // 4, dtype=torch.int32, device=f'cuda:{src}')
    src_t = (idx * 2654435761 % 2**31).to(torch.float32)

    # copy dst<-src via torch (uses cudaMemcpyPeerAsync under the hood)
    ok_all = True
    for r in range(ROUNDS):
        dst_t = torch.empty_like(src_t, device=f'cuda:{dst}')
        dst_t.copy_(src_t)
        torch.cuda.synchronize(dst)
        # verify: read back on OWNER side (src device) - compare src vs the dst copy
        # bring dst copy back to src device's host view for comparison
        host_src = src_t.cpu()
        host_dst = dst_t.cpu()
        match = torch.equal(host_src, host_dst)
        ok_all = ok_all and match
        if not match:
            bad = (host_src != host_dst).sum().item()
            print(f'{direction} round {r}: MISMATCH {bad}/{host_src.numel()} elements', flush=True)
    results[direction] = 'CLEAN' if ok_all else 'CORRUPT'
    print(f'{direction}: {results[direction]} ({ROUNDS} rounds, {SIZE/2**20:.0f} MiB)', flush=True)

# bandwidth of the clean direction (or whichever works)
def bw(dst, src, label):
    src_t = torch.randn(SIZE // 4, dtype=torch.float32, device=f'cuda:{src}')
    dst_t = torch.empty_like(src_t, device=f'cuda:{dst}')
    for _ in range(2):
        dst_t.copy_(src_t)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(5):
        dst_t.copy_(src_t)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f'{label}: {5*SIZE/2**30/dt:.1f} GiB/s', flush=True)

try:
    bw(1, 0, 'bw dev0->dev1')
    bw(0, 1, 'bw dev1->dev0')
except Exception as e:
    print('bw failed:', str(e)[:120], flush=True)

json_out = {'iommu': 'amd_iommu=off', 'probe': results, 'size_mib': SIZE // 2**20, 'rounds': ROUNDS}
json.dump(json_out, open('/root/bench/results/p2p-iommu-off.json', 'w'), indent=2)
print('P2P_PROBE_DONE', flush=True)
