# Two-model 262K experiment: shared expert arena vs hugetlb pool (borg, 2026-10-13)

## Verdict
**PROVEN.** flashnext (Strata, Qwen3.8-Flash-Next IQ3_S) serves at **262144 ctx / k8v4 KV / parallel 4**
concurrently with BTL-4 (superfluid/llama.cpp, 262144 ctx, 4 lanes) using
`--shared-expert-arena /dev/shm/strata-arena` instead of a 48 GiB hugetlb pool. Both services
were left **active** (final state: fn :8080 @262K, BTL-4 :8453 @262K). 524288 also works
(32.4 tok/s decode) but costs ~35% vs 262K (48-50 tok/s), so 262K is the final config.

## Evidence (exact log lines)

Shared arena bound, no MAP_HUGETLB / THP path (strata-iq3_s.log):
```
strata generate: expert arena: cudaHostRegister PORTABLE ok; MAP_SHARED file /dev/shm/strata-arena (pack hash checked)
```
(The hugetlb-era boots instead showed MAP_HUGETLB / largepage lines; here "large pages skipped ... 4 KB pages" for the host-registered copy only - the THP-compaction wedge of Strata issue #771 does not occur.)

Load-time warning that turned out benign:
```
WARNING: the expert arena needs 46.84 GiB of RAM but 34.04 GiB is available (12.81 GiB short)
```
The "short" RAM is the MAP_SHARED arena itself, accounted as tmpfs/shmem (free shows shared=46G),
not anonymous RAM. No swap: Swap used stayed 0 through all runs (one run touched 22 MB).

Memory at final state (`free -g`): total 124, used 111, free 1, shared 46, buff/cache 87, available 13, swap 0.

## BTL-4 q4_0 trim: NOT POSSIBLE
- `--cache-type-k/q4_0` -> `superfluid: serve has no option --cache-type-k`
- `--kv-bits 4` -> `superfluid: --kv-bits 4: the llamacpp runtime keeps its KV type per context and exposes no such knob` (help marks it basert-only; basert is not installed - libbaseRT missing)
- Trim applied: `--max-batch 8 -> 4` in /etc/systemd/system/btl4.service (daemon-reload + restart).
- BTL-4 footprint at 4 lanes/262K: ~31 GB total process footprint (weights 9.28 GiB + f16 KV for 262K x 4 lanes + ROCm/runtime overhead), measured as used-memory delta with flashnext down.

## Arena file mechanics (learned by failure)
- Engine requires the file to be EXACTLY 50294992896 B (46.84 GiB arena + 896 B header).
  Wrong size -> exit with the expected size printed: `shared arena /dev/shm/strata-arena is 51539607552 B, expected 50294992896 B including its header`.
- A zero-filled file of the exact size -> `has an incompatible or missing header` (pack-hash check).
- **Let the engine create the file**: on first boot with the path absent, it writes the header itself,
  then reuses the pack-bound file on later boots regardless of --max-context (same file served 262K and 524288).
- Boot with shared arena: **262K up in 1.4 min** (vs 8.7 min for the old 128K int8 config) - the arena
  load reads from the page-cached tmpfs file at 4.25 GiB/s.

## Runs
| run | fn ctx | fn KV | BTL-4 | up | boot | Paris | decode tok/s (client, 256-tok x3) |
|---|---|---|---|---|---|---|---|
| baseline | 131072 | int8/resident 32768 | down | yes | - | OK | 61.5 |
| C2 | 131072 | int8 | up, 4 lanes | yes | ~1 min | OK | - |
| C3 | 262144 | k8v4 | up, 4 lanes | yes | 1.4 min | OK | 45.0 / 50.2 / 48.3 |
| C3b | 524288 | k8v4 | up, 4 lanes | yes | 0.4 min | OK | 32.4 x3 |
| C4 final | 262144 | k8v4 | up, 4 lanes | yes | 0.7 min | OK | 42.0 / 49.9 / 50.1 |

## System changes made
1. `/etc/systemd/system/btl4.service`: `--max-batch 8` -> `--max-batch 4`. daemon-reload done. (q4_0 KV flags rejected by superfluid - see above.)
2. `/root/strata/strata-iq3_s.json` args: `--max-context 262144`, `--kv k8v4`, `--kv-resident` removed (incompatible with k8v4), added `--shared-expert-arena /dev/shm/strata-arena`; top-level `parallel` stays 4. Backup of the 128K config: `/root/strata/strata-iq3_s.json.bak-128k`.
3. `/dev/shm/strata-arena`: 50294992896 B MAP_SHARED arena file (engine-created). Persists until reboot (tmpfs); engine reuses or recreates it.
4. Created `/root/bench/results/two-model-262k-experiment.json` and this summary.
5. No git pushes. vm.nr_hugepages left at 0; hugepages cmdline arg untouched.

## Caveats
- The engine's "12.81 GiB short" warning fires every boot (benign, see above) but means a second
  large anonymous allocation at boot could now trigger pressure; swap is the fail-safe (32G, untouched).
- 524288 works but decode drops to ~32 tok/s and prompt rate ~48 tok/s; 262K kept as final.
- If BTL-4 KV grows beyond 4 lanes x 262K f16, revisit: q4 KV needs either basert runtime installed
  or a newer superfluid llamacpp adapter exposing cache-type flags.
