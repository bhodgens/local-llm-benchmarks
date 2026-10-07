# Benchmark Protocols — reproducible runs on any host

This file is the contract for every benchmark in this repo. A number in
`report.html`, `final_results.json`, `BORG_RESULTS.md`, or
`benchmarks/*/progress.json` is only comparable if the protocol below was
followed. When adding a new host, copy the relevant protocol block and fill in
a host section; do not invent a new protocol.

## Hosts

| Host | Hardware | GPU id convention | Reports |
|---|---|---|---|
| V100 / CUDA host (socrates) | V100 32 GB (user sim: 2nd GPU) | CUDA0 | `report.html`, `final_results.json` |
| 3060 | RTX 3060 | CUDA0 | `report.html` (gpu=3060 rows) |
| borg | R9700 32 GB (hip:0) + Ryzen AI MAX+ 395 Strix Halo 124 GB (hip:1), ROCm 7.2.1 | hip:0 = R9700, hip:1 = Strix | `BORG_RESULTS.md`, `BORG_REPORT.html`, `report.html` BORG section |

`hip:0`/`CUDA0` mappings are load-bearing: every launcher pins devices
explicitly. borg's R9700 is `hip:0` even though the Strix Halo is the "main"
APU.

## Engine rule

Benchmarks run through the engine each model family is served by in production;
the engine is recorded per row. On borg: luce_server for Qwen 3.8 27B /
Laguna / DeepSeek, Strata for Qwen 3.8 Flash Next (+ Swift/Coder fine-tunes),
Kyojin EXL3 for GLM-5.3-Flash, patched llama.cpp for Kolibri-1. Cross-engine
loading is NOT possible in general — see BORG_RESULTS.md "engine portability"
(kolibri1/glm5next archs rejected by luce_server; GGML type-42 fork collision
blocks Strata GGUFs on luce_server).

## 1. Throughput (prefill / decode)

Entry point: borg `scripts/borg/bench_borg_lanes.py` (registry-driven), CUDA
host `scripts/run_v100_benchmarks.py`-style probes.

- One model at a time (host lock); discard one warmup request before measuring
  (first-request prefill reads ~2x low on some engines).
- Prefill probes: unique random text (no prefix cache reuse) at ~500 and ~2500
  tokens; report cold AND warm.
- Decode: 512-token generations, 2 prompts × 2 runs, at temp 0.3 (sampled) and
  temp 0 (greedy — where speculative decoding engages).
- Record: prefill-1st-req, prefill-cold, prefill-2.5K, decode-sampled,
  decode-greedy, spec-decode ran?, accept rate, boot seconds.
- Numbers come from the engine's own timers, never client wall-clock.

## 2. HumanEval (164 problems, execution-scored)

Script: borg `scripts/borg/run_humaneval_local.py`.

- Do NOT use lm-evaluation-harness API mode for chat servers: prompts arrive
  double-escaped and every model scores a fake 0.0 (see script header).
- All 164 problems, temperature 0, max_tokens 1024.
- Score by executing the official tests (pass@1), code extracted from fences;
  strip fences, keep imports, `from typing import *` shim if missing.
- Sensitive to servers returning `content: null` (Strata): fall back to
  `reasoning_content`.

## 3. LiveCodeBench (codegeneration)

Scripts: CUDA `scripts/run_livecodebench.py`; borg
`scripts/borg/run_borg_coding_eval.py`.

- Scenario `codegeneration`, `release_latest`, n=1, temperature 0,
  max_tokens 4096, thinking disabled, `--use_cache`, `--evaluate`.
- **Question set is part of the protocol.** V100: first 75 of release_latest
  (fork's ordering) = `benchmarks/v100-2026-09/lcb_question_ids.json`
  (2023-05..2023-10, Codeforces 1873_A...). borg reuses that exact set via
  `--question_ids_file` (filter patch required — stock LCB has no such flag).
  borg's earlier 12-problem 2025-04 window is NOT comparable and is archived
  in `benchmarks/borg-2026-10/12prob-backup/`.
- For reasoning models: force `reasoning_effort=none` (or equivalent) — else
  reasoning consumes the whole token budget and content comes back null
  (patched in borg's LCB clone `oai_runner.py`; patch scripts in
  `scripts/borg/patch_oai_runner*.py`).
- The server must receive its own API model id, not the harness's store key.

## 4. tau2-bench (agentic tool use)

Scripts: CUDA `scripts/run_tau2bench.py`; borg `run_borg_coding_eval.py`.

- Domain `airline`, 15 tasks, 1 trial, seed 42, temperature 0, max_steps 30
  (reduce for context-limited models; record the value).
- User simulator differs by host and is recorded per row: V100 uses a dedicated
  Qwopus simulator server; borg uses self-play (same lane server plays user).
  tau2 scores are comparable within a simulator class, not across.
- Reward = mean over task simulations; parse from
  `data/simulations/<save_to>/results.json` → `simulations[].reward_info.reward`
  (that path, not top-level `results`).

## 5. Memory footprint

Per loaded model: VRAM per GPU via `rocm-smi --showmeminfo vram` (or
nvidia-smi), host RAM via `free`, sampled while the model is loaded and idle,
before any inference. Recorded per row in BORG_RESULTS.md's memory table and
`report.html` detail modals.

## Adding a new host

1. Add a `serve_<model>.sh` launcher per model, pinning devices explicitly.
2. Copy the registry pattern from `scripts/borg/bench_borg_lanes.py`.
3. Run the protocols above unchanged; only the host section in this file and
   the device pins change.
4. Put raw run evidence under `benchmarks/<host>-<date>/` (question ids,
   progress.json, output dir listings) and a README — summaries alone are not
   enough to re-derive or audit a result later.
5. Add the host to the table above and to `generate_report.py`'s inputs.

## Known portability limits (do not rediscover these)

- luce_server rejects `kolibri1` and `glm5next` architectures.
- Strata-format sharded GGUFs cannot load on luce_server: GGML type id 42 is a
  fork collision (Strata `Q2_0` vs lucebox `TQ3_0` — same id, different block
  layouts). Fixing requires a format mapping or re-quantization.
- lucebox Laguna backend (patched) supports `n_head_arr[64]` — Laguna-S-2.1
  loads with the split layout; stock XS-only builds cap at 40 layers.
- lm-eval API-mode HumanEval against chat servers is invalid on this repo's
  servers (double-escaped prompts → fake 0.0).
