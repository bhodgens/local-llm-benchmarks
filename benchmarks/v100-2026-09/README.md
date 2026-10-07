# V100 coding-eval campaign (2026-09) — run evidence

Committed so the V100 protocol is reproducible without the CUDA host.
`final_results.json` (repo root) holds the summary scores; this directory holds
the evidence they were derived from.

## Files

- `lcb_question_ids.json` — the exact 75 LiveCodeBench question_ids used by
  every V100 LCB run. `scripts/run_livecodebench.py` used a LCB fork with
  `--num_problems 75`, which selected the FIRST 75 of release_latest in the
  fork's ordering: Codeforces 1873_A.. plus AtCoder, contest months
  2023-05..2023-10. Recovered from the run's own output JSONs on socrates.
- `progress.json` — the full campaign progress file (`/tmp/coding-bench/
  progress.json` on socrates): 91 model entries with per-model
  humaneval/livecodebench/tau2 sub-results, failures, timings, template and
  gpu fields. 45 MB of raw generations and `*_eval_all.json` files remain
  host-only under `/home/caimlas/git/LiveCodeBench/output/`.

## Protocol

- LiveCodeBench: codegeneration scenario, release_latest, n=1, temperature 0,
  max_tokens 4096, 75 problems (ids above), thinking disabled.
- tau2-bench: airline, 15 tasks, seed 42, user simulator = V100 Qwopus
  (port 8081) — note this differs from borg's self-play simulator.
- HumanEval: 164 problems.

## Consumers

- `scripts/generate_report.py` reads progress.json format (paths hardcoded to
  the CUDA host at line 10-13 — borg's equivalent is
  `scripts/borg/run_borg_coding_eval.py`, which writes the same shape to
  `/root/bench/coding-eval-progress.json`).
- borg's 75-problem re-run (`run_borg_coding_eval.py` +
  `--question_ids_file /tmp/v100_lcb_ids.json`, filter patch in borg's LCB
  clone) uses `lcb_question_ids.json` for apples-to-apples scoring.
