# borg coding-eval raw evidence — 2026-10

Run evidence for the numbers in `BORG_RESULTS.md` and the report.html BORG
section. borg-local copies live at `/root/bench/coding-eval-progress.json`
(+ `-12prob.json.bak` for the superseded 12-problem LCB window).

## Files

- `progress.json` — orchestrator output in the repo's progress.json shape:
  per-model livecodebench / tau2 sub-results, status, eval errors.
- `12prob-summary.json` — the superseded 12-problem LCB window scores
  (2025-04 date window). Not comparable to the V100 75-problem set; kept for
  history. tau2 values here are final.

## Protocol deltas vs V100 (all disclosed)

- LiveCodeBench: same 75 questions (see ../v100-2026-09/), same sampling
  params; hardware and serving engines differ.
- tau2: same tasks/seed; user simulator is SELF-PLAY (the lane server plays
  the user) instead of the V100's dedicated Qwopus simulator. Comparable
  within borg, not head-to-head with V100 tau2.
- laguna-s21 tau2: all 15 sims terminated with infrastructure errors
  (litellm ContextWindowExceeded at max_steps=15) — model-speed property,
  recorded as N/A.
- kolibri-1 / vision / coder tau2: partial scoring (7/5/5 of 15 tasks
  completed within limits); n recorded in progress.json.
