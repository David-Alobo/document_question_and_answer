# Repeat runs: four set-ups answered with seeds 1 and 2, and the judge rerun, 2026-11-01

## Setup

- Script: `eval/qasper/run_eval.py` with `--seed` (answer model) and `--judge-seed` (judge), at
  commit `3c084b3`, run on maddog in two lanes (Ollama on GPU 5, port 11500, and GPU 3, port
  11510), 2026-11-01 21:52 to 2026-11-02 08:24 (`run.log`).
- Seed 0 of each set-up is the original run; this folder holds seeds 1 and 2:

  | Folder | Set-up | Seed 0 (original run) |
  |---|---|---|
  | `fixed_seed1`, `fixed_seed2` | the app: 1,000-character chunks, top 4 | `2026-11-01-qasper-dev` (fixed) |
  | `oracle_seed1`, `oracle_seed2` | annotators' evidence paragraphs | `2026-11-01-qasper-dev` (oracle) |
  | `c500-k8_seed1`, `c500-k8_seed2` | 500-character chunks, top 8 | `2026-11-01-qasper-dev-c500-k8` |
  | `c500-k8-rerank_seed1`, `c500-k8-rerank_seed2` | the same, reranked | `2026-11-01-qasper-dev-c500-k8-rerank` |
  | `fixed_judge_seed1` | seed-0 fixed answers, judged again with judge seed 1 | `2026-11-01-qasper-dev` (fixed) |

- Retrieval is deterministic, so each run reused its set-up's seed-0 retrieval. The copies the
  runs read were byte-identical to the seed-0 files (MD5 `383a3b44...` fixed and oracle,
  `ee9e28bc...` c500-k8, `a5930909...` c500-k8-rerank) and are not repeated here.
- Model, prompt, judge and settings as in `2026-11-01-qasper-dev`; only the seeds differ.
- Every run's failed judgements (the judge's reasoning filling its window) were judged again with
  `--retry-errors` (marked in `run.log`); the last record per question counts.
- Analysis: `python eval/qasper/repeats.py` writes `summary.json`. It compares runs on the
  questions judged in all three runs, with exact McNemar tests on correct versus not.

## Results

| Set-up | Questions | Seed 0 | Seed 1 | Seed 2 | Majority | Correct every run | Correct in no run | Same call in all runs |
|---|---|---|---|---|---|---|---|---|
| fixed (app) | 1,000 | 53.5 | 53.2 | 54.5 | 54.1 | 40.7 | 33.6 | 74.3 |
| oracle | 892 | 69.1 | 70.9 | 68.9 | 70.4 | 57.5 | 19.1 | 76.6 |
| c500-k8 | 1,001 | 57.1 | 55.7 | 56.2 | 55.7 | 45.7 | 32.3 | 77.9 |
| c500-k8-rerank | 994 | 56.6 | 59.3 | 55.7 | 57.6 | 43.3 | 29.3 | 72.5 |

Judged correct, %.

- **Single answers vary, rates do not.** A quarter of questions changed between correct and not
  across three runs. Only 1.6% of the fixed app's answers were word for word the same in all three.
- **Noise floor.** The largest difference between two runs of one set-up was 3.5 points
  (c500-k8-rerank, seeds 1 and 2: 113 against 78, p = 0.014).
- **Against the app's set-up, per seed** (points, McNemar p):
  - oracle: +16.5, +19.0, +15.5 (all p < 10⁻¹⁴);
  - c500-k8: +3.5, +2.5, +1.8 (p = 0.035, 0.14, 0.30);
  - c500-k8-rerank: +3.0, +5.9, +1.4 (p = 0.095, 0.00059, 0.43).
- **Judge rerun:** the same verdict for 894 of 1,002 answers (89.2%), the same correct-or-not call
  for 93.4%. The correct rate moved by 0.6 points (36 against 30, p = 0.54).

## Files

| File | Contents |
|---|---|
| `<setup>_seed<n>/answers_fixed.jsonl` (`answers_oracle.jsonl` for oracle) | Per question: answer, seed, tokens, time |
| `<setup>_seed<n>/judgments_*.jsonl` | Per question: verdict, the judge's reasoning, judge seed; retries appended |
| `<setup>_seed<n>/summary.json`, `summary.md` | That run's rates, as `run_eval.py summarize` writes them |
| `<setup>_seed<n>/console.log` | That run's console output |
| `run.log` | Start and end of every stage in both lanes |
| `summary.json` | `repeats.py` output: per set-up and seed, between runs, between set-ups, judge rerun |
