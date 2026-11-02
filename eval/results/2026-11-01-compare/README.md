# Comparison questions over two papers: the app's rule against two chunks per paper, 2026-11-01

## Setup

- Script: `eval/qasper/compare_eval.py` (commit `3c084b3`), stages `build`, `run`, `judge`,
  `summarize`, run on maddog 2026-11-01 23:18 to 2026-11-02 02:19 (`run.log`).
- Pairs: QASPER development questions asked of several papers were grouped into four topics
  (baselines, datasets, evaluation metrics, English only). Within each topic, 30 pairs of papers
  were drawn (seed 0), giving 120 comparison questions such as "Compare the two papers: what
  baselines does each one compare against?" (`pairs.json`).
- Both papers are loaded into one store, as the app does with two files, and the context is built
  two ways:
  - *app*: the top 4 chunks overall, then the best chunk of any file not yet among them, up to 8;
  - *balanced*: the top 2 chunks of each paper (4 in all).
- Model, prompt and judge as in `2026-11-01-qasper-dev`. Each answer is judged twice, once per
  paper, against that paper's references; a pair is correct only if both judgements are.
- `compare_findable.py` (run 2026-11-02) checks the evidence matching: it counts, for each
  paper, the chunks that hold the paper's annotated evidence, with the same splitter and matching
  rule (`findable.json`).

## Results

| Rule | Evidence from both papers in context | Correct for both papers | Papers answered correctly |
|---|---|---|---|
| app | 5 / 120 (4.2%) | 21.2% [14.8, 29.4] (25/118) | 50.4% (119/236) |
| balanced | 3 / 120 (2.5%) | 28.7% [21.2, 37.5] (33/115) | 53.9% (124/230) |

- **Balanced against app, correct for both:** 21 pairs gained, 12 lost of 114 judged under both
  (exact McNemar p = 0.16).
- **Why evidence from both papers is rare:** not the matching. Each paper's evidence is in its
  own chunks for 105 of 119 papers, and for both papers in 83 of 120 pairs. The comparison
  question, worded for both papers, seldom retrieves the passages that answer each paper's own
  question: 47 (app) and 43 (balanced) of 240 paper slots had their evidence in the context.

## Files

| File | Contents |
|---|---|
| `pairs.json` | The 120 pairs: topic, comparison question, each paper's own question |
| `answers.jsonl` | Per pair and rule: the context, whether each paper's evidence is in it, chunks per paper, the answer |
| `judgments.jsonl` | Per pair, rule and paper: verdict and the judge's reasoning |
| `summary.json` | Rates per rule and the paired comparisons |
| `findable.json` | Papers and pairs with findable evidence; paper slots with evidence in context per rule |
| `run.log` | Console output |
