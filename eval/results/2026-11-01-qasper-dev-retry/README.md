# Retry on a miss: the model checks its context and may search again, 2026-11-01

## Setup

- Script: `eval/qasper/retry_eval.py --base eval/results/2026-11-01-qasper-dev` (commit
  `3c084b3`), then `run_eval.py judge` and `summarize`, run on maddog 2026-11-01 23:15 to
  2026-11-02 03:36 (`run.log`).
- For each question, the app's four retrieved chunks are shown to the answer model with the
  question, and it says whether they hold the answer. If it says no, it writes a search query, the
  paper is searched again with it, and the top four chunks of the new search replace the first
  ones. The answer is then generated from the final chunks exactly as the app does, so the prompt
  length is the app's.
- Model, prompt, judge and settings as in `2026-11-01-qasper-dev`.

## Results

- **Retries:** the model asked for a new search for 177 of 1,005 questions (17.6%); 3 questions
  have no PDF.
- **The check:** of the 493 questions with text evidence whose top four chunks held none, the
  model said the chunks held the answer for 396 (80.3%). Of the 398 whose chunks did hold
  evidence, it asked for a new search for 29.
- **Retrieval:** evidence in the final top four for 45.0% of the 891 questions with text
  evidence, against 44.7% without the retry (13 gained, 10 lost; p = 0.68).
- **Answers:** 54.8% [51.7, 57.8] judged correct (549/1,002), against 53.6% for the app
  (91 gained, 79 lost; exact McNemar p = 0.40). Of the 177 retried questions, 94 were correct
  without the retry and 102 with it.

## Files

| File | Contents |
|---|---|
| `retry.jsonl` | Per question: the model's check and its search query (null if it did not retry) |
| `retrieval.jsonl.gz` | Per question: the final four chunks and whether each holds evidence |
| `answers_fixed.jsonl`, `judgments_fixed.jsonl` | Answers and verdicts, as in a `run_eval.py` run |
| `papers.jsonl` | The PDFs used (copied from the base run) |
| `summary.json`, `summary.md` | `run_eval.py summarize` output |
| `run.log` | Console output |
