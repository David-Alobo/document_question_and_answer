# QASPER dev: 500-character chunks, top 8 (candidate A), 2026-11-01

The first of the two leading set-ups from the retrieval sweep
(`../2026-11-01-retrieval-sweep-dev`), run on answers. It sends about the same amount of text
as the app (8 × 500 characters against 4 × 1,000), but in smaller pieces.

## Setup

As `../2026-11-01-qasper-dev` (fixed condition only), except for the retrieval settings:

- Code: branch `feat/multi-doc`, commit `80120bb` (adds `--chunk-size`, `--chunk-overlap`, `--k`
  and `--rerank` to `run_eval.py`), copied to maddog. `config.json` records the settings.
- Retrieval: the app's `Utils.split_pdf` with 500-character chunks and 100 overlap (20%, as in
  the app), `LocalVectorStore` with all-MiniLM-L6-v2, the top 8 chunks, no reranker.
- Same PDFs (`papers.jsonl` has the same SHA-256 hashes), same answering model and settings,
  same judge and prompt.
- Ollama server on GPU 3 (port 11510). Candidate B ran on the other server at the same time,
  so `seconds` per answer depends on load and is not comparable between runs; prompt tokens
  are.
- The run was started in one queue with candidate B, stopped, and resumed on its own
  ("resumed after splitting the queue" in `run.log`). Records written before the stop were
  kept; no question was answered or judged twice, apart from the retries below.
- **Retry:** 2 judgements came back empty (the judge's reasoning filled its window). Both were
  retried with `--retry-errors` (20,480-token window, 12,000 tokens); one then succeeded
  (`5fa36dc8…`, correct) and one failed again (`e431661f…`). That question counts as not
  judged, which the summary treats as not correct. All attempts are kept in
  `judgments_fixed.jsonl`; the last record per question counts.

## Results

| | Questions | Judged correct | Correct or partial | Answer F1 | Mean prompt tokens |
|---|---|---|---|---|---|
| App (1,000 / top 4), from `../2026-11-01-qasper-dev` | 1,002 | 53.6% [50.5, 56.7] | 72.3% | 11.6 | 1,154 |
| This run (500 / top 8) | 1,002 | 57.1% [54.0, 60.1] | 75.2% | 11.6 | 1,230 |

Wilson 95% intervals.

- **Paired against the app** on the same 1,002 questions: 148 questions became correct and
  113 stopped being correct (exact two-sided McNemar p = 0.035; Holm-corrected over the two
  candidates, p = 0.07, so not different from the app at the 0.05 level). Candidate B, with
  reranking, is in `../2026-11-01-qasper-dev-c500-k8-rerank`.
- **Retrieval:** evidence was findable for 832 of 891 questions with text evidence (93.4%,
  against 91.5% with 1,000-character chunks), and the top 8 held it for 57.6% [54.2, 60.9] of
  those. `summary.md` has the curve for k = 1–10.
- **Where answers are lost:** correct for 64.1% [59.7, 68.3] of questions whose top 8 held
  evidence and 46.4% [41.6, 51.2] of those whose top 8 did not.
- The prompt grew by 7% (mean prompt tokens, from the model's own counts).

## Files

As in `../2026-11-01-qasper-dev`, for the fixed condition: `papers.jsonl`,
`retrieval.jsonl.gz`, `answers_fixed.jsonl`, `judgments_fixed.jsonl`, `summary.json`,
`summary.md`, `run.log`, plus `config.json`.

## Reproduce

```bash
eval/qasper/run_all.sh eval/results/2026-11-01-qasper-dev-c500-k8 http://127.0.0.1:11434 \
  --data eval/data --conditions fixed --chunk-size 500 --chunk-overlap 100 --k 8
```
