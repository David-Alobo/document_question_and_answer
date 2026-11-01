# QASPER dev: 500-character chunks, top 8, reranked (candidate B), 2026-11-01

The second of the two leading set-ups from the retrieval sweep
(`../2026-11-01-retrieval-sweep-dev`), run on answers: as candidate A
(`../2026-11-01-qasper-dev-c500-k8`), but the top 20 chunks from MiniLM are reranked by a
cross-encoder before the top 8 are sent.

## Setup

As `../2026-11-01-qasper-dev` (fixed condition only), except for the retrieval settings:

- Code: branch `feat/multi-doc`, commit `80120bb`, copied to maddog. `config.json` records the
  settings.
- Retrieval: the app's `Utils.split_pdf` with 500-character chunks and 100 overlap,
  `LocalVectorStore` with all-MiniLM-L6-v2 and `reranker=Utils.load_reranker()`
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, reranking the top 20), the top 8 chunks.
- Same PDFs (`papers.jsonl`), same answering model and settings, same judge and prompt.
- The main Ollama server (port 11500), at the same time as candidate A on the other server, so
  `seconds` per answer depends on load and is not comparable between runs; prompt tokens are.
- The run was started in one queue with candidate A, stopped, and resumed on its own. Records
  written before the stop were kept; no question was answered or judged twice, apart from the
  retries below.
- **Retry:** 4 judgements came back empty (the judge's reasoning filled its window). All four
  succeeded on retry with `--retry-errors` (20,480-token window, 12,000 tokens): 2 correct, 1
  partially correct, 1 incorrect. All attempts are kept in `judgments_fixed.jsonl`; the last
  record per question counts.

## Results

| | Questions | Judged correct | Correct or partial | Answer F1 | Mean prompt tokens |
|---|---|---|---|---|---|
| App (1,000 / top 4), from `../2026-11-01-qasper-dev` | 1,002 | 53.6% [50.5, 56.7] | 72.3% | 11.6 | 1,154 |
| Candidate A (500 / top 8) | 1,002 | 57.1% [54.0, 60.1] | 75.2% | 11.6 | 1,230 |
| This run (500 / top 8, reranked) | 1,002 | 56.6% [53.5, 59.6] | 75.9% | 12.4 | 1,197 |

Wilson 95% intervals.

- **Paired against the app** on the same 1,002 questions: 166 became correct and 136 stopped
  being correct (exact McNemar; Holm-corrected over the two candidates, p = 0.095; candidate A:
  148 / 113, Holm p = 0.07). Neither candidate differs from the app at the 0.05 level after
  correction.
- **Against candidate A**: 114 questions became correct and 119 stopped being correct
  (p = 0.79). Reranking did not change answers.
- **Retrieval improved**: of the 832 findable questions (the same chunks as candidate A), the top
  8 held evidence for 64.4% [61.1, 67.6] against 57.6% [54.2, 60.9] without reranking.
  `summary.md` has the curve for k = 1–10.
- **Where answers are lost:** correct for 61.6% [57.4, 65.6] of questions whose top 8 held
  evidence and 48.5% [43.3, 53.6] of those whose top 8 did not.

The paired tests are computed by `paper/make_numbers.py` (function `candidates`).

## Files

As in `../2026-11-01-qasper-dev`, for the fixed condition: `papers.jsonl`,
`retrieval.jsonl.gz`, `answers_fixed.jsonl`, `judgments_fixed.jsonl`, `summary.json`,
`summary.md`, `run.log`, plus `config.json`.

## Reproduce

```bash
eval/qasper/run_all.sh eval/results/2026-11-01-qasper-dev-c500-k8-rerank http://127.0.0.1:11434 \
  --data eval/data --conditions fixed --chunk-size 500 --chunk-overlap 100 --k 8 --rerank
```
