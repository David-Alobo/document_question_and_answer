# Section-aware chunking at 500 characters, top 8: retrieval only, 2026-11-01

## Setup

- Script: `eval/qasper/run_eval.py retrieve --chunking sections --chunk-size 500
  --chunk-overlap 100 --k 8` (commit `3c084b3`), run on maddog 2026-11-01 23:10 (`run.log`).
- Chunking as in `2026-11-01-qasper-dev-sections`, at the settings of the
  `2026-11-01-qasper-dev-c500-k8` set-up (`config.json`). Retrieval only; no answers were
  generated.
- `run_failed_no_papers_list.log` is a first launch that found no PDFs and stopped; nothing from
  it is used.

## Results

Of the 891 questions with text evidence, the top eight held evidence for 55.4% [52.2, 58.7],
against 53.8% [50.5, 57.0] with fixed 500-character chunks (94 gained, 79 lost; exact McNemar
p = 0.29). Computed by `paper/make_numbers.py` from the two `retrieval.jsonl.gz` files.

## Files

| File | Contents |
|---|---|
| `retrieval.jsonl.gz` | Per question: the retrieved chunks and whether each holds evidence |
| `papers.jsonl`, `config.json` | The PDFs used and the retrieval settings |
| `run.log`, `run_failed_no_papers_list.log` | Console output |
