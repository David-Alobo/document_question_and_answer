# Multi-document retrieval on QASPER dev, 2026-11-01

Does loading other papers hurt retrieval for a question about one of them?

## Setup

- Script: `eval/qasper/multidoc_eval.py` (branch `feat/multi-doc`), run on maddog on CPU.
- Data: the QASPER v0.3 dev split, 281 papers. 280 arXiv PDFs were downloaded (one failed) and
  split and embedded with the app's own settings (1,000-character chunks, 200 overlap,
  all-MiniLM-L6-v2).
- For each paper, 9 other dev papers were drawn at random (seed 0). With N = 0, 2, 4 or 9 of them
  loaded alongside it, the app's `LocalVectorStore` was searched with three rules:
  - *single*: the top 4 chunks (the rule with one file).
  - *multi*: the top 4, then the best chunk of each other file, up to 8 (the rule with several
    files and "All documents").
  - *selected*: the top 4 chunks of the question's own paper (the user picks it under
    "Ask about").
- A question counts if a chunk of its own paper holds gold evidence (matched as in
  `run_eval.py`). That leaves 815 questions.

## Results

| Other papers | Rule | Evidence in context | Chunks in context (mean) | From the right paper (mean) | Lost / gained vs alone |
|---|---|---|---|---|---|
| 0 | any | 48.8% [45.4, 52.3] (398/815) | 4.00 | 4.00 | — |
| 2 | single | 37.7% [34.4, 41.0] | 4.00 | 2.78 | 91 / 0 |
| 2 | multi | 40.2% [36.9, 43.7] | 5.49 | 2.91 | 70 / 0 |
| 4 | single | 32.3% [29.1, 35.6] | 4.00 | 2.38 | 135 / 0 |
| 4 | multi | 36.4% [33.2, 39.8] | 7.27 | 2.59 | 101 / 0 |
| 9 | single | 28.1% [25.1, 31.3] | 4.00 | 2.02 | 169 / 0 |
| 9 | multi | 31.9% [28.8, 35.2] (260/815) | 8.00 | 2.23 | 138 / 0 |
| 2, 4, 9 | selected | 48.8% [45.4, 52.3] | 4.00 | 4.00 | 0 / 0 |

Intervals are Wilson 95%. "Lost / gained" counts questions whose evidence left or entered the
context compared with the paper loaded alone.

- Other papers crowd out the one a question is about. With 9 others loaded, about half of the top
  4 chunks come from other papers, and 138 of the 398 questions whose evidence was retrieved
  lose it. No question gained.
- The several-files rule recovers a little (31.9% against 28.1% with 9 others), mostly by
  widening the context to 8 chunks.
- Choosing the paper under "Ask about" restores the single-paper result exactly, as it must: it
  searches the same chunks with the same embeddings.

This measures retrieval only, not answers. It is a lower bound in one respect: the evidence
match needs 5-word overlaps with QASPER's paragraph text, so a chunk that paraphrases the
evidence after PDF extraction is not counted.

## Files

| File | Contents |
|---|---|
| `results.csv.gz` | One row per question, N and rule: evidence in context, chunks in context, chunks from the right paper, rank of the first evidence chunk |
| `summary.json`, `summary.md` | The table above |
| `distractors.json` | Seed and number of papers |
| `run.log` | Console output of both runs |
| `run1_without_selected/` | The first run, before the *selected* rule existed. All 6,520 of its rows are identical in the rerun. |

## Reproduce

```bash
uv run python eval/qasper/run_eval.py prepare --out <any folder>        # QASPER and the PDFs
uv run python eval/qasper/multidoc_eval.py run --data eval/data --out eval/results/2026-11-01-multidoc-dev
uv run python eval/qasper/multidoc_eval.py summarize --out eval/results/2026-11-01-multidoc-dev
```
