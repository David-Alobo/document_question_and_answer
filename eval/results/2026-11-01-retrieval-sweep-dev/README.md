# Retrieval sweep on QASPER dev, 2026-11-01

Which chunking, embedding model, keyword search and reranking put the evidence in front of the
model most often? The sweep is retrieval only, with no language model.

## Setup

- Script: `eval/qasper/retrieval_sweep.py`, launcher `eval/qasper/run_sweep.sh`, run on maddog on
  CPU in 4 shards of 6 threads (branch `feat/better-retrieval`, commit `293ae33` plus sharding).
- Data: QASPER v0.3 dev, using the 280 arXiv PDFs downloaded by `run_eval.py prepare`.
- There are 36 set-ups, 12 retrievers × 3 chunkings:
  - **Chunking** (size/overlap in characters): 500/100, 1000/200 (the app's), 1500/300.
  - **Retrievers:**
    - dense: all-MiniLM-L6-v2 (the app's), BAAI/bge-small-en-v1.5, intfloat/e5-small-v2;
    - BM25 (`rank-bm25`);
    - hybrid: dense plus BM25 by reciprocal rank fusion (k = 60);
    - reranked: MiniLM dense, BM25 and the three hybrids, each with its top 20 reranked by
      cross-encoder/ms-marco-MiniLM-L-6-v2.
- For each question and set-up, the script records the rank of the first chunk holding gold
  evidence, matched as in `run_eval.py`.
- Questions: the 782 whose evidence is findable under every chunking, so every set-up is scored
  on the same questions.
- **hit@budget** counts a hit within about 4,000 characters of context (k = 8, 4 or 3 chunks of
  500, 1000 or 1500 characters). This compares set-ups at the same prompt length. hit@4 does
  not: four 1,500-character chunks hold three times the text of four 500-character ones.
- Paired exact McNemar tests compare each set-up with the app's (`c1000/dense-minilm`), with Holm
  correction over the 35 comparisons, separately for hit@4 and hit@budget (`summary.json`).
  (An earlier version of this README and the commit that added the sweep said 33 set-ups and 32
  comparisons; the code always ran and corrected over all 36 and 35.)

## Results at equal prompt length (hit@budget)

| Set-up | hit@budget [95% CI] | k | Gained / lost vs app | Holm p |
|---|---|---|---|---|
| `c500/rerank-hybrid-bge` | 65.9% [62.5, 69.1] | 8 | +207 / −78 | 4.0e-13 |
| `c500/rerank-dense-minilm` | 65.5% [62.1, 68.7] | 8 | +194 / −68 | 1.1e-13 |
| `c500/rerank-hybrid-e5` | 65.0% [61.6, 68.2] | 8 | +205 / −83 | 1.4e-11 |
| `c500/hybrid-bge` | 64.3% [60.9, 67.6] | 8 | +201 / −84 | 9.8e-11 |
| `c500/dense-bge` | 63.0% [59.6, 66.4] | 8 | +187 / −80 | 1.5e-09 |
| `c1000/rerank-hybrid-minilm` | 58.7% [55.2, 62.1] | 4 | +149 / −76 | 3.1e-05 |
| `c500/dense-minilm` | 58.4% [55.0, 61.8] | 8 | +139 / −68 | 2.3e-05 |
| **`c1000/dense-minilm` (the app)** | **49.4% [45.9, 52.9]** | 4 | — | — |

`summary.md` has every set-up, with hit@1, hit@4, hit@10 and MRR@10.

What the sweep shows:
1. **Smaller chunks, more of them:** 500-character chunks, 8 at a time, beat 1,000-character
   chunks, 4 at a time, with the same model (58.4% against 49.4%). That needs no new model.
2. **Reranking adds about 7 points on top** (65.5% with MiniLM). It costs one extra small model
   and about 20 cross-encoder scores per question.
3. Once reranking is used, the embedding model and BM25 matter little: the top 5 set-ups are
   within 1.2 points, inside each other's intervals.
4. BM25 alone is worst at every chunk size.

## Caveat: the evidence match may favour small chunks

A chunk counts as holding evidence if it shares half of an evidence paragraph's word 5-grams, or
if half of the chunk's own 5-grams lie in the paragraph. A 500-character chunk inside a long
evidence paragraph passes the second test while holding only part of the paragraph, perhaps not
the part with the answer. So part of the advantage of 500-character chunks may come from the
metric. **The shortlisted set-ups must therefore be confirmed on answers** (generation and
judging with `run_eval.py`), not shipped on this table alone.

## Files

| File | Contents |
|---|---|
| `ranks_0.csv.gz` … `ranks_3.csv.gz` | One row per question, chunking and retriever: number of chunks, whether evidence is findable, rank of the first evidence chunk (one file per shard) |
| `summary.json`, `summary.md` | All metrics, intervals and paired tests |
| `run.log`, `shard_*.log` | Console output |

## Reproduce

```bash
uv sync --group eval
eval/qasper/run_sweep.sh eval/results/2026-11-01-retrieval-sweep-dev 4 6 --data eval/data
```
