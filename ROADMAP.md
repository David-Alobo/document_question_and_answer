# Roadmap

Every change below is tested on the QASPER development set with the scripts in `eval/qasper/`,
paired against the app's current defaults, and its results folder kept in `eval/results/`.
The test split is reserved for the final comparison.

## Now

- Retrieval candidates on answers: 500-character chunks, top 8, with and without the
  cross-encoder reranker, against the fixed app. Switch the defaults if a candidate wins.
- Paper and SI: candidate results, human labels for the judge check, the remaining SI sections.

## Next

1. **Section-aware chunking** (the idea behind STAIR, arXiv 2609.03874: retrieve along the
   document's structure, not fixed-length chunks). Split on section headings, put the section
   title in front of each chunk, optionally rank sections before chunks. Test in the retrieval
   sweep first (no language model), then on answers. The STAIR model itself is a generative
   retriever trained on a fixed corpus, so it does not suit PDFs uploaded at runtime.
2. **Per-paper comparisons.** For "compare these documents", gather evidence from each paper
   separately (e.g. 2 chunks per file), then compare. Targets the mixed-up papers seen in
   `eval/results/2026-11-01-ui-check-multidoc`.
3. **Retry when retrieval misses.** Ask the model whether the chunks answer the question; if
   not, rewrite the query and search again. Costs an extra model call per question.

## Later

- Final comparison on the QASPER test split, and PaperQA2 on the same questions.
- Choose the project name; update the README and profile card.

## Frameworks considered

- **LangChain** is already used for PDF loading and splitting. Moving more of the app onto it
  would not change answers.
- **LangGraph** is worth adopting only if the workflow grows branches, saved state or a human
  approval step. Items 2 and 3 above are written in plain Python first.
