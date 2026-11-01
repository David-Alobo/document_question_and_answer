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

## After the roadmap

1. **Final test and publication.** Run the final design once on the QASPER test split, with
   PaperQA2 on the same questions. Post the paper and SI on arXiv, archive code and results on
   Zenodo (DOI cited in the paper), choose the project name and tag v1.0; update the README
   and profile card.
2. **Beyond NLP papers.** Build a small question set from chemistry papers in the QASPER format
   (question, answer, evidence paragraph) and rerun the evaluation. Check what chemistry
   paper-QA benchmarks already exist before calling it new.
3. **Users** (optional). A public demo (e.g. a Hugging Face Space) and a small study in which
   colleagues ask real questions and record whether answers and page citations were right.
4. **One larger direction** (optional, chosen from what 3 shows): tables and figures; questions
   across many papers, closer to PaperQA2; or a vanilla JavaScript front end replacing
   Streamlit.

Items 1 and 2 come first: a held-out test and a second field strengthen the paper more than any
extra feature.

## Frameworks considered

- **LangChain** is already used for PDF loading and splitting. Moving more of the app onto it
  would not change answers.
- **LangGraph** is worth adopting only if the workflow grows branches, saved state or a human
  approval step. Items 2 and 3 above are written in plain Python first.
