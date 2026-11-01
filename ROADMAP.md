# Roadmap

Every change below is tested on the QASPER development set with the scripts in `eval/qasper/`,
paired against the app's current defaults, and its results folder kept in `eval/results/`.
The test split is reserved for the final comparison.

## Now

- Paper and SI: human labels for the judge check, the remaining SI sections.
- Done: retrieval candidates on answers (500-character chunks, top 8, with and without
  reranking). Neither beat the app's settings after correction, so the defaults are unchanged.

## The paper

One paper. Working title: *Beyond retrieval: evaluating the reliability of retrieval-augmented
question answering over scientific papers*. The title is settled once the results below are in;
if an improvement in section C works clearly, the "beyond retrieval" framing is revisited.

### A. Evaluation (written)

The shipped bug, the fixed app against the oracle, the retrieval sweep, the two candidates on
answers, crowding by other papers, and the judge check.

### B. Reliability (needed for the word "reliability")

1. **Repeat runs.** Ask every question three times with different seeds and report how often
   the judged verdict changes, overall and per set-up. Tells which of the paper's differences
   exceed run-to-run variation.
2. **Citation check.** For a random sample of answers, check whether the cited pages hold the
   claims (by hand, with the judge as a second rater). Answers can be judged correct while
   citing the wrong page.

### C. Improvements (a section of the paper)

Each one is tested on answers, paired against the app, with the same judge.

3. **Section-aware chunking** (the idea behind STAIR, arXiv 2609.03874: retrieve along the
   document's structure, not fixed-length chunks). Split on section headings, put the section
   title in front of each chunk, optionally rank sections before chunks. Test in the retrieval
   sweep first (no language model), then on answers. The STAIR model itself is a generative
   retriever trained on a fixed corpus, so it does not suit PDFs uploaded at runtime.
4. **Per-paper comparisons.** For "compare these documents", gather evidence from each paper
   separately (e.g. 2 chunks per file), then compare. Targets the mixed-up papers seen in
   `eval/results/2026-11-01-ui-check-multidoc`; needs a small set of comparison questions.
5. **Retry when retrieval misses.** Ask the model whether the chunks answer the question; if
   not, rewrite the query and search again. Costs an extra model call per question.

Whatever works becomes the app's default; whatever does not is reported as a null result.

### D. Generalisation and final test

6. **Chemistry papers.** Build a small question set from chemistry papers in the QASPER format
   (question, answer, evidence paragraph) and rerun the evaluation. Check what chemistry
   paper-QA benchmarks already exist before calling it new.
7. **Final test.** Run the final design once on the QASPER test split, with PaperQA2 on the
   same questions.

Order: B, then C, then D. The test split (7) is run last, once the design is fixed.

### E. Publication

Post the paper and SI on arXiv, archive code and results on Zenodo (DOI cited in the paper),
choose the project name and tag v1.0; update the README and profile card.

## After the paper (optional)

- **Users.** A public demo (e.g. a Hugging Face Space) and a small study in which colleagues
  ask real questions and record whether answers and page citations were right.
- **One larger direction**, chosen from what the users show: tables and figures; questions
  across many papers, closer to PaperQA2; or a vanilla JavaScript front end replacing
  Streamlit.

## Frameworks considered

- **LangChain** is already used for PDF loading and splitting. Moving more of the app onto it
  would not change answers.
- **LangGraph** is worth adopting only if the workflow grows branches, saved state or a human
  approval step. Items 4 and 5 above are written in plain Python first.
