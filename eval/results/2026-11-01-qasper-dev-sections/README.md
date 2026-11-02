# Section-aware chunking at the app's settings (1,000 characters, top 4), 2026-11-01

## Setup

- Script: `eval/qasper/run_eval.py --chunking sections` (commit `3c084b3`), stages `retrieve`,
  `generate`, `judge`, `summarize`, run on maddog 2026-11-01 23:10 to 2026-11-02 06:29
  (`run.log`).
- Chunking: `src/utils/sections.py` splits the PDF text at section headings found from the
  lines themselves (numbered, Roman-numeral and the usual unnumbered names), so no chunk spans two
  sections, and starts each chunk with "Section: <title>". Otherwise as the app: 1,000
  characters, 200 overlap, MiniLM, top 4 (`config.json`).
- Model, prompt, judge and settings as in `2026-11-01-qasper-dev`.
- `run_failed_no_papers_list.log` is a first launch that found no PDFs (its `papers.jsonl` was
  missing) and stopped; nothing from it is used.
- `section_coverage.py` (run 2026-11-02) counts how many of QASPER's section names the heading
  finder finds in the PDFs (`section_coverage.json`). A name such as "Experiments ::: Setup"
  counts by its last part; run-in paragraph headings ending in a full stop are left out.

## Results

- **Headings:** 2,919 of 3,621 QASPER section names found (80.6% [79.3, 81.9]), over 280 papers.
- **Retrieval:** of the 891 questions with text evidence, the top four held evidence for 47.5%
  [44.2, 50.8], against 44.7% [41.4, 47.9] with the app's chunking (117 gained, 92 lost; exact
  McNemar p = 0.097). The 500-character variant is in `2026-11-01-qasper-dev-sections-c500-k8`.
- **Answers:** 54.5% [51.4, 57.6] judged correct (546/1,002), against 53.6% for the app (139
  gained, 130 lost; p = 0.63).

## Files

| File | Contents |
|---|---|
| `retrieval.jsonl.gz` | Per question: the retrieved chunks and whether each holds evidence |
| `answers_fixed.jsonl`, `judgments_fixed.jsonl` | Answers and verdicts |
| `papers.jsonl`, `config.json` | The PDFs used and the retrieval settings |
| `summary.json`, `summary.md` | `run_eval.py summarize` output |
| `section_coverage.json` | QASPER section names found by the heading finder |
| `run.log`, `run_failed_no_papers_list.log` | Console output |
