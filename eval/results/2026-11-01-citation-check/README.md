# Citation check: do the fixed app's cited pages hold its claims? 2026-11-01

## Setup

- Script: `eval/qasper/citation_check.py` (commit `3c084b3`), on the seed-0 fixed answers of
  `2026-11-01-qasper-dev`.
- `claims`: each answer is split into sentences that carry a page citation; a sentence ending in
  a colon takes the list after it. Every citation is checked against the pages of the four chunks
  the model was given. One cited statement per answer, with all its pages in the context, is drawn
  at random (seed 0).
- `judge`: `qwen3:32b` (temperature 0, reasoning on) is shown the statement and the passages from
  its cited pages, and says whether they support it (supported, partially supported, not
  supported; `SUPPORT_PROMPT` in the script). Run on maddog, 2026-11-01 to 2026-11-02 00:38
  (`console.log`). The first launch was stopped part-way; a second launch judged the remaining
  781. `support_judgments.jsonl` has 925 records for 915 statements: 10 were judged in both
  launches, and the last record per statement counts.
- Hand labels: 50 of the drawn statements (`to_label_by_hand.csv`), labelled by one of us, blind
  to the judge, on `labelling_page.html`, which shows the question, the statement and the passages
  as the model saw them.
- `summarize --labels labels_human.csv` writes `summary.json`.

## Results

- **Citations:** 935 of 1,002 answers (93.3%) cite at least one page. Of 1,588 page citations,
  1,528 (96.2% [95.2, 97.1]) point to a page in the model's context; 60 do not.
- **Judge, all 915 checked statements:** 78.4% [75.6, 80.9] supported, 15.0% partially, 6.7% not.
- **Hand labels (50):** the labeller found 42 supported, 2 partially and 6 not; the judge 45, 4
  and 1. Exact agreement 39 of 50, Cohen's κ = 0.08. The judge called supported all 6 statements
  the labeller marked not supported, so its verdicts are not used as a failure rate.

## Files

| File | Contents |
|---|---|
| `statements.jsonl.gz` | Per answer: its cited statements, each citation and whether its page was in context |
| `counts.json` | Answers, statements and citations counted |
| `drawn.json` | The statement drawn per answer, with the passages from its cited pages |
| `support_judgments.jsonl` | The judge's verdict and reasoning per drawn statement |
| `to_label_by_hand.csv`, `labelling_page.html` | The 50-statement sample and the page used to label it |
| `labels_human.csv` | The hand labels |
| `summary.json` | Rates, support by the answer's own verdict, and agreement with the hand labels |
| `summary_sample_check.json` | An earlier summary made while the judge was still running (207 statements); superseded by `summary.json` |
| `console.log` | Console output of the judge stage |
