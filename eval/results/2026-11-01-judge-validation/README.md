# Judge validation, 2026-11-01

How far can the Qwen3-32B judge's verdicts be trusted?

## Sample

50 of the 1,002 non-empty fixed-app answers in `../2026-11-01-qasper-dev`, drawn with
`random.Random(0).sample` from the answers sorted by question id (`sample.json`: question,
references, answer).

## Labels so far

`labels_claude.csv` was made by **Claude, the AI assistant that ran this evaluation, not by a
human**:
- It was made before reading any of the judge's verdicts for these questions, and committed
  (`193f02c`) before the comparison was run.
- The grades and rules are the judge's (correct / partially_correct / incorrect; "the paper
  does not say" is correct only if an annotator marked the question unanswerable).
- Each label is marked *clear* or *borderline*, with a note.

These labels are a cross-check by a second, different model. **They are not human labels.**
`to_label_by_hand.csv` has the same 50 answers with no label or verdict, for a human to label
blind. Their labels should be saved as `labels_<name>.csv` (columns `question_id,label`) and
compared with `python compare.py labels_<name>.csv`.

## Claude's labels against the judge

| | Judge: correct | partially | incorrect |
|---|---|---|---|
| Claude: correct | 21 | 4 | 0 |
| partially | 4 | 3 | 2 |
| incorrect | 2 | 1 | 13 |

- **Agreement:** exact on 37/50 (Cohen's κ = 0.57); correct-vs-not on 40/50 (κ = 0.60).
- **Agreement by confidence:** 33/40 on the answers Claude marked *clear*, and 4/10 on the
  *borderline* ones.
- **No net direction in this sample:** the judge called 27 of the 50 correct, Claude 25.
- **Where the judge erred:**
  - In two clear cases it credited an answer that did not answer. "Is the dataset balanced?"
    (reference: No) was answered "the paper does not say". "Only English?" (reference: Yes) got
    a hedge.
  - In four it called a complete answer partial because the answer added or left out detail.
- `comparison_labels_claude.json` lists every disagreement with both explanations.

What this supports: the judge's correct rate on the fixed app is probably within a few points of
what a careful grader would give, with errors in both directions. What it does not support:
a judge accuracy figure for the paper, which needs human labels.
