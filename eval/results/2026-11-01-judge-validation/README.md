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

Read alone, Claude's labels suggested the judge's correct rate was within a few points of a
careful grader's. The human labels below do not support that.

## Human labels

`labels_human.csv` was made by a person (the first author), blind to the judge's verdicts and
to Claude's labels, on 2026-11-01.

- **How:** on a web page (`labelling_page.html`) that showed each question, the annotators'
  reference answers and the app's answer, with three buttons and an optional note. The page held
  no verdicts. Labels were saved as the buttons were pressed and exported to the CSV.
- **Instructions on the page:** judge the content against the references (one matching
  reference is enough); correct = gives the reference's content, extra detail allowed if it does
  not contradict; partially correct = part of it, or right content mixed with something wrong;
  incorrect = wrong or misses the point; "the paper does not say" is correct only when an
  annotator marked the question unanswerable; ignore page citations.
- **Examples given beforehand**, taken from the sample itself: #1 (incorrect: says the paper
  does not mention it, no annotator marked it unanswerable), #3 (either "yes" or "the paper does
  not say" would be correct), #5 (only "the paper does not say" is correct).
- The labeller graded against the references without reading the papers, as the judge does.

### Against the judge

| | Judge: correct | partially | incorrect |
|---|---|---|---|
| Human: correct | 15 | 2 | 0 |
| partially | 5 | 2 | 4 |
| incorrect | 7 | 4 | 11 |

- **Agreement:** exact on 28/50 (Cohen's κ = 0.32); correct-vs-not on 36/50 (κ = 0.45).
- **Direction:** the judge called 27 of the 50 correct, the human 17. In 12 answers the judge
  said correct and the human did not; in 2 the reverse.
- **Human against Claude:** exact on 32/50 (κ = 0.45).

### Rule check (exploratory, done after seeing the judge's verdicts)

Four human labels contradict the instructions above in cases where they are mechanical: one
annotator's reference matches the answer given, and one matching reference is enough.

| # | References | Answer | Human label |
|---|---|---|---|
| 3 | Yes, Unanswerable | the paper does not say | incorrect |
| 8 | Yes, Unanswerable | the paper does not say | incorrect |
| 37 | Unanswerable, PPDB and BookCorpus, Unanswerable | no dataset is named | incorrect |
| 49 | No, Yes | position is not given as input (= no) | partially correct |

`labels_human_rule_applied.csv` sets these four to correct and changes nothing else. Against
the judge: exact on 32/50 (κ = 0.43), correct-vs-not on 40/50 (κ = 0.60); human correct 21,
judge 27. The judge stays more lenient: 8 answers it called correct the human did not, against 2.
The labels as given are the primary result; this file is a sensitivity check, since the four were
reviewed knowing the judge's verdicts.

### What this means

- On this sample the judge credited 27/50 answers (54%), close to its 53.6% over all questions.
  The human credited 17/50 (34%), or 21/50 (42%) with the rule applied. The judged rate for the
  fixed app is therefore probably an upper bound.
- The judge's extra credit went mostly to hedged or incomplete answers that the human called
  partial or wrong.
- Paired comparisons between set-ups use the same judge on the same questions; if its leniency
  is similar across set-ups they are less affected. These labels do not test that.
- One labeller and 50 answers: the direction is clear, the size is not. A second labeller on the
  same 50 would show how much is the judge and how much the labeller.

`comparison_labels_human.json` and `comparison_labels_human_rule_applied.json` list every
disagreement with the human's note and the judge's explanation.
