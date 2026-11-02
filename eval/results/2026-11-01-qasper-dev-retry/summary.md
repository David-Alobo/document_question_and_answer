# QASPER dev: 1005 questions, 281 papers (280 PDFs downloaded)

## Retrieval (questions with text evidence)

Evidence present in the PDF's chunks: 91.5% [89.5, 93.1] (815/891)

| k | Evidence in top k (of findable) |
|---|---|
| 1 | 19.8% [17.2, 22.6] (161/815) |
| 2 | 31.8% [28.7, 35.1] (259/815) |
| 3 | 42.1% [38.7, 45.5] (343/815) |
| 4 | 49.2% [45.8, 52.6] (401/815) |
| 5 | 54.6% [51.2, 58.0] (445/815) |
| 6 | 59.1% [55.7, 62.5] (482/815) |
| 7 | 63.2% [59.8, 66.4] (515/815) |
| 8 | 66.7% [63.4, 69.9] (544/815) |
| 9 | 69.1% [65.8, 72.2] (563/815) |
| 10 | 70.9% [67.7, 73.9] (578/815) |

## Answers (judged)

| Condition | Questions | Correct | Correct or partial | Answer F1 (official) |
|---|---|---|---|---|
| fixed | 1002 | 54.8% [51.7, 57.8] (549/1002) | 73.6% [70.7, 76.2] (737/1002) | 11.7 |

Fixed app, correct when the top 4 chunks held evidence: 66.3% [61.6, 70.8] (266/401); when they did not: 43.3% [38.9, 47.7] (212/490)
