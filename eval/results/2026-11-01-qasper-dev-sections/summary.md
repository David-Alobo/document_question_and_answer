# QASPER dev: 1005 questions, 281 papers (280 PDFs downloaded)

## Retrieval (questions with text evidence)

Evidence present in the PDF's chunks: 92.9% [91.1, 94.4] (828/891)

| k | Evidence in top k (of findable) |
|---|---|
| 1 | 20.8% [18.1, 23.7] (172/828) |
| 2 | 34.9% [31.7, 38.2] (289/828) |
| 3 | 45.4% [42.0, 48.8] (376/828) |
| 4 | 51.1% [47.7, 54.5] (423/828) |
| 5 | 57.0% [53.6, 60.3] (472/828) |
| 6 | 61.5% [58.1, 64.7] (509/828) |
| 7 | 66.2% [62.9, 69.3] (548/828) |
| 8 | 69.1% [65.9, 72.1] (572/828) |
| 9 | 72.3% [69.2, 75.3] (599/828) |
| 10 | 74.3% [71.2, 77.1] (615/828) |

## Answers (judged)

| Condition | Questions | Correct | Correct or partial | Answer F1 (official) |
|---|---|---|---|---|
| fixed | 1002 | 54.5% [51.4, 57.6] (546/1002) | 74.4% [71.6, 77.0] (745/1002) | 11.2 |

Fixed app, correct when the top 4 chunks held evidence: 67.4% [62.8, 71.7] (285/423); when they did not: 39.1% [34.8, 43.6] (183/468)
