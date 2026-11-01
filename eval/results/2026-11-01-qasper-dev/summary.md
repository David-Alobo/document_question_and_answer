# QASPER dev: 1005 questions, 281 papers (280 PDFs downloaded)

## Retrieval (questions with text evidence)

Evidence present in the PDF's chunks: 91.5% [89.5, 93.1] (815/891)

| k | Evidence in top k (of findable) |
|---|---|
| 1 | 19.4% [16.8, 22.2] (158/815) |
| 2 | 31.5% [28.4, 34.8] (257/815) |
| 3 | 42.1% [38.7, 45.5] (343/815) |
| 4 | 48.8% [45.4, 52.3] (398/815) |
| 5 | 55.2% [51.8, 58.6] (450/815) |
| 6 | 60.6% [57.2, 63.9] (494/815) |
| 7 | 65.6% [62.3, 68.8] (535/815) |
| 8 | 69.9% [66.7, 73.0] (570/815) |
| 9 | 73.0% [69.9, 75.9] (595/815) |
| 10 | 75.3% [72.3, 78.2] (614/815) |

## Answers (judged)

| Condition | Questions | Correct | Correct or partial | Answer F1 (official) |
|---|---|---|---|---|
| shipped | 1005 | 0.0% [0.0, 0.4] (0/1005) | 0.0% [0.0, 0.4] (0/1005) | 0.0 |
| fixed | 1002 | 53.6% [50.5, 56.7] (537/1002) | 72.3% [69.4, 74.9] (724/1002) | 11.6 |
| oracle | 892 | 69.1% [65.9, 72.0] (616/892) | 83.9% [81.3, 86.1] (748/892) | 19.0 |

Fixed app, correct when the top 4 chunks held evidence: 64.8% [60.0, 69.4] (258/398); when they did not: 42.6% [38.3, 47.0] (210/493)
