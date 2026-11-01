# Multi-document retrieval: 815 questions with findable evidence

Each question's paper is loaded with N other dev papers (seed 0). *single*: top 4 chunks; *multi*: top 4 plus the best chunk of each other file, up to 8 (the app's rule with several files). Lost/gained: questions whose evidence left/entered the context compared with the paper alone.

| Other papers | Rule | Evidence in context | Context chunks (mean) | From the right paper (mean) | Lost / gained |
|---|---|---|---|---|---|
| 0 | multi | 48.8% [45.4, 52.3] (398/815) | 4.00 | 4.00 | 0 / 0 |
| 0 | single | 48.8% [45.4, 52.3] (398/815) | 4.00 | 4.00 | 0 / 0 |
| 2 | multi | 40.2% [36.9, 43.7] (328/815) | 5.49 | 2.91 | 70 / 0 |
| 2 | single | 37.7% [34.4, 41.0] (307/815) | 4.00 | 2.78 | 91 / 0 |
| 4 | multi | 36.4% [33.2, 39.8] (297/815) | 7.27 | 2.59 | 101 / 0 |
| 4 | single | 32.3% [29.1, 35.6] (263/815) | 4.00 | 2.38 | 135 / 0 |
| 9 | multi | 31.9% [28.8, 35.2] (260/815) | 8.00 | 2.23 | 138 / 0 |
| 9 | single | 28.1% [25.1, 31.3] (229/815) | 4.00 | 2.02 | 169 / 0 |
