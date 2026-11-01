# QASPER dev: the app as shipped, fixed, and with oracle evidence, 2026-11-01

## Setup

- Script: `eval/qasper/run_eval.py`, launcher `eval/qasper/run_all.sh` (branch `fix/rag-pipeline`,
  commit `fb9dc9c`), run on maddog. Embedding and retrieval ran on CPU; the models ran on GPU in
  Ollama 0.12.6.
- Data: the QASPER v0.3 dev split (`qasper-train-dev-v0.3.tgz`): 281 papers and 1,005 questions.
  The PDFs came from `export.arxiv.org`, one every 3 s. 280 downloaded; 1 failed
  (`papers.jsonl` has sizes, SHA-256 hashes and the error).
- Pipeline: the app's own `Utils.split_pdf` (1,000-character chunks, 200 overlap),
  `LocalVectorStore` (all-MiniLM-L6-v2, FAISS) and `Utils.build_messages`, with the top 4 chunks.
- Answering model: `llama3.1:8b` (Q4_K_M), with the app's settings: temperature 0.1, at most
  1,000 tokens. Also seed 0 and a 16,384-token context. Each question was asked on its own.
- Conditions:
  - *shipped*: the system prompt alone, as the app sent the first question before the fix;
  - *fixed*: the fixed app's prompt with the 4 retrieved chunks;
  - *oracle*: the same prompt with the annotators' text evidence paragraphs instead.
- Judge: `qwen3:32b` (Q4_K_M), reasoning on, temperature 0, at most 6,000 tokens, an
  8,192-token context, and the prompt in `run_eval.py` (`JUDGE_PROMPT`). Empty answers are graded
  incorrect by rule, without the judge (`"by": "rule"`). The official QASPER evaluator gives
  Answer F1.
- The oracle answers were judged by a second Ollama server (GPU 3, port 11510;
  `oracle_judge_second_server.log`), to share the load. Each question was judged once
  (checked: no duplicate question ids).
- **Retry:** 3 fixed-condition judgements came back empty. The judge's reasoning had filled
  the 8,192-token window before it gave a verdict. They were judged again with
  `--retry-errors` (20,480-token window, 12,000 tokens), and the retries were appended to
  `judgments_fixed.jsonl` after the failed records. The last record per question counts. Before
  the retry, fixed was 536/1,002 correct; after it, 537/1,002.

## Results

| Condition | Questions | Judged correct | Correct or partial | Answer F1 (official) |
|---|---|---|---|---|
| shipped | 1,005 | 0.0% [0.0, 0.4] (0) | 0.0% [0.0, 0.4] | 0.0 |
| fixed | 1,002 | 53.6% [50.5, 56.7] (537) | 72.3% [69.4, 74.9] (724) | 11.6 |
| oracle | 892 | 69.1% [65.9, 72.0] (616) | 83.9% [81.3, 86.1] (748) | 19.0 |

Wilson 95% intervals.

- **Shipped:** all 1,005 answers were empty. Given only the system prompt, the model replied
  with nothing.
- **Fixed:** 3 questions belong to the paper whose PDF did not download and were not asked.
- **Oracle:** asked for the 892 questions whose annotators marked text evidence.
- **Retrieval:** a chunk of the PDF held the gold evidence for 815 of the 891 questions with
  text evidence and a PDF (91.5%). For those, the top 4 chunks held it for 48.8% [45.4, 52.3]
  (`summary.md` has the curve for k = 1–10).
- **Where answers are lost:** the fixed app was judged correct for 64.8% [60.0, 69.4] of
  questions whose top 4 chunks held evidence, and for 42.6% [38.3, 47.0] of those whose top 4
  did not.
- **Official F1** is low for every condition, because it compares tokens with short reference
  answers, and the app answers in full sentences. `summary.json` has it by answer type, along
  with judged correctness by the first annotator's answer type.

## Files

| File | Contents |
|---|---|
| `papers.jsonl` | Per paper: download status, size, SHA-256 |
| `retrieval.jsonl.gz` | Per question: chunk count, findability, evidence chunks, top 10 retrieved chunks (text, page, whether evidence) |
| `answers_<condition>.jsonl` | Per question: answer, token usage, seconds |
| `judgments_<condition>.jsonl` | Per question: verdict, explanation, the judge's reasoning, token usage |
| `summary.json`, `summary.md` | The numbers above |
| `run.log` | Console output, including a failed retry attempt (an import error from an old deployment) before the successful one |

## Reproduce

```bash
eval/qasper/run_all.sh eval/results/2026-11-01-qasper-dev http://127.0.0.1:11434 --data eval/data
uv run python eval/qasper/run_eval.py judge --out eval/results/2026-11-01-qasper-dev --conditions fixed oracle --retry-errors
uv run python eval/qasper/run_eval.py summarize --out eval/results/2026-11-01-qasper-dev
```
