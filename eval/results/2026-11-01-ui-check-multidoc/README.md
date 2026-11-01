# UI check with two documents, 2026-11-01

An end-to-end check of the multi-document app (branch `feat/multi-doc`, commit `83600e7`) in
headless Chrome. The setup is as in `../2026-11-01-ui-check`: the unchanged app, with Groq
requests sent to Llama 3.1 8B in Ollama through `scripts/groq_to_ollama.py`, and every request
logged to `model_requests.jsonl`.

Papers (QASPER dev; hashes in `pdf_sha256.txt`):
- `1810.08699.pdf`, *pioNER: Datasets and Baselines for Armenian Named Entity Recognition* (40 chunks)
- `1912.01214.pdf`, *Cross-lingual Pre-training Based Transfer for Zero-shot Neural Machine Translation* (49 chunks)

## Steps and results

| Step | Result | Screenshot |
|---|---|---|
| Upload the NER paper; ask "What NER models were evaluated?" | Stanford NER, spaCy 2.0 and the biLSTM-CRF model, the annotators' answer, from pages 4–5 | `01_one_file.png` |
| Add the translation paper | "2 documents ready" after 2.0 s; the first file was not embedded again | `02_two_files.png` |
| "Compare these documents" with "All documents" | Compares the two papers, labelling each source by file and page | `03_compare.png` |
| "Ask about" set to the translation paper; "Which datasets were used for zero-shot translation?" | All 4 passages come from that paper | `04_question_about_one_paper.png` |

## A weakness this exposed

For the comparison, the top 4 passages all came from the NER paper. The translation paper
contributed only its one guaranteed passage, a results table. The answer then mixed the papers:
it says both use test sets in "Armenian, English, German, Spanish and Russian". A comparison
needs balanced evidence, for example 2 passages from each file. That change can be measured
with `eval/qasper/multidoc_eval.py`.

## Reproduce

As in `../2026-11-01-ui-check`, with
`uv run --no-project --with websocket-client python scripts/drive_multi.py http://127.0.0.1:8599 1810.08699.pdf screenshots 1912.01214.pdf`.
`drive_multi.py` imports nothing from `drive.py`; both are kept for reference.
