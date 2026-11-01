# UI check, 2026-11-01

An end-to-end check of the fixed Streamlit app (branch `fix/rag-pipeline`) in headless Chrome:
upload a paper, ask a question, ask a follow-up, clear the conversation.

## Setup

- The app ran unchanged with `GROQ_BASE_URL=http://127.0.0.1:8600 GROQ_API_KEY=local-test`. The
  Groq SDK sends requests to `GROQ_BASE_URL`, where `scripts/groq_to_ollama.py` forwarded them to
  Llama 3.1 8B in Ollama (the weights behind Groq's `llama-3.1-8b-instant`), mapping the model
  name and logging every request to `model_requests.jsonl`.
- `scripts/drive.py` drove Chrome through the DevTools protocol: it chose the file in the
  uploader, typed into the chat box, clicked the buttons, and took the screenshots.
- Paper: arXiv 1912.01214 (*Cross-lingual Pre-training Based Transfer for Zero-shot Neural Machine
  Translation*), a QASPER dev paper; SHA-256 in `pdf_sha256.txt`. 49 chunks.

## What it showed

| Step | Result | Screenshot |
|---|---|---|
| Upload | "Document ready: 49 chunks" | `02_ready.png` |
| "Which datasets did they experiment with?" | Names MultiUN and Europarl (the annotators' answer), cites p. 4, lists 4 source passages with pages | `03_answer_with_sources.png` |
| "Can you elaborate on that?" (example button) | Answered with the earlier exchange in context | `04_follow_up.png` |
| Clear conversation | 0 answers shown afterwards | `05_cleared.png` |

`model_requests.jsonl` shows what reached the model. The first request has the system prompt and
one user message holding the 4 passages and the question. The second has the system prompt, the
first exchange once, then the new question with its passages. Before the fix, the first request
would have held only the system prompt.

**A weakness this exposed:** the follow-up "Can you elaborate on that?" is used as the search query
as written, so it retrieved passages about freezing encoder layers (p. 7) rather than about the
datasets, and the answer drifted to that topic. Rewriting a follow-up into a standalone question
before retrieval would fix this; it is on the roadmap.

## Reproduce

```bash
ssh -N -L 11501:127.0.0.1:11500 maddog            # or any Ollama with llama3.1:8b
python scripts/groq_to_ollama.py 8600 http://127.0.0.1:11501
GROQ_BASE_URL=http://127.0.0.1:8600 GROQ_API_KEY=local-test \
    uv run streamlit run main.py --server.headless true --server.port 8599
uv run --no-project --with websocket-client python scripts/drive.py http://127.0.0.1:8599 1912.01214.pdf screenshots
```
