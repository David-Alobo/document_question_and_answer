"""Retry when retrieval misses: ask the model whether the passages answer the question first.

For each question, the app's four retrieved chunks (from a saved run) are shown to the answer
model with the question, and the model says whether they hold the answer. If it says no, it
writes a search query, the paper is searched again with that query, and the top four chunks of
the new search replace the first ones. The answer is then generated from the final chunks exactly
as the app does. The prompt length stays the same as the app's; the cost is one extra model call
per question, two when the model says no.

The output folder has the same files as a run_eval.py run (papers.jsonl, config.json,
retrieval.jsonl.gz with the final chunks, answers_fixed.jsonl), plus retry.jsonl with each
question's check and query, so run_eval.py judge and summarize work on it unchanged.

Usage:
  python eval/qasper/retry_eval.py --base eval/results/2026-11-01-qasper-dev --out eval/results/<folder> \
      --data ../data --ollama http://127.0.0.1:11500
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from run_eval import (
    OllamaClient,
    _append_jsonl,
    _config,
    _load_split,
    _messages,
    _ngrams,
    _read_jsonl,
    _tokens,
    matches,
    references,
)

from src.utils.utils import Utils

CHECK_PROMPT = """Below are passages from a research paper, each starting with its page number, and a question about the paper.

<passages>
{context}
</passages>

<question>
{question}
</question>

Do these passages contain the information needed to answer the question? Reply with only YES or NO."""

REWRITE_PROMPT = """The passages retrieved from a research paper did not answer this question:

<question>
{question}
</question>

Write one search query that would find the part of the paper that answers it: the key terms a
paragraph answering it would contain, as plain words without quotes or operators such as AND
and OR. Reply with only the query."""


def run(args):
    """Check, retry and answer every question; write the run files.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    from langchain_core.documents import Document

    from src.database.vectorstore import LocalVectorStore

    base, out, data_dir = Path(args.base), Path(args.out), Path(args.data)
    out.mkdir(parents=True, exist_ok=True)
    for name in ("papers.jsonl", "config.json"):
        if (base / name).exists() and not (out / name).exists():
            shutil.copy(base / name, out / name)
    config = _config(argparse.Namespace(chunk_size=None, chunk_overlap=None, k=None, rerank=None), base)
    k = config["k"]
    papers = _load_split(data_dir, "dev", args.limit)
    questions = {q["question_id"]: q for p in papers.values() for q in p["qas"]}
    rows = [r for r in _read_jsonl(base / "retrieval.jsonl.gz") if r["paper_id"] in papers]
    done = {r["question_id"] for r in _read_jsonl(out / "answers_fixed.jsonl")}
    client = OllamaClient(args.ollama, seed=args.seed)
    model = Utils.load_embedding_model()

    def ask(prompt, max_tokens):
        """One short model call at temperature 0.

        Args:
            prompt (str): The user message.
            max_tokens (int): Token limit.

        Returns:
            str: The reply, stripped.
        """
        reply = client.create(messages=[{"role": "user", "content": prompt}], model=args.model,
                              temperature=0.0, max_tokens=max_tokens)
        return reply.choices[0].message.content.strip()

    by_paper = {}
    for r in rows:
        if r["question_id"] not in done:
            by_paper.setdefault(r["paper_id"], []).append(r)
    for paper_id, paper_rows in by_paper.items():
        store, chunk_grams = None, []
        if not paper_rows[0]["error"]:
            chunks = Utils.split_pdf(data_dir / "pdfs" / f"{paper_id}.pdf", config["chunk_size"],
                                     config["chunk_overlap"], sections=config.get("chunking") == "sections")
            store = LocalVectorStore(model)
            store.add_documents(chunks)
            chunk_grams = [_ngrams(_tokens(c.page_content)) for c in chunks]
            index = {id(c): i for i, c in enumerate(chunks)}

        def check(row):
            """Ask whether the first chunks answer the question, and for a query if not.

            Args:
                row (dict): The question's saved retrieval record.

            Returns:
                dict: The check reply and, if it was not yes, the rewritten query.
            """
            if row["error"]:
                return {"check": None, "query": None}
            docs = [Document(page_content=c["text"], metadata={"page": c["page"]}) for c in row["retrieved"][:k]]
            context = Utils.format_context(docs)
            reply = ask(CHECK_PROMPT.format(context=context, question=row["question"]), 5)
            if reply.upper().startswith("YES"):
                return {"check": reply, "query": None}
            return {"check": reply, "query": ask(REWRITE_PROMPT.format(question=row["question"]), 60)}

        with ThreadPoolExecutor(args.workers) as pool:
            checks = list(pool.map(check, paper_rows))

        finals = []
        for row, c in zip(paper_rows, checks):
            new = dict(row, retry=c)
            if c["query"] and store is not None:
                q = questions[row["question_id"]]
                para_grams = [_ngrams(_tokens(ev)) for r in references(q) for ev in r["evidence"]]
                found = {i for i, cg in enumerate(chunk_grams) if any(matches(cg, pg) for pg in para_grams)}
                new["retrieved"] = [
                    {"chunk": index[id(d)], "page": d.metadata.get("page"), "evidence": index[id(d)] in found,
                     "text": d.page_content}
                    for d in store.similarity_search(c["query"], k=k)
                ]
            finals.append(new)

        def answer(row):
            """Answer from the final chunks with the app's call.

            Args:
                row (dict): The retrieval record with the final chunks.

            Returns:
                dict: The answer record, as run_eval.py writes it.
            """
            q = questions[row["question_id"]]
            result = {"question_id": row["question_id"], "condition": "fixed", "model": args.model,
                      "seed": args.seed}
            messages = _messages("fixed", row, q["question"], references(q), k)
            if messages is None:
                result.update(answer=None, skipped=True)
                return result
            start = time.time()
            try:
                response = client.chat.completions.create(messages=messages, model=args.model, temperature=0.1,
                                                          max_tokens=1000)
                result.update(answer=response.choices[0].message.content, usage=response.usage)
            except Exception as e:  # noqa: BLE001 - recorded per question
                result.update(answer=None, error=str(e))
            result["seconds"] = round(time.time() - start, 2)
            return result

        with ThreadPoolExecutor(args.workers) as pool:
            answers = list(pool.map(answer, finals))
        for row, c, a in zip(finals, checks, answers):
            _append_jsonl(out / "retrieval.jsonl.gz", {key: v for key, v in row.items() if key != "retry"})
            _append_jsonl(out / "retry.jsonl", {"question_id": row["question_id"], **c})
            _append_jsonl(out / "answers_fixed.jsonl", a)
        print(paper_id, len(paper_rows), "questions,", sum(bool(c["query"]) for c in checks), "retried", flush=True)


def main():
    """Parse arguments and run."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True, help="run whose retrieval is the first attempt")
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="eval/data")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="only the first N papers (smoke test)")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
