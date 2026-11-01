"""Evaluate the app's question-answering pipeline on QASPER (questions about NLP papers).

The pipeline under test is the app's own code: the PDF is split by `Utils.split_pdf`,
embedded into `LocalVectorStore`, the top 4 chunks are retrieved, and the answer comes
from `Utils.get_groq_response`. The model runs in Ollama instead of Groq, through a small
client with the same `chat.completions.create` interface.

Stages (each resumes where it stopped):

    prepare    download QASPER and the arXiv PDFs of the chosen split
    retrieve   split, embed and retrieve the top 10 chunks for every question
    generate   answer every question under each condition
    judge      grade every answer against the annotators' answers with a second model
    summarize  write summary.json and summary.md

Conditions:

    shipped   what the app sent before the fix: the system prompt only, without the
              question or the document (the first question of every conversation)
    fixed     the fixed app: system prompt, the 4 retrieved chunks and the question
    oracle    the same prompt with the annotators' evidence paragraphs instead of the
              retrieved chunks, an upper bound for the retrieval step
"""

import argparse
import gzip
import hashlib
import importlib.util
import json
import math
import re
import sys
import tarfile
import time
import unicodedata
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.utils import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL, RERANKER_MODEL, Utils

QASPER_URLS = {
    "dev": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-train-dev-v0.3.tgz",
    "test": "https://qasper-dataset.s3.us-west-2.amazonaws.com/qasper-test-and-evaluator-v0.3.tgz",
}
QASPER_FILES = {"dev": "qasper-dev-v0.3.json", "test": "qasper-test-v0.3.json"}
EVALUATOR_URL = QASPER_URLS["test"]  # the official evaluator ships with the test tarball
ARXIV_PDF = "https://export.arxiv.org/pdf/{}"
USER_AGENT = "document-question-and-answer-eval/0.2 (QASPER benchmark)"

K_APP = 4  # chunks the app sends to the model (default; see --k)
K_MAX = 10  # chunks retrieved for the recall curve (at least --k)
CONDITIONS = ("shipped", "fixed", "oracle")

# The system prompt the app shipped with before this fix, for the "shipped" condition
SHIPPED_SYSTEM_PROMPT = """You are a document analysis assistant with conversation memory. Your capabilities:

            1. DOCUMENT GROUNDING: Always base answers on the provided document context
            2. CONVERSATION AWARENESS: Remember and reference previous exchanges when relevant
            3. REFERENCE RESOLUTION: When users say "that", "it", "the topic", understand what they're referring to
            4. CLARITY: If a reference is ambiguous, ask for clarification
            5. ACCURACY: Never make up information not in the document or conversation

            You maintain context across the conversation while staying grounded in the document."""

JUDGE_PROMPT = """You are grading an answer to a question about a research paper.

<question>
{question}
</question>

Reference answers, written by annotators who read the paper (they may disagree with each other):
<references>
{references}
</references>

The candidate answer to grade:
<candidate>
{candidate}
</candidate>

Grade the candidate against the references:
- "correct": it gives the substance of at least one reference answer. Extra detail is fine unless it contradicts that reference.
- "partially_correct": it gives part of a reference answer's substance but misses or gets wrong another part.
- "incorrect": it contradicts every reference, gives a different answer, says the paper does not give the answer, asks for clarification instead of answering, or is empty.
{unanswerable_rule}
Judge only the text inside <candidate>. Reply with JSON: {{"verdict": "correct" | "partially_correct" | "incorrect", "explanation": "<one sentence>"}}"""

# Added to the judge prompt only when at least one annotator marked the question unanswerable
UNANSWERABLE_RULE = """
Reference {numbers} is "UNANSWERABLE": that annotator found no answer in the paper. A candidate that says the paper does not give this information agrees with that reference and is "correct".
"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
        "explanation": {"type": "string"},
    },
    "required": ["verdict", "explanation"],
}


class OllamaClient:
    """A minimal client with the Groq/OpenAI `chat.completions.create` interface, backed by Ollama.

    The native /api/chat endpoint is used so the context window can be set: Groq serves
    these models with 131k tokens, while Ollama's default would silently cut long prompts.
    """

    def __init__(self, base_url, num_ctx=16384, seed=0, timeout=900):
        """Create the client.

        Args:
            base_url (str): Ollama server, e.g. http://127.0.0.1:11500.
            num_ctx (int): Context window in tokens.
            seed (int): Sampling seed, for repeatable runs.
            timeout (int): Seconds to wait for one reply.
        """
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.seed = seed
        self.timeout = timeout
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.last_usage = None

    def call(self, payload):
        """POST a payload to /api/chat and return the decoded reply.

        Args:
            payload (dict): The request body.

        Returns:
            dict: Ollama's JSON reply.
        """
        request = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def create(self, messages, model, temperature=0.0, max_tokens=1000, **extra):
        """Send a chat request in the chat-completions shape.

        Args:
            messages (list): Chat messages.
            model (str): Ollama model name.
            temperature (float): Sampling temperature.
            max_tokens (int): Maximum tokens to generate.
            **extra: Extra top-level fields for Ollama, e.g. format or think.

        Returns:
            SimpleNamespace: An object with choices[0].message.content and usage.
        """
        reply = self.call(
            {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx,
                    "seed": self.seed,
                },
                **extra,
            }
        )
        usage = {
            "prompt_tokens": reply.get("prompt_eval_count"),
            "completion_tokens": reply.get("eval_count"),
            "done_reason": reply.get("done_reason"),
        }
        message = SimpleNamespace(content=reply["message"]["content"], thinking=reply["message"].get("thinking"))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


# ---------------------------------------------------------------- data


def _download(url, path):
    """Download a URL to a file.

    Args:
        url (str): Source URL.
        path (Path): Destination file.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=120) as response:
        path.write_bytes(response.read())


def _sha256(path):
    """Hash a file.

    Args:
        path (Path): The file.

    Returns:
        str: Hex SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_split(data_dir, split, limit=None):
    """Load one QASPER split.

    Args:
        data_dir (Path): Folder holding the extracted QASPER files.
        split (str): "dev" or "test".
        limit (int): Keep only the first this many papers (for smoke tests).

    Returns:
        dict: Papers keyed by arXiv id, in file order.
    """
    papers = json.loads((data_dir / QASPER_FILES[split]).read_text())
    return dict(list(papers.items())[:limit]) if limit else papers


def _load_evaluator(data_dir):
    """Import the official QASPER evaluator from the data folder.

    Args:
        data_dir (Path): Folder holding qasper_evaluator.py.

    Returns:
        module: The evaluator module.
    """
    spec = importlib.util.spec_from_file_location("qasper_evaluator", data_dir / "qasper_evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(args):
    """Download QASPER, the official evaluator and the split's arXiv PDFs.

    PDFs are fetched one every 3 seconds, as arXiv asks of automated clients. Each
    paper's download status, size and hash go to papers.jsonl in the output folder.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    data_dir, out = Path(args.data), Path(args.out)
    (data_dir / "pdfs").mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    for url in {QASPER_URLS[args.split], EVALUATOR_URL}:
        tgz = data_dir / url.rsplit("/", 1)[1]
        if not tgz.exists():
            _download(url, tgz)
        with tarfile.open(tgz) as archive:
            archive.extractall(data_dir, filter="data")

    papers = _load_split(data_dir, args.split, args.limit)
    done = {row["paper_id"] for row in _read_jsonl(out / "papers.jsonl")}
    for paper_id, paper in papers.items():
        if paper_id in done:
            continue
        pdf = data_dir / "pdfs" / f"{paper_id}.pdf"
        row = {"paper_id": paper_id, "title": paper["title"], "n_questions": len(paper["qas"])}
        try:
            if not pdf.exists():
                _download(ARXIV_PDF.format(paper_id), pdf)
                time.sleep(3)
            if pdf.read_bytes()[:5] != b"%PDF-":
                raise ValueError("not a PDF")
            row.update(status="ok", bytes=pdf.stat().st_size, sha256=_sha256(pdf))
        except Exception as e:  # noqa: BLE001 - recorded per paper
            pdf.unlink(missing_ok=True)
            row.update(status="failed", error=str(e))
        _append_jsonl(out / "papers.jsonl", row)
        print(paper_id, row["status"], flush=True)


# ---------------------------------------------------------------- evidence matching


def _tokens(text):
    """Normalise text to lowercase word tokens, undoing PDF line-break hyphenation.

    Args:
        text (str): Raw text.

    Returns:
        list: Alphanumeric tokens.
    """
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)
    return re.findall(r"[a-z0-9]+", text.lower())


def _ngrams(tokens, n=5):
    """Word n-grams of a token list (the whole list if it is shorter than n).

    Args:
        tokens (list): Tokens.
        n (int): n-gram length.

    Returns:
        set: Tuples of n tokens.
    """
    if len(tokens) < n:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def matches(chunk_grams, paragraph_grams):
    """Whether a chunk holds (part of) an evidence paragraph.

    A match needs at least half of the paragraph's word 5-grams in the chunk, or, for a
    paragraph longer than a chunk, at least half of the chunk's 5-grams in the paragraph.

    Args:
        chunk_grams (set): 5-grams of the chunk.
        paragraph_grams (set): 5-grams of the evidence paragraph.

    Returns:
        bool: True if they overlap enough.
    """
    if not chunk_grams or not paragraph_grams:
        return False
    shared = len(chunk_grams & paragraph_grams)
    return shared >= 0.5 * len(paragraph_grams) or shared >= 0.5 * len(chunk_grams)


def references(question):
    """The annotators' answers in the official evaluator's form.

    Args:
        question (dict): A QASPER question with its answers.

    Returns:
        list: Dicts with answer, type and text-only evidence paragraphs.
    """
    refs = []
    for annotation in question["answers"]:
        a = annotation["answer"]
        if a["unanswerable"]:
            refs.append({"answer": "Unanswerable", "type": "none", "evidence": []})
            continue
        if a["extractive_spans"]:
            answer, kind = ", ".join(a["extractive_spans"]), "extractive"
        elif a["free_form_answer"]:
            answer, kind = a["free_form_answer"], "abstractive"
        else:
            answer, kind = ("Yes" if a["yes_no"] else "No"), "boolean"
        evidence = [e for e in a["evidence"] if "FLOAT SELECTED" not in e]
        refs.append({"answer": answer, "type": kind, "evidence": evidence})
    return refs


# ---------------------------------------------------------------- retrieval


def retrieve(args):
    """Split, embed and search every paper with the app's code; record what was retrieved.

    For each question: the top K_MAX chunks (text, page), whether each holds a gold
    evidence paragraph, and whether any chunk of the whole PDF does ("findable").

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    from src.database.vectorstore import LocalVectorStore

    data_dir, out = Path(args.data), Path(args.out)
    config = _config(args, out, write=True)
    papers = _load_split(data_dir, args.split, args.limit)
    status = {row["paper_id"]: row for row in _read_jsonl(out / "papers.jsonl")}
    done = {row["question_id"] for row in _read_jsonl(out / "retrieval.jsonl.gz")}
    model = Utils.load_embedding_model()
    reranker = Utils.load_reranker() if config["rerank"] else None

    for paper_id, paper in papers.items():
        todo = [q for q in paper["qas"] if q["question_id"] not in done]
        if not todo:
            continue
        chunks, error = [], None
        if status.get(paper_id, {}).get("status") == "ok":
            try:
                chunks = Utils.split_pdf(
                    data_dir / "pdfs" / f"{paper_id}.pdf", config["chunk_size"], config["chunk_overlap"]
                )
            except Exception as e:  # noqa: BLE001 - recorded per question
                error = f"split failed: {e}"
        else:
            error = "no PDF"
        if not chunks and not error:
            error = "no text extracted"
        store = None
        if chunks:
            store = LocalVectorStore(model, reranker=reranker)
            store.add_documents(chunks)
            chunk_grams = [_ngrams(_tokens(c.page_content)) for c in chunks]

        rows = []
        for q in todo:
            refs = references(q)
            evidence = sorted({e for r in refs for e in r["evidence"]})
            row = {
                "paper_id": paper_id,
                "question_id": q["question_id"],
                "question": q["question"],
                "n_chunks": len(chunks),
                "error": error,
                "n_evidence": len(evidence),
                "retrieved": [],
            }
            if store is not None:
                para_grams = [_ngrams(_tokens(e)) for e in evidence]
                found = [i for i, cg in enumerate(chunk_grams) if any(matches(cg, pg) for pg in para_grams)]
                row["findable"] = bool(found) if evidence else None
                row["evidence_chunks"] = found
                found_set = set(found)
                index = {id(c): i for i, c in enumerate(chunks)}
                for doc in store.similarity_search(q["question"], k=max(K_MAX, config["k"])):
                    i = index[id(doc)]
                    row["retrieved"].append(
                        {"chunk": i, "page": doc.metadata.get("page"), "evidence": i in found_set,
                         "text": doc.page_content}
                    )
            rows.append(row)
        for row in rows:
            _append_jsonl(out / "retrieval.jsonl.gz", row)
        print(paper_id, len(chunks), "chunks", len(rows), "questions", error or "", flush=True)


# ---------------------------------------------------------------- generation


def _messages(condition, row, question, refs, k=K_APP):
    """The chat messages for one question under one condition.

    Args:
        condition (str): "shipped", "fixed" or "oracle".
        row (dict): The question's retrieval record.
        question (str): The question.
        refs (list): The annotators' answers (for the oracle evidence).
        k (int): Retrieved chunks sent to the model.

    Returns:
        list or None: Messages, or None if the condition does not apply.
    """
    from langchain_core.documents import Document

    if condition == "shipped":
        return [{"role": "system", "content": SHIPPED_SYSTEM_PROMPT}]
    if condition == "fixed":
        if row["error"]:
            return None
        docs = [Document(page_content=r["text"], metadata={"page": r["page"]}) for r in row["retrieved"][:k]]
        return Utils.build_messages(Utils.format_context(docs), question, [])
    evidence = sorted({e for r in refs for e in r["evidence"]})
    if not evidence:
        return None
    docs = [Document(page_content=e) for e in evidence]
    return Utils.build_messages(Utils.format_context(docs), question, [])


def generate(args):
    """Answer every question under each condition with the app's generation settings.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    data_dir, out = Path(args.data), Path(args.out)
    questions = {q["question_id"]: q for p in _load_split(data_dir, args.split, args.limit).values() for q in p["qas"]}
    rows = list(_read_jsonl(out / "retrieval.jsonl.gz"))
    client = OllamaClient(args.ollama)
    k = _config(args, out)["k"]

    for condition in args.conditions:
        path = out / f"answers_{condition}.jsonl"
        done = {r["question_id"] for r in _read_jsonl(path)}

        def answer(row, condition=condition):
            q = questions[row["question_id"]]
            messages = _messages(condition, row, q["question"], references(q), k)
            result = {"question_id": row["question_id"], "condition": condition, "model": args.model}
            if messages is None:
                result.update(answer=None, skipped=True)
                return result
            start = time.time()
            try:
                # The same call the app makes (temperature 0.1, max_tokens 1000)
                response = client.chat.completions.create(
                    messages=messages, model=args.model, temperature=0.1, max_tokens=1000
                )
                result.update(answer=response.choices[0].message.content, usage=response.usage)
            except Exception as e:  # noqa: BLE001 - recorded per question
                result.update(answer=None, error=str(e))
            result["seconds"] = round(time.time() - start, 2)
            return result

        todo = [r for r in rows if r["question_id"] not in done]
        with ThreadPoolExecutor(args.workers) as pool:
            for i, result in enumerate(pool.map(answer, todo), 1):
                _append_jsonl(path, result)
                if i % 50 == 0:
                    print(condition, i, "/", len(todo), flush=True)
        print(condition, "done", flush=True)


# ---------------------------------------------------------------- judging


def judge(args):
    """Grade every answer against the annotators' answers with a judge model.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    data_dir, out = Path(args.data), Path(args.out)
    questions = {q["question_id"]: q for p in _load_split(data_dir, args.split, args.limit).values() for q in p["qas"]}
    client = OllamaClient(args.ollama, num_ctx=8192)

    for condition in args.conditions:
        path = out / f"judgments_{condition}.jsonl"
        done = {r["question_id"] for r in _read_jsonl(path)}
        todo = [r for r in _read_jsonl(out / f"answers_{condition}.jsonl") if r["question_id"] not in done]

        def grade(answer, condition=condition):
            q = questions[answer["question_id"]]
            result = {"question_id": answer["question_id"], "condition": condition, "judge": args.judge}
            if answer.get("answer") is None:
                result.update(verdict=None, explanation="no answer")
                return result
            if not answer["answer"].strip():
                # Graded by code: the judge has been seen inventing content for empty answers
                result.update(verdict="incorrect", explanation="empty answer", by="rule")
                return result
            refs = references(q)
            listed = "\n".join(
                f"{i}. {'UNANSWERABLE' if r['type'] == 'none' else r['answer']}" for i, r in enumerate(refs, 1)
            )
            unanswerable = [str(i) for i, r in enumerate(refs, 1) if r["type"] == "none"]
            rule = UNANSWERABLE_RULE.format(numbers=", ".join(unanswerable)) if unanswerable else ""
            prompt = JUDGE_PROMPT.format(
                question=q["question"], references=listed, candidate=answer["answer"], unanswerable_rule=rule
            )
            try:
                response = client.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=args.judge,
                    temperature=0.0,
                    max_tokens=6000,
                    format=JUDGE_SCHEMA,
                    think=True,
                )
                result.update(json.loads(response.choices[0].message.content))
                result.update(by="judge", thinking=response.choices[0].message.thinking, usage=response.usage)
            except Exception as e:  # noqa: BLE001 - recorded per question
                result.update(verdict=None, error=str(e))
            return result

        with ThreadPoolExecutor(args.workers) as pool:
            for i, result in enumerate(pool.map(grade, todo), 1):
                _append_jsonl(path, result)
                if i % 50 == 0:
                    print("judge", condition, i, "/", len(todo), flush=True)
        print("judge", condition, "done", flush=True)


# ---------------------------------------------------------------- summary


def wilson(k, n, z=1.96):
    """Wilson score interval for a proportion.

    Args:
        k (int): Successes.
        n (int): Trials.
        z (float): Normal quantile (1.96 for 95%).

    Returns:
        tuple: (rate, low, high) as fractions, or (None, None, None) when n is 0.
    """
    if n == 0:
        return (None, None, None)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def _rate(k, n):
    """A rate with its Wilson interval, ready for JSON.

    Args:
        k (int): Successes.
        n (int): Trials.

    Returns:
        dict: k, n, rate and 95% interval in percent.
    """
    p, lo, hi = wilson(k, n)
    pct = (lambda x: None if x is None else round(100 * x, 1))
    return {"k": k, "n": n, "pct": pct(p), "ci95": [pct(lo), pct(hi)]}


def summarize(args):
    """Compute retrieval recall, judge verdicts and official Answer F1; write summary files.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    data_dir, out = Path(args.data), Path(args.out)
    papers = _load_split(data_dir, args.split, args.limit)
    questions = {q["question_id"]: q for p in papers.values() for q in p["qas"]}
    evaluator = _load_evaluator(data_dir)
    gold = evaluator.get_answers_and_evidence(papers, True)
    retrieval = {r["question_id"]: r for r in _read_jsonl(out / "retrieval.jsonl.gz")}
    paper_rows = list(_read_jsonl(out / "papers.jsonl"))
    config = _config(args, out)
    k_app = config["k"]

    summary = {
        "split": args.split,
        "papers": len(papers),
        "papers_with_pdf": sum(r["status"] == "ok" for r in paper_rows),
        "questions": len(questions),
        "questions_with_retrieval": sum(not r["error"] for r in retrieval.values()),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
        "k_app": k_app,
        "rerank": config["rerank"],
        "reranker_model": RERANKER_MODEL if config["rerank"] else None,
    }

    # Retrieval: questions with text evidence whose PDF was processed
    with_evidence = [r for r in retrieval.values() if not r["error"] and r["n_evidence"]]
    findable = [r for r in with_evidence if r["findable"]]
    summary["retrieval"] = {
        "questions_with_text_evidence": len(with_evidence),
        "evidence_found_in_pdf_chunks": _rate(len(findable), len(with_evidence)),
        "hit_at_k": {
            k: _rate(sum(any(c["evidence"] for c in r["retrieved"][:k]) for r in findable), len(findable))
            for k in range(1, K_MAX + 1)
        },
    }

    by_condition = {}
    for condition in CONDITIONS:
        answers = {r["question_id"]: r for r in _read_jsonl(out / f"answers_{condition}.jsonl")}
        verdicts = {r["question_id"]: r for r in _read_jsonl(out / f"judgments_{condition}.jsonl")}
        if not answers:
            continue
        scored = [qid for qid, a in answers.items() if not a.get("skipped")]
        counts = {v: 0 for v in ("correct", "partially_correct", "incorrect", None)}
        for qid in scored:
            counts[verdicts.get(qid, {}).get("verdict")] += 1
        n = len(scored)
        predicted = {qid: {"answer": answers[qid].get("answer") or "", "evidence": []} for qid in scored}
        official = evaluator.evaluate({qid: gold[qid] for qid in scored}, predicted)
        entry = {
            "questions": n,
            "skipped": len(answers) - n,
            "generation_errors": sum(1 for qid in scored if answers[qid].get("answer") is None),
            "correct": _rate(counts["correct"], n),
            "correct_or_partial": _rate(counts["correct"] + counts["partially_correct"], n),
            "incorrect": _rate(counts["incorrect"], n),
            "not_judged": counts[None],
            "answer_f1_official": round(100 * official["Answer F1"], 1),
            "answer_f1_by_type": {t: round(100 * v, 1) for t, v in official["Answer F1 by type"].items()},
        }
        if condition == "fixed":
            hit = [qid for qid in scored if qid in retrieval and retrieval[qid]["n_evidence"]
                   and any(c["evidence"] for c in retrieval[qid]["retrieved"][:k_app])]
            miss = [qid for qid in scored if qid in retrieval and retrieval[qid]["n_evidence"] and qid not in hit]
            entry["correct_when_evidence_retrieved"] = _rate(
                sum(verdicts.get(q, {}).get("verdict") == "correct" for q in hit), len(hit))
            entry["correct_when_evidence_missed"] = _rate(
                sum(verdicts.get(q, {}).get("verdict") == "correct" for q in miss), len(miss))
        # By the type of the first annotator's answer
        by_type = {}
        for qid in scored:
            kind = references(questions[qid])[0]["type"]
            by_type.setdefault(kind, []).append(verdicts.get(qid, {}).get("verdict") == "correct")
        entry["correct_by_first_annotator_type"] = {t: _rate(sum(v), len(v)) for t, v in sorted(by_type.items())}
        by_condition[condition] = entry
    summary["conditions"] = by_condition

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(_markdown(summary))
    print(_markdown(summary))


def _markdown(summary):
    """Render the summary as Markdown tables.

    Args:
        summary (dict): The output of summarize().

    Returns:
        str: Markdown text.
    """
    fmt = (lambda r: "n/a" if r["pct"] is None else f"{r['pct']}% [{r['ci95'][0]}, {r['ci95'][1]}] ({r['k']}/{r['n']})")
    lines = [
        (f"# QASPER {summary['split']}: {summary['questions']} questions, {summary['papers']} papers "
        f"({summary['papers_with_pdf']} PDFs downloaded)"),
        "",
        "## Retrieval (questions with text evidence)",
        "",
        f"Evidence present in the PDF's chunks: {fmt(summary['retrieval']['evidence_found_in_pdf_chunks'])}",
        "",
        "| k | Evidence in top k (of findable) |",
        "|---|---|",
    ]
    lines += [f"| {k} | {fmt(r)} |" for k, r in summary["retrieval"]["hit_at_k"].items()]
    lines += ["", "## Answers (judged)", "",
              "| Condition | Questions | Correct | Correct or partial | Answer F1 (official) |",
              "|---|---|---|---|---|"]
    for name, c in summary["conditions"].items():
        lines.append(f"| {name} | {c['questions']} | {fmt(c['correct'])} | {fmt(c['correct_or_partial'])} | "
                     f"{c['answer_f1_official']} |")
    fixed = summary["conditions"].get("fixed")
    if fixed:
        lines += ["", (f"Fixed app, correct when the top {summary['k_app']} chunks held evidence: "
                  f"{fmt(fixed['correct_when_evidence_retrieved'])}; when they did not: "
                  f"{fmt(fixed['correct_when_evidence_missed'])}")]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- io


def _config(args, out, write=False):
    """The retrieval settings of a results folder.

    The retrieve stage writes them to config.json; later stages read them from there, so a
    folder's answers and summary always match how its chunks were retrieved. A folder made
    before config.json existed used the defaults.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        out (Path): The results folder.
        write (bool): Record the settings from args (retrieve stage).

    Returns:
        dict: chunk_size, chunk_overlap, k and rerank.

    Raises:
        ValueError: If args ask for settings other than those the folder was made with.
    """
    path = out / "config.json"
    wanted = {"chunk_size": args.chunk_size, "chunk_overlap": args.chunk_overlap, "k": args.k, "rerank": args.rerank}
    if path.exists():
        saved = json.loads(path.read_text())
        if write and saved != wanted:
            raise ValueError(f"{out} was made with {saved}, not {wanted}")
        return saved
    if write:
        out.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(wanted, indent=2))
        return wanted
    return {"chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP, "k": K_APP, "rerank": False}


def _read_jsonl(path):
    """Read a JSON-lines file (gzip if it ends in .gz); a missing file reads as empty.

    Args:
        path (Path): The file.

    Yields:
        dict: One record per line.
    """
    if not path.exists():
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _append_jsonl(path, row):
    """Append one record to a JSON-lines file (gzip if it ends in .gz).

    Args:
        path (Path): The file.
        row (dict): The record.
    """
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "at", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    """Parse arguments and run one stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "retrieve", "generate", "judge", "summarize"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--data", default="eval/data", help="QASPER files and PDFs (not committed)")
    parser.add_argument("--out", required=True, help="results folder")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:8b", help="answer model (the app's default is Llama 3.1 8B)")
    parser.add_argument("--judge", default="qwen3:32b")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS), choices=CONDITIONS)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, help="only the first N papers (smoke test)")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE, help="retrieve stage: characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, default=CHUNK_OVERLAP, help="retrieve stage")
    parser.add_argument("--k", type=int, default=K_APP, help="retrieve stage: chunks sent to the model")
    parser.add_argument("--rerank", action="store_true", help="retrieve stage: rerank with the cross-encoder")
    args = parser.parse_args()
    {"prepare": prepare, "retrieve": retrieve, "generate": generate, "judge": judge, "summarize": summarize}[
        args.stage
    ](args)


if __name__ == "__main__":
    main()
