"""Check whether the pages an answer cites hold what the answer says.

The app labels each retrieved chunk with its page ("[p. 4]") and the model cites those labels.
This script takes the saved answers of one run and:

  claims     splits every answer into cited statements, records whether each cited page was in
             the model's context, and draws one statement per answer (seed 0) plus a blind sample
             of 50 of them for labelling by hand
  judge      asks the judge model whether the passages from the cited pages support each drawn
             statement
  summarize  rates of support, overall and by the answer's verdict, and, once hand labels exist,
             agreement between the hand labels and the judge

Usage:
  python eval/qasper/citation_check.py claims --run eval/results/2026-11-01-qasper-dev --out eval/results/<folder>
  python eval/qasper/citation_check.py judge --out eval/results/<folder> --ollama http://127.0.0.1:11500
  python eval/qasper/citation_check.py summarize --out eval/results/<folder> [--labels labels_<name>.csv]
"""

import argparse
import collections
import csv
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import K_APP, OllamaClient, _append_jsonl, _config, _rate, _read_jsonl

CITATION = re.compile(r"\[(?:Used pages:\s*)?pp?\.\s*([^\]]+)\]")
SAMPLE_SIZE = 50
VERDICTS = ["supported", "partially_supported", "not_supported"]

SUPPORT_PROMPT = """You are checking whether a statement is supported by the passages it cites.

The statement comes from an answer to a question about a research paper. The passages are the
text of the paper's pages that the statement cites, as the answering model saw them.

<question>
{question}
</question>

<statement>
{claim}
</statement>

<cited_passages>
{passages}
</cited_passages>

Judge only from the passages, not from what you know about the paper or the topic.
- "supported": everything the statement says about the paper is stated in the passages or follows
  directly from them.
- "partially_supported": part of what it says is in the passages, but part is missing from them
  or goes beyond them.
- "not_supported": the passages do not state it, or contradict it.
Ignore wording, citation formatting, and phrases such as "according to the document". A statement
that the paper does not give some information is supported only if the passages are about that
topic and do not give it.

Reply with JSON: {{"verdict": "supported" | "partially_supported" | "not_supported", "explanation": "<one sentence>"}}"""

SUPPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": VERDICTS},
        "explanation": {"type": "string"},
    },
    "required": ["verdict", "explanation"],
}


def cited_pages(text):
    """Page numbers in the citations of a piece of text.

    Args:
        text (str): Text with citations such as "[p. 4]", "[p. 4-5]" or "[p. 4, p. 6]".

    Returns:
        list: Sorted page numbers (1-based, as the app labels them); unparseable citations
            such as "[p. ?]" give none.
    """
    pages = set()
    for inside in CITATION.findall(text):
        for a, b in re.findall(r"(\d+)(?:\s*[-–]\s*(\d+))?", inside.split(" of ")[0]):
            lo, hi = int(a), int(b or a)
            if hi - lo <= 5:
                pages.update(range(lo, hi + 1))
    return sorted(pages)


def statements(answer):
    """Split an answer into statements that carry citations.

    Sentences are split at line breaks and sentence ends. A cited sentence that ends with a
    colon introduces a list, so the list items that follow are part of the statement.

    Args:
        answer (str): The model's answer.

    Returns:
        list: Statements (str) with at least one citation.
    """
    lines = [line.strip() for line in answer.splitlines()]
    out, i = [], 0
    while i < len(lines):
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z\[])", lines[i]):
            if not CITATION.search(sentence):
                continue
            if sentence.endswith(":"):
                j = i + 1
                while j < len(lines) and (re.match(r"^([-*•]|\d+[.)])\s", lines[j]) or not lines[j]):
                    if lines[j]:
                        sentence += "\n" + lines[j]
                    j += 1
            out.append(sentence)
        i += 1
    return out


def claims(args):
    """Extract cited statements from every answer and draw the samples.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = _config(argparse.Namespace(chunk_size=None, chunk_overlap=None, k=None, rerank=None), run)["k"] or K_APP
    retrieval = {r["question_id"]: r for r in _read_jsonl(run / "retrieval.jsonl.gz")}
    verdicts = {r["question_id"]: r["verdict"] for r in _read_jsonl(run / f"judgments_{args.condition}.jsonl")}
    rows, totals = [], collections.Counter()
    for a in _read_jsonl(run / f"answers_{args.condition}.jsonl"):
        r = retrieval.get(a["question_id"])
        if not a.get("answer") or not r or not r.get("retrieved"):
            continue
        context = r["retrieved"][:k]
        in_context = {c["page"] + 1 for c in context if isinstance(c.get("page"), int)}
        totals["answers"] += 1
        found = statements(a["answer"])
        totals["answers_citing"] += bool(found)
        for n, claim in enumerate(found):
            pages = cited_pages(claim)
            totals["statements"] += 1
            totals["citations"] += len(pages)
            totals["citations_in_context"] += sum(p in in_context for p in pages)
            rows.append({
                "question_id": a["question_id"], "question": r["question"], "n": n, "claim": claim,
                "pages": pages, "pages_in_context": [p for p in pages if p in in_context],
                "passages": [{"page": c["page"] + 1, "text": c["text"]} for c in context
                             if isinstance(c.get("page"), int) and c["page"] + 1 in pages],
                "answer_verdict": verdicts.get(a["question_id"]),
            })
    for path in (out / "statements.jsonl.gz",):
        path.unlink(missing_ok=True)
        for row in rows:
            _append_jsonl(path, row)

    # One checkable statement per answer: every cited page was in the context
    rng = random.Random(0)
    by_answer = collections.defaultdict(list)
    for row in rows:
        if row["pages"] and len(row["pages_in_context"]) == len(row["pages"]):
            by_answer[row["question_id"]].append(row)
    drawn = [rng.choice(by_answer[q]) for q in sorted(by_answer)]
    (out / "drawn.json").write_text(json.dumps(drawn, indent=1, ensure_ascii=False), encoding="utf-8")
    sample = random.Random(0).sample(sorted(drawn, key=lambda x: x["question_id"]), SAMPLE_SIZE)
    with open(out / "to_label_by_hand.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "question_id", "question", "statement", "cited_passages",
                    "your_label (supported / partially_supported / not_supported)"])
        for i, s in enumerate(sample, 1):
            passages = "\n\n".join(f"[p. {p['page']}] {p['text']}" for p in s["passages"])
            w.writerow([i, s["question_id"], s["question"], s["claim"], passages, ""])
    totals["answers_with_checkable_statement"] = len(drawn)
    (out / "counts.json").write_text(json.dumps(totals, indent=2))
    print(json.dumps(totals, indent=2))


def judge(args):
    """Ask the judge whether the cited passages support each drawn statement.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    out = Path(args.out)
    drawn = json.loads((out / "drawn.json").read_text(encoding="utf-8"))
    path = out / "support_judgments.jsonl"
    done = {r["question_id"] for r in _read_jsonl(path) if r.get("verdict")}
    client = OllamaClient(args.ollama, num_ctx=12288)

    def grade(s):
        """Judge one statement.

        Args:
            s (dict): A drawn statement with its passages.

        Returns:
            dict: The verdict record.
        """
        passages = "\n\n".join(f"[p. {p['page']}] {p['text']}" for p in s["passages"])
        prompt = SUPPORT_PROMPT.format(question=s["question"], claim=s["claim"], passages=passages)
        result = {"question_id": s["question_id"], "n": s["n"], "judge": args.judge}
        try:
            response = client.create(messages=[{"role": "user", "content": prompt}], model=args.judge,
                                     temperature=0.0, max_tokens=8000, format=SUPPORT_SCHEMA, think=True)
            result.update(json.loads(response.choices[0].message.content))
            result.update(thinking=response.choices[0].message.thinking, usage=response.usage)
        except Exception as e:  # noqa: BLE001 - recorded per statement
            result.update(verdict=None, error=str(e))
        return result

    # The hand-labelled sample first, so the judge can be checked before the rest finishes
    with open(out / "to_label_by_hand.csv", encoding="utf-8") as f:
        sample = {r["question_id"] for r in csv.DictReader(f)}
    todo = sorted((s for s in drawn if s["question_id"] not in done), key=lambda s: s["question_id"] not in sample)
    with ThreadPoolExecutor(args.workers) as pool:
        for i, result in enumerate(pool.map(grade, todo), 1):
            _append_jsonl(path, result)
            if i % 50 == 0:
                print("support", i, "/", len(todo), flush=True)
    print("support done", flush=True)


def kappa(pairs, cats):
    """Cohen's kappa for two raters.

    Args:
        pairs (list): (rater A, rater B) labels.
        cats (list): The categories.

    Returns:
        float: kappa, or None when agreement by chance is total.
    """
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    pe = sum(ca[c] * cb[c] for c in cats) / n**2
    return None if pe == 1 else round((po - pe) / (1 - pe), 3)


def summarize(args):
    """Rates of support, and agreement with hand labels if given; writes summary.json.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    out = Path(args.out)
    counts = json.loads((out / "counts.json").read_text())
    drawn = {s["question_id"]: s for s in json.loads((out / "drawn.json").read_text(encoding="utf-8"))}
    judged = {r["question_id"]: r for r in _read_jsonl(out / "support_judgments.jsonl")}
    summary = {
        "answers": counts["answers"],
        "answers_citing": _rate(counts["answers_citing"], counts["answers"]),
        "citations_in_context": _rate(counts["citations_in_context"], counts["citations"]),
        "judged_statements": sum(1 for r in judged.values() if r.get("verdict")),
        "support": {v: _rate(sum(r.get("verdict") == v for r in judged.values()),
                             sum(1 for r in judged.values() if r.get("verdict"))) for v in VERDICTS},
        "support_by_answer_verdict": {},
    }
    for av in ["correct", "partially_correct", "incorrect"]:
        rs = [judged[q] for q, s in drawn.items() if s["answer_verdict"] == av and judged.get(q, {}).get("verdict")]
        summary["support_by_answer_verdict"][av] = {v: _rate(sum(r["verdict"] == v for r in rs), len(rs))
                                                    for v in VERDICTS}
    if args.labels:
        with open(out / args.labels, encoding="utf-8") as f:
            labels = {r["question_id"]: r for r in csv.DictReader(f)}
        pairs = [(labels[q]["label"], judged[q]["verdict"]) for q in labels]
        binary = [(a == "supported", b == "supported") for a, b in pairs]
        summary["hand_labels"] = {
            "file": args.labels,
            "n": len(pairs),
            "exact_agreement": sum(a == b for a, b in pairs),
            "kappa": kappa(pairs, VERDICTS),
            "supported_vs_not_agreement": sum(a == b for a, b in binary),
            "kappa_supported_vs_not": kappa(binary, [True, False]),
            "labeller": {v: sum(a == v for a, _ in pairs) for v in VERDICTS},
            "judge": {v: sum(b == v for _, b in pairs) for v in VERDICTS},
            "confusion_rows_labeller_cols_judge": {a: {b: sum(1 for x, y in pairs if (x, y) == (a, b)) for b in VERDICTS}
                                                   for a in VERDICTS},
            "disagreements": [{"question_id": q, "labeller": labels[q]["label"], "judge": judged[q]["verdict"],
                               "labeller_note": labels[q].get("note", ""),
                               "judge_explanation": judged[q].get("explanation", "")}
                              for q in labels if labels[q]["label"] != judged[q]["verdict"]],
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "hand_labels"}, indent=2))
    if "hand_labels" in summary:
        print(json.dumps({k: v for k, v in summary["hand_labels"].items() if k != "disagreements"}, indent=2))


def main():
    """Parse arguments and run one stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["claims", "judge", "summarize"])
    parser.add_argument("--out", required=True, help="results folder for the citation check")
    parser.add_argument("--run", help="claims stage: the evaluation run whose answers are checked")
    parser.add_argument("--condition", default="fixed", help="claims stage: which answers")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--judge", default="qwen3:32b")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--labels", help="summarize stage: hand-label CSV in --out (question_id,label[,note])")
    args = parser.parse_args()
    {"claims": claims, "judge": judge, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
