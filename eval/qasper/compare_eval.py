"""Comparison questions over two papers: does gathering evidence from each paper help?

QASPER has no questions about two papers, but many of its questions are asked of several papers
("What datasets are used?"), each with its own annotated answer and evidence. Pairs of such papers
give comparison questions with a gold answer for each paper:

  build   group the development questions into four topics (baselines, datasets, evaluation
          metrics, English only), draw pairs of papers within each topic (seed 0) and write
          pairs.json with one comparison question per pair
  run     load both papers into one store, as the app does, and build the context two ways:
            app        the top 4 chunks overall, then the best chunk of any file not yet among
                       them, up to 8 (the app's rule with several files)
            balanced   the top 2 chunks of each paper (4 in all, no longer than the app's)
          record whether each paper's evidence is in the context, and answer with the app's
          prompt and model
  judge   grade each answer twice, once per paper, against that paper's references
  summarize  rates per rule: evidence from both papers in the context, and answers correct for
          both papers; paired comparison of the rules

Usage:
  python eval/qasper/compare_eval.py build --data ../data --out eval/results/<folder>
  python eval/qasper/compare_eval.py run --data ../data --out eval/results/<folder> --ollama <url>
  python eval/qasper/compare_eval.py judge --data ../data --out eval/results/<folder> --ollama <url>
  python eval/qasper/compare_eval.py summarize --out eval/results/<folder>
"""

import argparse
import itertools
import json
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from repeats import mcnemar
from run_eval import (
    JUDGE_PROMPT,
    JUDGE_SCHEMA,
    UNANSWERABLE_RULE,
    OllamaClient,
    _append_jsonl,
    _load_split,
    _ngrams,
    _rate,
    _read_jsonl,
    _tokens,
    matches,
    references,
)

from src.utils.utils import Utils

TOPICS = {
    "baselines": (r"^(what|which)\b[^?]*\bbaselines?\b", "Compare the two papers: what baselines does each one compare against?"),
    "datasets": (r"^(what|which)\b[^?]*\bdatasets?\b[^?]*\b(used|use|experiment|evaluate)",
                 "Compare the two papers: what datasets does each one use?"),
    "metrics": (r"^(what|which)\b[^?]*\bmetrics?\b", "Compare the two papers: what evaluation metrics does each one use?"),
    "english": (r"\bonly\b[^?]*\benglish\b", "Compare the two papers: does each one report results only on English data?"),
}
# Questions that mention the topic but ask something else (improvement over a baseline, dataset size)
EXCLUDE = r"\b(improve|improvement|better|outperform|size|how many|how big|source|strong|weak|performance|score)"
PAIRS_PER_TOPIC = 30
RULES = ("app", "balanced")
TOP_K, MAX_CONTEXT, PER_PAPER = 4, 8, 2

PAPER_RULE = """
The candidate answers a question about two papers, {file} and another. Grade only what it says
about {file}, as an answer to the question above; ignore what it says about the other paper.
"""


def build(args):
    """Draw the paper pairs and write pairs.json.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    papers = _load_split(data_dir, "dev")
    pairs = []
    for topic, (pattern, question) in TOPICS.items():
        chosen = {}
        for pid, p in sorted(papers.items()):
            if not (data_dir / "pdfs" / f"{pid}.pdf").exists():
                continue
            for q in p["qas"]:
                refs = references(q)
                text = q["question"].lower().strip()
                if re.search(pattern, text) and not re.search(EXCLUDE, text) and any(r["evidence"] for r in refs):
                    chosen.setdefault(pid, q)  # one question per paper and topic
        candidates = list(itertools.combinations(sorted(chosen), 2))
        drawn = random.Random(0).sample(candidates, min(PAIRS_PER_TOPIC, len(candidates)))
        for a, b in drawn:
            pairs.append({"pair_id": f"{topic}:{a}:{b}", "topic": topic, "question": question,
                          "papers": [{"paper_id": pid, "question_id": chosen[pid]["question_id"],
                                      "paper_question": chosen[pid]["question"]} for pid in (a, b)]})
        print(topic, len(chosen), "papers,", len(drawn), "pairs", flush=True)
    (out / "pairs.json").write_text(json.dumps(pairs, indent=1), encoding="utf-8")


def run(args):
    """Search, record evidence and answer each pair under both rules.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    from src.database.vectorstore import LocalVectorStore

    data_dir, out = Path(args.data), Path(args.out)
    papers = _load_split(data_dir, "dev")
    questions = {q["question_id"]: q for p in papers.values() for q in p["qas"]}
    pairs = json.loads((out / "pairs.json").read_text(encoding="utf-8"))
    done = {(r["pair_id"], r["rule"]) for r in _read_jsonl(out / "answers.jsonl")}
    model = Utils.load_embedding_model()
    probe = LocalVectorStore(model)
    client = OllamaClient(args.ollama, seed=args.seed)
    cache = {}

    def load(pid):
        """Split and embed one paper once, labelled with its file name as in the app.

        Args:
            pid (str): arXiv id.

        Returns:
            tuple: (chunks, embeddings).
        """
        if pid not in cache:
            chunks = Utils.split_pdf(data_dir / "pdfs" / f"{pid}.pdf")
            for c in chunks:
                c.metadata["source"] = f"{pid}.pdf"
            cache[pid] = (chunks, probe.encode(chunks))
        return cache[pid]

    jobs = []
    for pair in pairs:
        store = LocalVectorStore(model)
        gold = {}
        for p in pair["papers"]:
            chunks, emb = load(p["paper_id"])
            store.add_documents(chunks, emb)
            evidence = [_ngrams(_tokens(e)) for r in references(questions[p["question_id"]]) for e in r["evidence"]]
            gold[p["paper_id"]] = {id(c) for c in chunks if any(matches(_ngrams(_tokens(c.page_content)), g)
                                                                for g in evidence)}
        contexts = {
            "app": store.similarity_search(pair["question"], k=TOP_K, per_source=1, max_total=MAX_CONTEXT),
            "balanced": [c for p in pair["papers"] for c in
                         store.similarity_search(pair["question"], k=PER_PAPER, sources=[f"{p['paper_id']}.pdf"])],
        }
        for rule, docs in contexts.items():
            if (pair["pair_id"], rule) in done:
                continue
            record = {"pair_id": pair["pair_id"], "rule": rule, "seed": args.seed,
                      "context": [{"source": d.metadata["source"], "page": d.metadata.get("page"),
                                   "text": d.page_content} for d in docs],
                      "evidence": {pid: any(id(d) in g for d in docs) for pid, g in gold.items()},
                      "chunks_per_paper": {pid: sum(d.metadata["source"] == f"{pid}.pdf" for d in docs)
                                           for pid in gold}}
            jobs.append((record, Utils.build_messages(Utils.format_context(docs), pair["question"], [])))
        print(pair["pair_id"], flush=True)

    def answer(job):
        """Answer one pair with the app's call.

        Args:
            job (tuple): (record, messages).

        Returns:
            dict: The record with the answer added.
        """
        record, messages = job
        try:
            response = client.chat.completions.create(messages=messages, model=args.model, temperature=0.1,
                                                      max_tokens=1000)
            record.update(answer=response.choices[0].message.content)
        except Exception as e:  # noqa: BLE001 - recorded per pair
            record.update(answer=None, error=str(e))
        return record

    with ThreadPoolExecutor(args.workers) as pool:
        for record in pool.map(answer, jobs):
            _append_jsonl(out / "answers.jsonl", record)


def judge(args):
    """Grade each answer once per paper against that paper's references.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    data_dir, out = Path(args.data), Path(args.out)
    papers = _load_split(data_dir, "dev")
    questions = {q["question_id"]: q for p in papers.values() for q in p["qas"]}
    pairs = {p["pair_id"]: p for p in json.loads((out / "pairs.json").read_text(encoding="utf-8"))}
    done = {(r["pair_id"], r["rule"], r["paper_id"]) for r in _read_jsonl(out / "judgments.jsonl") if r.get("verdict")}
    client = OllamaClient(args.ollama, num_ctx=12288)
    jobs = [(a, p) for a in _read_jsonl(out / "answers.jsonl") for p in pairs[a["pair_id"]]["papers"]
            if (a["pair_id"], a["rule"], p["paper_id"]) not in done]

    def grade(job):
        """Judge what one answer says about one paper.

        Args:
            job (tuple): (answer record, paper entry of the pair).

        Returns:
            dict: The verdict record.
        """
        a, p = job
        result = {"pair_id": a["pair_id"], "rule": a["rule"], "paper_id": p["paper_id"]}
        if not a.get("answer"):
            result.update(verdict="incorrect", explanation="no answer", by="rule")
            return result
        refs = references(questions[p["question_id"]])
        listed = "\n".join(f"{i}. {'UNANSWERABLE' if r['type'] == 'none' else r['answer']}"
                           for i, r in enumerate(refs, 1))
        unanswerable = [str(i) for i, r in enumerate(refs, 1) if r["type"] == "none"]
        rule = UNANSWERABLE_RULE.format(numbers=", ".join(unanswerable)) if unanswerable else ""
        rule += PAPER_RULE.format(file=f"{p['paper_id']}.pdf")
        prompt = JUDGE_PROMPT.format(question=p["paper_question"], references=listed, candidate=a["answer"],
                                     unanswerable_rule=rule)
        try:
            response = client.create(messages=[{"role": "user", "content": prompt}], model=args.judge,
                                     temperature=0.0, max_tokens=8000, format=JUDGE_SCHEMA, think=True)
            result.update(json.loads(response.choices[0].message.content), by="judge")
        except Exception as e:  # noqa: BLE001 - recorded per verdict
            result.update(verdict=None, error=str(e))
        return result

    with ThreadPoolExecutor(args.workers) as pool:
        for result in pool.map(grade, jobs):
            _append_jsonl(out / "judgments.jsonl", result)


def summarize(args):
    """Rates per rule and the paired comparison; writes summary.json.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    out = Path(args.out)
    answers = {(a["pair_id"], a["rule"]): a for a in _read_jsonl(out / "answers.jsonl")}
    verdicts = {}
    for r in _read_jsonl(out / "judgments.jsonl"):
        if r.get("verdict"):
            verdicts[r["pair_id"], r["rule"], r["paper_id"]] = r["verdict"]
    pair_ids = sorted({p for p, _ in answers})
    per = {}
    for rule in RULES:
        both_ev, both_ok, either_ok, n_judged = {}, {}, {}, 0
        for pid in pair_ids:
            a = answers.get((pid, rule))
            if not a:
                continue
            both_ev[pid] = all(a["evidence"].values())
            v = [verdicts.get((pid, rule, paper)) for paper in a["evidence"]]
            if all(v):
                n_judged += 1
                both_ok[pid] = all(x == "correct" for x in v)
                either_ok[pid] = sum(x == "correct" for x in v)
        per[rule] = {"evidence": both_ev, "correct": both_ok, "papers_correct": either_ok}
    summary = {"pairs": len(pair_ids), "rules": {}}
    for rule in RULES:
        ev, ok, pc = per[rule]["evidence"], per[rule]["correct"], per[rule]["papers_correct"]
        summary["rules"][rule] = {
            "evidence_from_both_papers": _rate(sum(ev.values()), len(ev)),
            "correct_for_both_papers": _rate(sum(ok.values()), len(ok)),
            "papers_answered_correctly": _rate(sum(pc.values()), 2 * len(pc)),
        }
    for key in ("evidence", "correct"):
        a, b = per["balanced"][key], per["app"][key]
        qs = sorted(set(a) & set(b))
        gained = sum(a[q] and not b[q] for q in qs)
        lost = sum(b[q] and not a[q] for q in qs)
        summary[f"balanced_vs_app_{key}"] = {"n": len(qs), "gained": gained, "lost": lost, "p": mcnemar(gained, lost)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    """Parse arguments and run one stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["build", "run", "judge", "summarize"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--data", default="eval/data")
    parser.add_argument("--ollama", default="http://127.0.0.1:11434")
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--judge", default="qwen3:32b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    {"build": build, "run": run, "judge": judge, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
