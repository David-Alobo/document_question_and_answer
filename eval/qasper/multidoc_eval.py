"""Does loading other papers hurt retrieval for questions about one of them?

For every QASPER question, the app's vector store is filled with the question's paper
plus N other papers chosen at random (seeded), and searched with the app's own rules:

    single     top 4 chunks (the rule with one file)
    multi      top 4, then the best chunk of each other file, up to 8 (the rule with
               several files and "All documents" chosen)
    selected   top 4 of the question's paper only (the rule when the user picks that
               file under "Ask about")

Recorded per question and N: whether a chunk holding gold evidence (matched as in
run_eval.py, and only from the question's own paper) is in the context, how many
context chunks come from the question's paper, and the rank of the first evidence chunk.
N = 0 is the single-file baseline.

Usage:
    python eval/qasper/multidoc_eval.py run --data eval/data --out <folder>
    python eval/qasper/multidoc_eval.py summarize --out <folder>
"""

import argparse
import csv
import gzip
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import _load_split, _ngrams, _tokens, matches, references, wilson

from src.database.vectorstore import LocalVectorStore
from src.document_processor.processor import MAX_CONTEXT, TOP_K
from src.utils.utils import Utils

DISTRACTORS = (0, 2, 4, 9)
SEED = 0


def run(args):
    """Search every question's paper mixed with N other papers; write results.csv.gz.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    import torch

    torch.set_num_threads(args.threads)
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    papers = _load_split(data_dir, args.split, args.limit)
    model = Utils.load_embedding_model()
    probe = LocalVectorStore(model)

    # Split and embed every paper once, as the app caches each file
    cache = {}
    for paper_id in papers:
        pdf = data_dir / "pdfs" / f"{paper_id}.pdf"
        if not pdf.exists():
            continue
        try:
            chunks = Utils.split_pdf(pdf)
        except Exception as e:  # noqa: BLE001 - such papers are left out, as in the baseline
            print(paper_id, "failed:", e, flush=True)
            continue
        if not chunks:
            continue
        for chunk in chunks:
            chunk.metadata["source"] = f"{paper_id}.pdf"
        cache[paper_id] = (chunks, probe.encode(chunks))
    print(len(cache), "papers embedded", flush=True)

    rng = random.Random(SEED)
    ids = sorted(cache)
    rows = []
    for paper_id in ids:
        chunks, _ = cache[paper_id]
        grams = [_ngrams(_tokens(c.page_content)) for c in chunks]
        others = [p for p in ids if p != paper_id]
        mix = rng.sample(others, max(DISTRACTORS))  # nested: the first n are the n distractors
        for n in DISTRACTORS:
            store = LocalVectorStore(model)
            for pid in [paper_id, *mix[:n]]:
                store.add_documents(*cache[pid])
            for q in papers[paper_id]["qas"]:
                evidence = [_ngrams(_tokens(e)) for e in sorted({e for r in references(q) for e in r["evidence"]})]
                gold = {id(c) for c, g in zip(chunks, grams, strict=True) if any(matches(g, pg) for pg in evidence)}
                if not gold:
                    continue
                ranked = store.similarity_search(q["question"], k=len(store.documents))
                first = next((r for r, c in enumerate(ranked, 1) if id(c) in gold), None)
                for rule, context in (
                    ("single", store.similarity_search(q["question"], k=TOP_K)),
                    ("multi", store.similarity_search(q["question"], k=TOP_K, per_source=1, max_total=MAX_CONTEXT)
                     if n else store.similarity_search(q["question"], k=TOP_K)),
                    ("selected", store.similarity_search(q["question"], k=TOP_K, sources=[f"{paper_id}.pdf"])),
                ):
                    rows.append({
                        "paper_id": paper_id, "question_id": q["question_id"], "distractors": n, "rule": rule,
                        "evidence_in_context": int(any(id(c) in gold for c in context)),
                        "context_chunks": len(context),
                        "own_paper_chunks": sum(c.metadata["source"] == f"{paper_id}.pdf" for c in context),
                        "first_evidence_rank": "" if first is None else first,
                    })
        print(paper_id, flush=True)

    with gzip.open(out / "results.csv.gz", "wt", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "distractors.json").write_text(json.dumps({"seed": SEED, "papers": len(cache)}, indent=2))


def summarize(args):
    """Rates by number of distractors and rule, with paired counts against N = 0.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    out = Path(args.out)
    with gzip.open(out / "results.csv.gz", "rt", newline="") as f:
        rows = list(csv.DictReader(f))
    by = {}
    for r in rows:
        by.setdefault((int(r["distractors"]), r["rule"]), {})[r["question_id"]] = r
    base = by[(0, "single")]
    summary, lines = {}, [
        f"# Multi-document retrieval: {len(base)} questions with findable evidence",
        "",
        ("Each question's paper is loaded with N other dev papers (seed 0). *single*: top 4 chunks; "
        "*multi*: top 4 plus the best chunk of each other file, up to 8 (the app's rule with several files). "
        "Lost/gained: questions whose evidence left/entered the context compared with the paper alone."),
        "",
        "| Other papers | Rule | Evidence in context | Context chunks (mean) | From the right paper (mean) | Lost / gained |",
        "|---|---|---|---|---|---|",
    ]
    for (n, rule), qs in sorted(by.items()):
        k = sum(int(r["evidence_in_context"]) for r in qs.values())
        p, lo, hi = wilson(k, len(qs))
        lost = sum(int(base[q]["evidence_in_context"]) and not int(r["evidence_in_context"]) for q, r in qs.items())
        gained = sum(int(r["evidence_in_context"]) and not int(base[q]["evidence_in_context"]) for q, r in qs.items())
        ctx = sum(int(r["context_chunks"]) for r in qs.values()) / len(qs)
        own = sum(int(r["own_paper_chunks"]) for r in qs.values()) / len(qs)
        summary[f"{n}/{rule}"] = {
            "evidence_in_context": [k, len(qs), round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1)],
            "mean_context_chunks": round(ctx, 2), "mean_own_paper_chunks": round(own, 2),
            "lost_vs_alone": lost, "gained_vs_alone": gained,
        }
        lines.append(f"| {n} | {rule} | {100 * p:.1f}% [{100 * lo:.1f}, {100 * hi:.1f}] ({k}/{len(qs)}) | "
                     f"{ctx:.2f} | {own:.2f} | {lost} / {gained} |")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    """Parse arguments and run a stage."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["run", "summarize"])
    parser.add_argument("--split", default="dev", choices=["dev", "test"])
    parser.add_argument("--data", default="eval/data")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    {"run": run, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
