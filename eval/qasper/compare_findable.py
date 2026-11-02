"""Check the comparison run's evidence matching: is each paper's evidence findable in its chunks?

Tells whether "evidence from both papers in the context" is rare because the matching fails or
because the comparison question does not retrieve the evidence. Uses the same splitter and
matching rule as compare_eval.py.

Usage, from the repo root on a machine with the QASPER data and PDFs:
    python eval/qasper/compare_findable.py --data eval/data
Writes eval/results/2026-11-01-compare/findable.json.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_eval import _load_split, references
from run_eval import _ngrams, _read_jsonl, _tokens, matches

from src.utils.utils import Utils

OUT = ROOT / "eval" / "results" / "2026-11-01-compare"


def findable_chunks(pdf, evidence):
    """Count the chunks of a paper that hold (part of) any evidence paragraph.

    Args:
        pdf (Path): The paper's PDF.
        evidence (list): Evidence paragraphs from the references.

    Returns:
        int: Number of matching chunks.
    """
    grams = [_ngrams(_tokens(e)) for e in evidence]
    chunks = Utils.split_pdf(pdf)
    return sum(any(matches(_ngrams(_tokens(c.page_content)), g) for g in grams) for c in chunks)


def main():
    """Count findable evidence per paper and per pair, and evidence in context per rule."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="eval/data")
    args = parser.parse_args()
    data = Path(args.data)

    papers = _load_split(data, "dev")
    questions = {q["question_id"]: q for p in papers.values() for q in p["qas"]}
    pairs = json.loads((OUT / "pairs.json").read_text(encoding="utf-8"))
    found = {}
    for pair in pairs:
        for p in pair["papers"]:
            key = (p["paper_id"], p["question_id"])
            if key not in found:
                evidence = [e for r in references(questions[p["question_id"]]) for e in r["evidence"]]
                found[key] = findable_chunks(data / "pdfs" / f"{p['paper_id']}.pdf", evidence)

    out = {
        "papers": len(found),
        "papers_findable": sum(n > 0 for n in found.values()),
        "pairs": len(pairs),
        "pairs_findable_in_both": sum(all(found[(p["paper_id"], p["question_id"])] for p in pair["papers"])
                                      for pair in pairs),
    }
    records = list(_read_jsonl(OUT / "answers.jsonl"))
    for rule in ("app", "balanced"):
        slots = [hit for r in records if r["rule"] == rule for hit in r["evidence"].values()]
        out[f"{rule}_paper_slots_with_evidence"] = {"k": sum(slots), "n": len(slots)}
    (OUT / "findable.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
