"""Compare labels for the judge-validation sample with the judge's verdicts.

Usage: python compare.py labels_claude.csv [more label files...]
Writes comparison_<labels>.json next to each label file and prints a summary.
"""

import collections
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CATS = ["correct", "partially_correct", "incorrect"]


def verdicts():
    """The judge's last verdict per question for the fixed condition.

    Returns:
        dict: question id to judgement record.
    """
    path = HERE.parent / "2026-11-01-qasper-dev" / "judgments_fixed.jsonl"
    out = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        out[record["question_id"]] = record
    return out


def kappa(pairs):
    """Cohen's kappa for two raters over the three grades.

    Args:
        pairs (list): (rater A, rater B) labels.

    Returns:
        float: kappa.
    """
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    ca = collections.Counter(a for a, _ in pairs)
    cb = collections.Counter(b for _, b in pairs)
    expected = sum(ca[c] * cb[c] for c in CATS) / n**2
    return (observed - expected) / (1 - expected)


def compare(label_file):
    """Agreement between one label file and the judge.

    Args:
        label_file (str): CSV with question_id and label columns.

    Returns:
        dict: The comparison, also written to comparison_<name>.json.
    """
    with open(HERE / label_file, encoding="utf-8") as f:
        labels = {r["question_id"]: r for r in csv.DictReader(f)}
    judged = verdicts()
    pairs = [(labels[q]["label"], judged[q]["verdict"]) for q in labels]
    binary = [(a == "correct", b == "correct") for a, b in pairs]
    result = {
        "labels": label_file,
        "n": len(pairs),
        "exact_agreement": sum(a == b for a, b in pairs),
        "correct_vs_not_agreement": sum(a == b for a, b in binary),
        "kappa_three_grades": round(kappa(pairs), 3),
        "labeller_correct": sum(a == "correct" for a, _ in pairs),
        "judge_correct": sum(b == "correct" for _, b in pairs),
        "confusion_rows_labeller_cols_judge": {a: {b: sum(1 for x, y in pairs if (x, y) == (a, b)) for b in CATS}
                                               for a in CATS},
        "disagreements": [
            {"question_id": q, "labeller": labels[q]["label"], "judge": judged[q]["verdict"],
             "labeller_note": labels[q].get("note", ""), "judge_explanation": judged[q].get("explanation", "")}
            for q in labels if labels[q]["label"] != judged[q]["verdict"]
        ],
    }
    # kappa on the binary split, computed directly
    n = len(binary)
    po = sum(a == b for a, b in binary) / n
    pa, pb = sum(a for a, _ in binary) / n, sum(b for _, b in binary) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    result["kappa_correct_vs_not"] = round((po - pe) / (1 - pe), 3)
    name = Path(label_file).stem
    (HERE / f"comparison_{name}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    for f in sys.argv[1:]:
        r = compare(f)
        print(f, {k: v for k, v in r.items() if k not in ("disagreements",)})
