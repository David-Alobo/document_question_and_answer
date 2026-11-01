"""Run-to-run variation: the same questions answered three times (seeds 0, 1, 2) and judged.

Seed 0 is the original run of each set-up; seeds 1 and 2 are in eval/results/2026-11-01-repeats/.
For each set-up the script reports each run's correct rate, how often a question's verdict changes
between runs, and the size of the differences between runs of the same set-up next to the
differences between set-ups that the paper reports. The seed-0 fixed answers judged a second time
(judge seed 1) separate the judge's own variation from the answer model's.

Usage: python eval/qasper/repeats.py   (writes eval/results/2026-11-01-repeats/summary.json)
"""

import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import _rate, _read_jsonl

RESULTS = Path(__file__).resolve().parents[1] / "results"
REPEATS = RESULTS / "2026-11-01-repeats"
SETUPS = {
    "fixed": ("2026-11-01-qasper-dev", "fixed", "fixed"),
    "oracle": ("2026-11-01-qasper-dev", "oracle", "oracle"),
    "c500-k8": ("2026-11-01-qasper-dev-c500-k8", "fixed", "c500-k8"),
    "c500-k8-rerank": ("2026-11-01-qasper-dev-c500-k8-rerank", "fixed", "c500-k8-rerank"),
}
SEEDS = (0, 1, 2)
COMPARISONS = [("c500-k8", "fixed"), ("c500-k8-rerank", "fixed"), ("oracle", "fixed")]


def verdicts(folder, condition):
    """The last verdict per question in one run, leaving out failed judgements.

    Args:
        folder (Path): Results folder.
        condition (str): Condition name in the file names.

    Returns:
        dict: question id to verdict.
    """
    latest = {r["question_id"]: r for r in _read_jsonl(folder / f"judgments_{condition}.jsonl")}
    return {q: r["verdict"] for q, r in latest.items() if r.get("verdict")}


def answers(folder, condition):
    """The answer text per question in one run.

    Args:
        folder (Path): Results folder.
        condition (str): Condition name in the file names.

    Returns:
        dict: question id to answer text.
    """
    return {r["question_id"]: r.get("answer") for r in _read_jsonl(folder / f"answers_{condition}.jsonl")}


def mcnemar(gained, lost):
    """Exact two-sided McNemar p-value from the discordant pairs.

    Args:
        gained (int): Pairs correct only in the first run.
        lost (int): Pairs correct only in the second run.

    Returns:
        float: p-value.
    """
    from math import comb

    n = gained + lost
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(gained, lost) + 1)) / 2**n)


def paired(a, b):
    """Correct-vs-not comparison of two runs on the questions both judged.

    Args:
        a (dict): question id to verdict, first run.
        b (dict): question id to verdict, second run.

    Returns:
        dict: n, difference in correct rate (points), gained, lost and McNemar p.
    """
    qs = sorted(set(a) & set(b))
    gained = sum(a[q] == "correct" and b[q] != "correct" for q in qs)
    lost = sum(a[q] != "correct" and b[q] == "correct" for q in qs)
    return {"n": len(qs), "difference_points": round(100 * (gained - lost) / len(qs), 1),
            "gained": gained, "lost": lost, "p": mcnemar(gained, lost)}


def main():
    """Compute the summary and write it next to the repeat runs."""
    runs, texts = {}, {}
    for name, (base, condition, prefix) in SETUPS.items():
        for seed in SEEDS:
            folder = RESULTS / base if seed == 0 else REPEATS / f"{prefix}_seed{seed}"
            runs[name, seed] = verdicts(folder, condition)
            texts[name, seed] = answers(folder, condition)
    summary = {"setups": {}, "comparisons": {}}
    for name in SETUPS:
        per = [runs[name, s] for s in SEEDS]
        qs = sorted(set.intersection(*(set(v) for v in per)))
        same3 = sum(len({v[q] for v in per}) == 1 for q in qs)
        same2 = sum(len({v[q] == "correct" for v in per}) == 1 for q in qs)
        always = sum(all(v[q] == "correct" for v in per) for q in qs)
        never = sum(all(v[q] != "correct" for v in per) for q in qs)
        majority = sum(sum(v[q] == "correct" for v in per) >= 2 for q in qs)
        identical = sum(len({texts[name, s].get(q) for s in SEEDS}) == 1 for q in qs)
        summary["setups"][name] = {
            "questions_judged_in_all_runs": len(qs),
            "correct_by_seed": {s: _rate(sum(runs[name, s][q] == "correct" for q in qs), len(qs)) for s in SEEDS},
            "correct_by_majority": _rate(majority, len(qs)),
            "correct_in_every_run": _rate(always, len(qs)),
            "correct_in_no_run": _rate(never, len(qs)),
            "same_verdict_all_runs": _rate(same3, len(qs)),
            "same_correct_vs_not_all_runs": _rate(same2, len(qs)),
            "identical_answer_text_all_runs": _rate(identical, len(qs)),
            "between_runs": {f"{x}-{y}": paired(runs[name, x], runs[name, y])
                             for x, y in itertools.combinations(SEEDS, 2)},
        }
    for a, b in COMPARISONS:
        summary["comparisons"][f"{a} vs {b}"] = {
            "by_seed": {s: paired(runs[a, s], runs[b, s]) for s in SEEDS},
        }
    judge_again = verdicts(REPEATS / "fixed_judge_seed1", "fixed")
    first = runs["fixed", 0]
    qs = sorted(set(first) & set(judge_again))
    summary["judge_repeat"] = {
        "n": len(qs),
        "same_verdict": _rate(sum(first[q] == judge_again[q] for q in qs), len(qs)),
        "same_correct_vs_not": _rate(sum((first[q] == "correct") == (judge_again[q] == "correct") for q in qs), len(qs)),
        "between": paired(first, judge_again),
    }
    (REPEATS / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
