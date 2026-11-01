"""Write LaTeX macros and tables for the paper from the saved results.

Every number in the paper's text comes from a macro defined in generated/numbers.tex, and every
data table from a file in generated/, both written by this script from eval/results/. Edit
this script, never its output. Run: python paper/make_numbers.py
"""

import csv
import gzip
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "eval" / "results"
OUT = Path(__file__).resolve().parent / "generated"

SWEEP = RESULTS / "2026-11-01-retrieval-sweep-dev"
MULTIDOC = RESULTS / "2026-11-01-multidoc-dev"
BASELINE = RESULTS / "2026-11-01-qasper-dev"
JUDGE_CHECK = RESULTS / "2026-11-01-judge-validation"
# The two leading set-ups of the sweep, run on answers; macro prefix to folder
CANDIDATES = {
    "small": RESULTS / "2026-11-01-qasper-dev-c500-k8",
    "small rerank": RESULTS / "2026-11-01-qasper-dev-c500-k8-rerank",
}


def wilson(k, n, z=1.96):
    """Wilson score interval in percent.

    Args:
        k (int): Successes.
        n (int): Trials.
        z (float): Normal quantile.

    Returns:
        list: [rate, low, high] in percent.
    """
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return [100 * p, 100 * max(0.0, centre - half), 100 * min(1.0, centre + half)]


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test from the discordant counts.

    Args:
        b (int): Pairs where only the first condition succeeds.
        c (int): Pairs where only the second succeeds.

    Returns:
        float: The p-value.
    """
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2**n)


def latest(path):
    """Read a JSON-lines file of per-question records, keeping the last record per question.

    Args:
        path (Path): The file.

    Returns:
        dict: question id to record.
    """
    out = {}
    for line in path.open(encoding="utf-8"):
        record = json.loads(line)
        out[record["question_id"]] = record
    return out


def macro_name(name):
    """Turn a descriptive name into a LaTeX macro name (letters only).

    Args:
        name (str): e.g. "sweep base budget".

    Returns:
        str: e.g. "SweepBaseBudget".
    """
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six",
              "7": "Seven", "8": "Eight", "9": "Nine"}
    words = re.split(r"[^A-Za-z0-9]+", name)
    out = "".join(w[:1].upper() + w[1:] for w in words if w)
    return "".join(digits.get(c, c) for c in out)


def pct(x):
    """Format a percentage to one decimal.

    Args:
        x (float): Percentage.

    Returns:
        str: e.g. "49.4\\%".
    """
    return f"{x:.1f}\\%"


def ci(values):
    """Format [rate, low, high] as "rate [low, high]".

    Args:
        values (list): Percentages.

    Returns:
        str: LaTeX text.
    """
    return f"{values[0]:.1f}\\% [{values[1]:.1f}, {values[2]:.1f}]"


def pval(p):
    """Format a p-value: two significant figures, scientific below 0.001.

    Args:
        p (float): The p-value.

    Returns:
        str: LaTeX text (math mode where needed).
    """
    if p < 1e-3:
        mantissa, exponent = f"{p:.1e}".split("e")
        return f"${mantissa} \\times 10^{{{int(exponent)}}}$"
    return f"{p:.2g}"


def sweep(macros, tables):
    """Macros and the SI table for the retrieval sweep.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
        tables (dict): File name to LaTeX text, filled in place.
    """
    s = json.loads((SWEEP / "summary.json").read_text())
    r, t = s["results"], s["paired_vs_baseline"]
    macros["sweep questions"] = str(s["questions_with_evidence_findable_in_all_chunkings"])
    macros["sweep setups"] = str(len(r))
    macros["sweep comparisons"] = str(len(t["hit@budget"]))
    named = {
        "base": "c1000/dense-minilm",
        "small": "c500/dense-minilm",
        "small rerank": "c500/rerank-dense-minilm",
        "best": max(r, key=lambda c: r[c]["hit@budget"][0]),
        "big hybrid": "c1500/hybrid-minilm",
    }
    for label, config in named.items():
        macros[f"sweep {label} budget"] = ci(r[config]["hit@budget"])
        macros[f"sweep {label} budget rate"] = pct(r[config]["hit@budget"][0])
        macros[f"sweep {label} hit four"] = ci(r[config]["hit@4"])
        if config in t["hit@budget"]:
            c = t["hit@budget"][config]
            macros[f"sweep {label} gained"] = str(c["gained"])
            macros[f"sweep {label} lost"] = str(c["lost"])
            macros[f"sweep {label} p"] = pval(c["p_holm"])
    macros["sweep best name"] = "\\texttt{" + named["best"] + "}"

    rows = []
    for config in sorted(r, key=lambda c: -r[c]["hit@budget"][0]):
        e = r[config]
        c = t["hit@budget"].get(config)
        cmp = "--" if c is None else f"+{c['gained']} / $-${c['lost']} & {pval(c['p_holm'])}"
        if c is None:
            cmp = "\\multicolumn{2}{c}{baseline}"
        chunking, retriever = config.split("/")
        rows.append(f"{chunking[1:]} & {retriever} & {e['budget_k']} & {e['hit@budget'][0]:.1f} "
                    f"[{e['hit@budget'][1]:.1f}, {e['hit@budget'][2]:.1f}] & {e['hit@4'][0]:.1f} & "
                    f"{e['hit@1'][0]:.1f} & {e['mrr@10']:.3f} & {cmp} \\\\")
    tables["sweep_rows.tex"] = "\n".join(rows) + "\n"


def multidoc(macros, tables):
    """Macros and the table for the multi-document experiment.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
        tables (dict): File name to LaTeX text, filled in place.
    """
    s = json.loads((MULTIDOC / "summary.json").read_text())
    with gzip.open(MULTIDOC / "results.csv.gz", "rt", newline="") as f:
        n_questions = len({r["question_id"] for r in csv.DictReader(f)})
    macros["multidoc questions"] = str(n_questions)
    words = {0: "zero", 2: "two", 4: "four", 9: "nine"}
    for key, e in s.items():
        n, rule = key.split("/")
        name = f"multidoc {words[int(n)]} {rule}"
        macros[name] = ci(e["evidence_in_context"][2:])
        macros[name + " lost"] = str(e["lost_vs_alone"])
        macros[name + " own"] = f"{e['mean_own_paper_chunks']:.2f}"
        macros[name + " k"] = str(e["evidence_in_context"][0])
    rows = []
    for key in sorted(s, key=lambda k: (int(k.split("/")[0]), ["single", "multi", "selected"].index(k.split("/")[1]))):
        n, rule = key.split("/")
        e = s[key]
        v = e["evidence_in_context"]
        rows.append(f"{n} & {rule} & {v[2]:.1f} [{v[3]:.1f}, {v[4]:.1f}] & {e['mean_context_chunks']:.2f} & "
                    f"{e['mean_own_paper_chunks']:.2f} & {e['lost_vs_alone']} / {e['gained_vs_alone']} \\\\")
    tables["multidoc_rows.tex"] = "\n".join(rows) + "\n"


def baseline(macros, tables):
    """Macros and the answers table for the shipped, fixed and oracle conditions.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
        tables (dict): File name to LaTeX text, filled in place.
    """
    s = json.loads((BASELINE / "summary.json").read_text())
    answers = {c: latest(BASELINE / f"answers_{c}.jsonl") for c in ("shipped", "fixed", "oracle")}
    verdicts = {c: latest(BASELINE / f"judgments_{c}.jsonl") for c in ("shipped", "fixed", "oracle")}
    macros["baseline questions"] = f"{s['questions']:,}"
    macros["baseline papers"] = str(s["papers"])
    macros["baseline pdfs"] = str(s["papers_with_pdf"])
    macros["shipped empty"] = f"{sum(1 for a in answers['shipped'].values() if not (a.get('answer') or '').strip()):,}"

    rows = []
    for c, label in (("shipped", "Shipped"), ("fixed", "Fixed"), ("oracle", "Oracle")):
        e = s["conditions"][c]
        cor, cop = e["correct"], e["correct_or_partial"]
        macros[f"{c} n"] = f"{cor['n']:,}"
        macros[f"{c} correct"] = f"{cor['pct']:.1f}\\% [{cor['ci95'][0]:.1f}, {cor['ci95'][1]:.1f}]"
        macros[f"{c} correct rate"] = f"{cor['pct']:.1f}\\%"
        macros[f"{c} correct k"] = str(cor["k"])
        macros[f"{c} partial"] = f"{cop['pct']:.1f}\\% [{cop['ci95'][0]:.1f}, {cop['ci95'][1]:.1f}]"
        macros[f"{c} f one"] = f"{e['answer_f1_official']:.1f}"
        rows.append(f"{label} & {cor['n']:,} & {cor['pct']:.1f} [{cor['ci95'][0]:.1f}, {cor['ci95'][1]:.1f}] & "
                    f"{cop['pct']:.1f} [{cop['ci95'][0]:.1f}, {cop['ci95'][1]:.1f}] & {e['answer_f1_official']:.1f} \\\\")
    tables["answers_rows.tex"] = "\n".join(rows) + "\n"

    # Fixed against oracle, paired on the questions both answered
    both = [q for q in verdicts["oracle"] if verdicts["oracle"][q].get("verdict") and verdicts["fixed"].get(q, {}).get("verdict")]
    gained = sum(verdicts["oracle"][q]["verdict"] == "correct" and verdicts["fixed"][q]["verdict"] != "correct" for q in both)
    lost = sum(verdicts["fixed"][q]["verdict"] == "correct" and verdicts["oracle"][q]["verdict"] != "correct" for q in both)
    fixed_on_both = sum(verdicts["fixed"][q]["verdict"] == "correct" for q in both)
    macros["paired oracle n"] = str(len(both))
    macros["paired fixed on oracle questions"] = ci(wilson(fixed_on_both, len(both)))
    macros["paired oracle gained"] = str(gained)
    macros["paired oracle lost"] = str(lost)
    macros["paired oracle p"] = pval(mcnemar_exact(gained, lost))

    r = s["retrieval"]
    macros["text evidence questions"] = str(r["questions_with_text_evidence"])
    f = r["evidence_found_in_pdf_chunks"]
    macros["findable"] = f"{f['pct']:.1f}\\% [{f['ci95'][0]:.1f}, {f['ci95'][1]:.1f}]"
    macros["findable k"] = str(f["k"])
    h = r["hit_at_k"]["4"]
    macros["hit four"] = f"{h['pct']:.1f}\\% [{h['ci95'][0]:.1f}, {h['ci95'][1]:.1f}]"
    macros["hit ten"] = f"{r['hit_at_k']['10']['pct']:.1f}\\%"
    for key, name in (("correct_when_evidence_retrieved", "when hit"), ("correct_when_evidence_missed", "when miss")):
        e = s["conditions"]["fixed"][key]
        macros[f"correct {name}"] = f"{e['pct']:.1f}\\% [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}]"
        macros[f"correct {name} n"] = str(e["n"])

    type_rows = []
    names = {"extractive": "Extractive", "abstractive": "Abstractive", "boolean": "Yes/no", "none": "Unanswerable"}
    for t in ("extractive", "abstractive", "boolean", "none"):
        cells = []
        for c in ("fixed", "oracle"):
            e = s["conditions"][c]["correct_by_first_annotator_type"].get(t)
            f1 = s["conditions"][c]["answer_f1_by_type"].get(t)
            cells.append("--" if not e else f"{e['pct']:.1f} ({e['n']})")
            cells.append(f"{f1:.1f}")
        type_rows.append(f"{names[t]} & " + " & ".join(cells) + " \\\\")
    tables["types_rows.tex"] = "\n".join(type_rows) + "\n"


def candidates(macros, tables):
    """Macros and table rows for the leading retrieval set-ups against the fixed app, on answers.

    Each candidate is compared with the fixed app on the questions both scored, with an exact
    McNemar test on correct versus not (a missing verdict counts as not correct, as in the
    summaries), Holm-corrected over the candidates that have finished. Folders without a
    summary yet are skipped.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
        tables (dict): File name to LaTeX text, filled in place.
    """
    import statistics

    def row(folder):
        s = json.loads((folder / "summary.json").read_text())["conditions"]["fixed"]
        answers = latest(folder / "answers_fixed.jsonl")
        verdicts = latest(folder / "judgments_fixed.jsonl")
        scored = {q for q, a in answers.items() if not a.get("skipped")}
        timed = [a for q, a in answers.items() if q in scored and a.get("usage")]
        return {
            "summary": s,
            "correct": {q for q in scored if verdicts.get(q, {}).get("verdict") == "correct"},
            "scored": scored,
            "prompt_tokens": statistics.mean(a["usage"]["prompt_tokens"] for a in timed),
            "seconds": statistics.median(a["seconds"] for a in timed),
        }

    base = row(BASELINE)
    done = {name: row(folder) for name, folder in CANDIDATES.items() if (folder / "summary.json").exists()}

    tests = {}
    for name, r in done.items():
        both = base["scored"] & r["scored"]
        gained = len((r["correct"] - base["correct"]) & both)
        lost = len((base["correct"] - r["correct"]) & both)
        tests[name] = (len(both), gained, lost, mcnemar_exact(gained, lost))
    # Holm over the finished comparisons
    holm, running = {}, 0.0
    ordered = sorted(tests, key=lambda n: tests[n][3])
    for i, name in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * tests[name][3]))
        holm[name] = running

    labels = {"base": "1,000 / top 4 (app)", "small": "500 / top 8", "small rerank": "500 / top 8, reranked"}
    rows = []
    for name, r in [("base", base), *done.items()]:
        cor, cop = r["summary"]["correct"], r["summary"]["correct_or_partial"]
        cells = [labels[name], f"{cor['n']:,}", f"{cor['pct']:.1f} [{cor['ci95'][0]:.1f}, {cor['ci95'][1]:.1f}]",
                 f"{cop['pct']:.1f}", f"{r['summary']['answer_f1_official']:.1f}", f"{r['prompt_tokens']:,.0f}"]
        if name == "base":
            cells.append("--")
        else:
            n, gained, lost, _ = tests[name]
            cells.append(f"{gained} / {lost} (${pval(holm[name])}$)")
            macros[f"{name} correct"] = f"{cor['pct']:.1f}\\% [{cor['ci95'][0]:.1f}, {cor['ci95'][1]:.1f}]"
            macros[f"{name} correct rate"] = f"{cor['pct']:.1f}\\%"
            macros[f"{name} partial"] = f"{cop['pct']:.1f}\\%"
            macros[f"{name} paired n"] = f"{n:,}"
            macros[f"{name} gained"] = str(gained)
            macros[f"{name} lost"] = str(lost)
            macros[f"{name} holm p"] = pval(holm[name])
            macros[f"{name} not judged"] = str(r["summary"]["not_judged"])
        macros[f"{name} prompt tokens"] = f"{r['prompt_tokens']:,.0f}"
        macros[f"{name} seconds"] = f"{r['seconds']:.1f}"
        rows.append(" & ".join(cells) + " \\\\")
    macros["candidate comparisons"] = str(len(tests))
    tables["candidates_rows.tex"] = "\n".join(rows) + "\n"


def judge_check(macros):
    """Macros for the comparison of Claude's labels with the judge.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
    """
    c = json.loads((JUDGE_CHECK / "comparison_labels_claude.json").read_text())
    macros["judge check n"] = str(c["n"])
    macros["judge check exact"] = str(c["exact_agreement"])
    macros["judge check binary"] = str(c["correct_vs_not_agreement"])
    macros["judge check kappa"] = f"{c['kappa_three_grades']:.2f}"
    macros["judge check kappa binary"] = f"{c['kappa_correct_vs_not']:.2f}"
    macros["judge check judge correct"] = str(c["judge_correct"])
    macros["judge check claude correct"] = str(c["labeller_correct"])


def prompts(tables):
    """Copy the prompts, exactly as the code defines them, for listing in the SI.

    Args:
        tables (dict): File name to text, filled in place.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "eval" / "qasper"))
    from run_eval import JUDGE_PROMPT, SHIPPED_SYSTEM_PROMPT, UNANSWERABLE_RULE

    from src.utils.utils import SYSTEM_PROMPT, Utils

    tables["prompt_shipped.txt"] = SHIPPED_SYSTEM_PROMPT
    tables["prompt_system.txt"] = SYSTEM_PROMPT
    example = Utils.build_messages("[p. 4] <retrieved chunk>\n\n[p. 5] <retrieved chunk>", "<question>", [])
    tables["prompt_user.txt"] = example[-1]["content"]
    tables["prompt_judge.txt"] = JUDGE_PROMPT
    tables["prompt_judge_unanswerable.txt"] = UNANSWERABLE_RULE.strip()


def main():
    """Write generated/numbers.tex and the table files."""
    OUT.mkdir(exist_ok=True)
    macros, tables = {}, {}
    sweep(macros, tables)
    multidoc(macros, tables)
    baseline(macros, tables)
    candidates(macros, tables)
    judge_check(macros)
    prompts(tables)
    lines = ["% Generated by make_numbers.py from eval/results/. Do not edit."]
    lines += [f"\\newcommand{{\\{macro_name(k)}}}{{{v}}}" for k, v in sorted(macros.items())]
    (OUT / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for name, text in tables.items():
        header = "" if name.endswith(".txt") else "% Generated by make_numbers.py. Do not edit.\n"
        (OUT / name).write_text(header + text, encoding="utf-8")
    print(f"{len(macros)} macros, {len(tables)} tables")


if __name__ == "__main__":
    main()
