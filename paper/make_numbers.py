"""Write LaTeX macros and tables for the paper from the saved results.

Every number in the paper's text comes from a macro defined in generated/numbers.tex, and every
data table from a file in generated/, both written by this script from eval/results/. Edit
this script, never its output. Run: python paper/make_numbers.py
"""

import csv
import gzip
import json
import re
import statistics
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
REPEATS = RESULTS / "2026-11-01-repeats"
CITATIONS = RESULTS / "2026-11-01-citation-check"
SECTIONS = RESULTS / "2026-11-01-qasper-dev-sections"
SECTIONS_SMALL = RESULTS / "2026-11-01-qasper-dev-sections-c500-k8"
RETRY = RESULTS / "2026-11-01-qasper-dev-retry"
COMPARE = RESULTS / "2026-11-01-compare"


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

    # The two candidates against each other: does reranking change answers? (descriptive, outside the Holm family)
    if len(done) == 2:
        a, b = done["small"], done["small rerank"]
        both = a["scored"] & b["scored"]
        gained = len((b["correct"] - a["correct"]) & both)
        lost = len((a["correct"] - b["correct"]) & both)
        macros["rerank vs small gained"] = str(gained)
        macros["rerank vs small lost"] = str(lost)
        macros["rerank vs small p"] = pval(mcnemar_exact(gained, lost))
        # Split the A-vs-B change by whether reranking moved evidence into or out of the top 8
        def hits(folder):
            out = {}
            with gzip.open(folder / "retrieval.jsonl.gz", "rt", encoding="utf-8") as f:
                for line in f:
                    r = json.loads(line)
                    if not r["error"] and r["n_evidence"]:
                        out[r["question_id"]] = any(c["evidence"] for c in r["retrieved"][:8])
            return out

        hit_a, hit_b = hits(CANDIDATES["small"]), hits(CANDIDATES["small rerank"])
        added = [q for q in hit_b if hit_b[q] and not hit_a.get(q)]
        removed = [q for q in hit_b if hit_a.get(q) and not hit_b[q]]
        macros["rerank added n"] = str(len(added))
        macros["rerank added correct before"] = str(sum(q in a["correct"] for q in added))
        macros["rerank added correct after"] = str(sum(q in b["correct"] for q in added))
        macros["rerank removed n"] = str(len(removed))
        macros["rerank removed correct before"] = str(sum(q in a["correct"] for q in removed))
        macros["rerank removed correct after"] = str(sum(q in b["correct"] for q in removed))
        # Questions without text evidence count as unchanged (no evidence either way)
        same = [q for q in both if hit_a.get(q) == hit_b.get(q)]
        macros["rerank same n"] = str(len(same))
        macros["rerank same gained"] = str(sum(q in b["correct"] and q not in a["correct"] for q in same))
        macros["rerank same lost"] = str(sum(q in a["correct"] and q not in b["correct"] for q in same))

        retrieval = json.loads((CANDIDATES["small"] / "summary.json").read_text())["retrieval"]
        macros["small findable k"] = str(retrieval["evidence_found_in_pdf_chunks"]["k"])
        for name in done:
            hit = json.loads((CANDIDATES[name] / "summary.json").read_text())["retrieval"]["hit_at_k"]["8"]
            macros[f"{name} hit eight"] = f"{hit['pct']:.1f}\\% [{hit['ci95'][0]:.1f}, {hit['ci95'][1]:.1f}]"
            when = json.loads((CANDIDATES[name] / "summary.json").read_text())["conditions"]["fixed"]
            macros[f"{name} correct when hit"] = f"{when['correct_when_evidence_retrieved']['pct']:.1f}\\%"
            macros[f"{name} correct when miss"] = f"{when['correct_when_evidence_missed']['pct']:.1f}\\%"
    tables["candidates_rows.tex"] = "\n".join(rows) + "\n"


def matching(macros):
    """Macros for the SI section on evidence matching: downloads and findability by chunk size.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
    """
    papers = [json.loads(line) for line in (BASELINE / "papers.jsonl").open(encoding="utf-8")]
    failed = [p for p in papers if p["status"] != "ok"]
    macros["papers downloaded"] = str(len(papers) - len(failed))
    macros["papers failed"] = str(len(failed))
    macros["papers failed questions"] = str(sum(p["n_questions"] for p in failed))
    findable, chunks = {}, {}
    for path in sorted(SWEEP.glob("ranks_*.csv.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["retriever"] == "dense-minilm":
                    findable.setdefault(r["chunking"], set())
                    chunks.setdefault(r["chunking"], []).append(int(r["n_chunks"]))
                    if r["findable"] == "1":
                        findable[r["chunking"]].add(r["question_id"])
    for chunking, size in [("c500", "five hundred"), ("c1000", "thousand"), ("c1500", "fifteen hundred")]:
        macros[f"findable {size}"] = str(len(findable[chunking]))
        macros[f"median chunks {size}"] = str(int(statistics.median(chunks[chunking])))
    macros["empty pdfs"] = str(sum(min(c) == 0 for c in chunks.values()))


def judge_check(macros, tables):
    """Macros for the comparison of Claude's and the human labels with the judge.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
        tables (dict): File name to text, filled in place (the SI table of all labels).
    """
    c = json.loads((JUDGE_CHECK / "comparison_labels_claude.json").read_text())
    macros["judge check n"] = str(c["n"])
    macros["judge check exact"] = str(c["exact_agreement"])
    macros["judge check binary"] = str(c["correct_vs_not_agreement"])
    macros["judge check kappa"] = f"{c['kappa_three_grades']:.2f}"
    macros["judge check kappa binary"] = f"{c['kappa_correct_vs_not']:.2f}"
    macros["judge check judge correct"] = str(c["judge_correct"])
    macros["judge check claude correct"] = str(c["labeller_correct"])
    n = c["n"]
    macros["judge check judge rate"] = ci(wilson(c["judge_correct"], n))
    for name, stem in [("human", "labels_human"), ("human rule", "labels_human_rule_applied")]:
        h = json.loads((JUDGE_CHECK / f"comparison_{stem}.json").read_text())
        grid = h["confusion_rows_labeller_cols_judge"]
        macros[f"judge check {name} exact"] = str(h["exact_agreement"])
        macros[f"judge check {name} binary"] = str(h["correct_vs_not_agreement"])
        macros[f"judge check {name} kappa"] = f"{h['kappa_three_grades']:.2f}"
        macros[f"judge check {name} kappa binary"] = f"{h['kappa_correct_vs_not']:.2f}"
        macros[f"judge check {name} correct"] = str(h["labeller_correct"])
        macros[f"judge check {name} rate"] = ci(wilson(h["labeller_correct"], n))
        macros[f"judge check {name} judge only"] = str(sum(grid[a]["correct"] for a in grid if a != "correct"))
        macros[f"judge check {name} human only"] = str(sum(v for b, v in grid["correct"].items() if b != "correct"))
    with open(JUDGE_CHECK / "labels_human.csv", encoding="utf-8") as f:
        human = {r["question_id"]: r["label"] for r in csv.DictReader(f)}
    with open(JUDGE_CHECK / "labels_claude.csv", encoding="utf-8") as f:
        claude = {r["question_id"]: r["label"] for r in csv.DictReader(f)}
    macros["judge check human claude exact"] = str(sum(human[q] == claude[q] for q in human))
    judged = latest(RESULTS / "2026-11-01-qasper-dev" / "judgments_fixed.jsonl")
    with open(JUDGE_CHECK / "labels_human_rule_applied.csv", encoding="utf-8") as f:
        rule = {r["question_id"]: r["label"] for r in csv.DictReader(f)}
    with open(JUDGE_CHECK / "to_label_by_hand.csv", encoding="utf-8") as f:
        sample = list(csv.DictReader(f))
    short = {"correct": "C", "partially_correct": "P", "incorrect": "I"}
    rows = []
    for r in sample:
        q = r["question_id"]
        text = r["question"] if len(r["question"]) <= 70 else r["question"][:67].rstrip() + "..."
        mark = "$^\\dagger$" if rule[q] != human[q] else ""
        rows.append(f"{r['n']} & {latex_escape(text)} & {short[human[q]]}{mark} & {short[claude[q]]} & "
                    f"{short[judged[q]['verdict']]} \\\\")
    tables["judge_rows.tex"] = "\n".join(rows) + "\n"


def repeats(macros, tables):
    """Macros and the table for the repeat runs (seeds 0, 1 and 2 of four set-ups).

    The figures come from eval/results/2026-11-01-repeats/summary.json, written by
    eval/qasper/repeats.py.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
        tables (dict): File name to LaTeX text, filled in place.
    """
    s = json.loads((REPEATS / "summary.json").read_text())
    labels = {"fixed": "1,000 / top 4 (app)", "oracle": "Oracle evidence", "c500-k8": "500 / top 8",
              "c500-k8-rerank": "500 / top 8, reranked"}
    names = {"fixed": "fixed", "oracle": "oracle", "c500-k8": "small", "c500-k8-rerank": "small rerank"}
    rows, widest = [], None
    for setup, label in labels.items():
        e = s["setups"][setup]
        rates = [e["correct_by_seed"][seed]["pct"] for seed in "012"]
        name = f"repeat {names[setup]}"
        macros[f"{name} n"] = f"{e['questions_judged_in_all_runs']:,}"
        macros[f"{name} rates"] = ", ".join(f"{r:.1f}" for r in rates[:-1]) + f" and {rates[-1]:.1f}\\%"
        macros[f"{name} low"] = pct(min(rates))
        macros[f"{name} high"] = pct(max(rates))
        for key in ("correct_in_every_run", "correct_in_no_run", "same_correct_vs_not_all_runs",
                    "identical_answer_text_all_runs"):
            macros[f"{name} {key.replace('_', ' ')}"] = pct(e[key]["pct"])
        macros[f"{name} changed"] = pct(100 - e["same_correct_vs_not_all_runs"]["pct"])
        for d in e["between_runs"].values():
            if widest is None or abs(d["difference_points"]) > abs(widest[1]["difference_points"]):
                widest = (setup, d)
        cells = [label, f"{e['questions_judged_in_all_runs']:,}", *[f"{r:.1f}" for r in rates],
                 f"{e['correct_by_majority']['pct']:.1f}", f"{e['correct_in_every_run']['pct']:.1f}",
                 f"{e['correct_in_no_run']['pct']:.1f}", f"{e['same_correct_vs_not_all_runs']['pct']:.1f}"]
        rows.append(" & ".join(cells) + " \\\\")
    tables["repeats_rows.tex"] = "\n".join(rows) + "\n"

    # The largest difference between two runs of one set-up: the noise floor
    setup, d = widest
    macros["repeat widest setup"] = labels[setup].replace(" (app)", "")
    macros["repeat widest points"] = f"{abs(d['difference_points']):.1f}"
    macros["repeat widest gained"] = str(max(d["gained"], d["lost"]))
    macros["repeat widest lost"] = str(min(d["gained"], d["lost"]))
    macros["repeat widest p"] = pval(d["p"])

    for setup in ("c500-k8", "c500-k8-rerank", "oracle"):
        by_seed = s["comparisons"][f"{setup} vs fixed"]["by_seed"]
        name = f"repeat {names[setup]} vs fixed"
        diffs = [by_seed[seed]["difference_points"] for seed in "012"]
        macros[f"{name} diffs"] = ", ".join(f"{x:+.1f}" for x in diffs[:-1]) + f" and {diffs[-1]:+.1f}"
        macros[f"{name} ps"] = (", ".join(pval(by_seed[seed]["p"]) for seed in "01")
                                + f" and {pval(by_seed['2']['p'])}")
        macros[f"{name} significant runs"] = str(sum(by_seed[seed]["p"] < 0.05 for seed in "012"))
        macros[f"{name} max p"] = pval(max(by_seed[seed]["p"] for seed in "012"))

    j = s["judge_repeat"]
    macros["judge repeat n"] = f"{j['n']:,}"
    macros["judge repeat same"] = ci([j["same_verdict"]["pct"], *j["same_verdict"]["ci95"]])
    macros["judge repeat same k"] = str(j["same_verdict"]["k"])
    macros["judge repeat same binary"] = pct(j["same_correct_vs_not"]["pct"])
    macros["judge repeat changed binary"] = str(j["same_correct_vs_not"]["n"] - j["same_correct_vs_not"]["k"])
    macros["judge repeat diff"] = f"{abs(j['between']['difference_points']):.1f}"
    macros["judge repeat p"] = pval(j["between"]["p"])


def citations(macros):
    """Macros for the citation check: cited pages in context, and support of cited statements.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
    """
    s = json.loads((CITATIONS / "summary.json").read_text())
    c = json.loads((CITATIONS / "counts.json").read_text())
    macros["cite answers"] = f"{s['answers']:,}"
    macros["cite answers citing"] = f"{s['answers_citing']['k']:,}"
    macros["cite answers citing rate"] = pct(s["answers_citing"]["pct"])
    e = s["citations_in_context"]
    macros["cite pages"] = f"{e['n']:,}"
    macros["cite pages in context"] = f"{e['k']:,}"
    macros["cite pages in context rate"] = ci([e["pct"], *e["ci95"]])
    macros["cite pages in context pct"] = pct(e["pct"])
    macros["cite pages outside"] = str(e["n"] - e["k"])
    macros["cite statements"] = f"{c['statements']:,}"
    macros["cite judged"] = str(s["judged_statements"])
    grades = (("supported", "supported"), ("partially_supported", "partial"), ("not_supported", "not"))
    for key, name in grades:
        v = s["support"][key]
        macros[f"cite all {name}"] = ci([v["pct"], *v["ci95"]])
    h = s["hand_labels"]
    macros["cite hand n"] = str(h["n"])
    macros["cite unlabelled"] = str(s["judged_statements"] - h["n"])
    for who in ("labeller", "judge"):
        for key, name in grades:
            macros[f"cite {who} {name}"] = str(h[who][key])
        macros[f"cite {who} rate"] = pct(100 * h[who]["supported"] / h["n"])
    macros["cite hand exact"] = str(h["exact_agreement"])
    macros["cite hand kappa"] = f"{h['kappa']:.2f}"
    macros["cite not judged supported"] = str(h["confusion_rows_labeller_cols_judge"]["not_supported"]["supported"])


def retrieval_hits(folder, k):
    """Whether the top k chunks held evidence, per question with text evidence.

    Args:
        folder (Path): A results folder with retrieval.jsonl.gz.
        k (int): Chunks counted.

    Returns:
        dict: question id to True or False.
    """
    out = {}
    with gzip.open(folder / "retrieval.jsonl.gz", "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if not r["error"] and r["n_evidence"]:
                out[r["question_id"]] = any(c["evidence"] for c in r["retrieved"][:k])
    return out


def paired(macros, name, before, after):
    """Macros for a paired comparison of two sets of per-question outcomes.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
        name (str): Macro prefix.
        before (dict): question id to True or False.
        after (dict): question id to True or False, on the same questions.
    """
    both = [q for q in before if q in after]
    gained = sum(after[q] and not before[q] for q in both)
    lost = sum(before[q] and not after[q] for q in both)
    k_before, k_after = sum(before[q] for q in both), sum(after[q] for q in both)
    macros[f"{name} n"] = f"{len(both):,}"
    macros[f"{name} before"] = ci(wilson(k_before, len(both)))
    macros[f"{name} after"] = ci(wilson(k_after, len(both)))
    macros[f"{name} before rate"] = pct(100 * k_before / len(both))
    macros[f"{name} after rate"] = pct(100 * k_after / len(both))
    macros[f"{name} gained"] = str(gained)
    macros[f"{name} lost"] = str(lost)
    macros[f"{name} p"] = pval(mcnemar_exact(gained, lost))


def correct(folder):
    """Whether each scored question of a run was judged correct (a missing verdict is not).

    Args:
        folder (Path): A results folder with answers_fixed.jsonl and judgments_fixed.jsonl.

    Returns:
        dict: question id to True or False.
    """
    answers = latest(folder / "answers_fixed.jsonl")
    verdicts = latest(folder / "judgments_fixed.jsonl")
    return {q: verdicts.get(q, {}).get("verdict") == "correct" for q, a in answers.items() if not a.get("skipped")}


def changes(macros):
    """Macros for section-aware chunking, retry on a miss and comparison questions.

    Args:
        macros (dict): Name to LaTeX text, filled in place.
    """
    coverage = json.loads((SECTIONS / "section_coverage.json").read_text())["section_names_found"]
    macros["sections coverage"] = pct(coverage["pct"])
    macros["sections coverage k"] = f"{coverage['k']:,}"
    macros["sections coverage n"] = f"{coverage['n']:,}"
    paired(macros, "sections hit", retrieval_hits(BASELINE, 4), retrieval_hits(SECTIONS, 4))
    paired(macros, "sections small hit", retrieval_hits(CANDIDATES["small"], 8), retrieval_hits(SECTIONS_SMALL, 8))
    base_ok = correct(BASELINE)
    paired(macros, "sections answers", base_ok, correct(SECTIONS))

    checks = latest(RETRY / "retry.jsonl")
    retried = {q for q, c in checks.items() if c["query"]}
    before, after = retrieval_hits(BASELINE, 4), retrieval_hits(RETRY, 4)
    macros["retry checked"] = f"{len(checks):,}"
    macros["retry n"] = str(len(retried))
    macros["retry rate"] = pct(100 * len(retried) / len(checks))
    missed = [q for q in before if not before[q] and q in checks]
    held = [q for q in before if before[q] and q in checks]
    kept = sum(q not in retried for q in missed)
    macros["retry missed"] = str(len(missed))
    macros["retry missed kept"] = str(kept)
    macros["retry missed kept rate"] = pct(100 * kept / len(missed))
    macros["retry held"] = str(len(held))
    macros["retry held retried"] = str(sum(q in retried for q in held))
    paired(macros, "retry hit", before, after)
    retry_ok = correct(RETRY)
    paired(macros, "retry answers", base_ok, retry_ok)
    on = [q for q in retried if q in base_ok and q in retry_ok]
    macros["retry retried before"] = str(sum(base_ok[q] for q in on))
    macros["retry retried after"] = str(sum(retry_ok[q] for q in on))

    s = json.loads((COMPARE / "summary.json").read_text())
    f = json.loads((COMPARE / "findable.json").read_text())
    macros["compare pairs"] = str(s["pairs"])
    macros["compare findable both"] = str(f["pairs_findable_in_both"])
    macros["compare papers"] = str(f["papers"])
    macros["compare papers findable"] = str(f["papers_findable"])
    for rule in ("app", "balanced"):
        r = s["rules"][rule]
        both = r["correct_for_both_papers"]
        macros[f"compare {rule} both evidence"] = str(r["evidence_from_both_papers"]["k"])
        macros[f"compare {rule} both correct"] = ci([both["pct"], *both["ci95"]])
        macros[f"compare {rule} both correct n"] = str(both["n"])
        macros[f"compare {rule} papers correct"] = pct(r["papers_answered_correctly"]["pct"])
        slots = f[f"{rule}_paper_slots_with_evidence"]
        macros[f"compare {rule} slots"] = str(slots["k"])
        macros[f"compare {rule} slots n"] = str(slots["n"])
    t = s["balanced_vs_app_correct"]
    macros["compare n"] = str(t["n"])
    macros["compare gained"] = str(t["gained"])
    macros["compare lost"] = str(t["lost"])
    macros["compare p"] = pval(t["p"])


def latex_escape(text):
    """Escape the characters LaTeX treats as special.

    Args:
        text (str): Plain text.

    Returns:
        str: Text safe to place in a LaTeX table cell.
    """
    out = text.replace("\\", "\\textbackslash{}")
    for c in "&%$#_{}":
        out = out.replace(c, "\\" + c)
    return out.replace("~", "\\textasciitilde{}").replace("^", "\\textasciicircum{}")


def prompts(tables):
    """Copy the prompts, exactly as the code defines them, for listing in the SI.

    Args:
        tables (dict): File name to text, filled in place.
    """
    import sys

    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "eval" / "qasper"))
    from citation_check import SUPPORT_PROMPT
    from run_eval import JUDGE_PROMPT, SHIPPED_SYSTEM_PROMPT, UNANSWERABLE_RULE

    from src.utils.utils import SYSTEM_PROMPT, Utils

    tables["prompt_shipped.txt"] = SHIPPED_SYSTEM_PROMPT
    tables["prompt_system.txt"] = SYSTEM_PROMPT
    example = Utils.build_messages("[p. 4] <retrieved chunk>\n\n[p. 5] <retrieved chunk>", "<question>", [])
    tables["prompt_user.txt"] = example[-1]["content"]
    tables["prompt_judge.txt"] = JUDGE_PROMPT
    tables["prompt_judge_unanswerable.txt"] = UNANSWERABLE_RULE.strip()
    tables["prompt_support.txt"] = SUPPORT_PROMPT


def main():
    """Write generated/numbers.tex and the table files."""
    OUT.mkdir(exist_ok=True)
    macros, tables = {}, {}
    sweep(macros, tables)
    multidoc(macros, tables)
    baseline(macros, tables)
    candidates(macros, tables)
    matching(macros)
    judge_check(macros, tables)
    repeats(macros, tables)
    citations(macros)
    changes(macros)
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
