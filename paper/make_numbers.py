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
