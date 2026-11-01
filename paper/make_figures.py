"""Draw the paper's data figures from the saved results.

Every figure reads the same files as make_numbers.py, in eval/results/, and is written as a PDF
to figures/. Edit this script, never its output. Run: uv run --group paper python paper/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from make_numbers import BASELINE, CANDIDATES, MULTIDOC, SWEEP

FIGURES = Path(__file__).resolve().parent / "figures"

# Categorical slots 1-3 of the dataviz reference palette, in fixed order; text in ink, not series colours
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED = "#0b0b0b", "#52514e"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK,
    "ytick.labelcolor": INK,
    "legend.frameon": False,
    "pdf.fonttype": 42,
})


def _interval(entry):
    """Rate and error-bar half-widths from a summary entry with a 95% interval.

    Args:
        entry (dict): A summary entry with "pct" and "ci95" ([low, high]) in percent.

    Returns:
        tuple: (rate, [[below], [above]]) for matplotlib's xerr or yerr.
    """
    return entry["pct"], [[entry["pct"] - entry["ci95"][0]], [entry["ci95"][1] - entry["pct"]]]


def _dot(ax, x, y, err, color, horizontal=True):
    """Draw one rate with its interval as a dot and a thin whisker.

    Args:
        ax (matplotlib.axes.Axes): The axes.
        x (float): Rate (horizontal) or position (vertical).
        y (float): Position (horizontal) or rate (vertical).
        err (list): [[below], [above]] half-widths.
        color (str): Marker colour.
        horizontal (bool): Whether the rate runs along the x axis.
    """
    kwargs = {"xerr": err} if horizontal else {"yerr": err}
    ax.errorbar(x, y, fmt="o", color=color, ms=5, elinewidth=1.2, capsize=0, mec="white", mew=0.8, **kwargs)


def answers_figure():
    """Figure: evidence in the context and judged correctness for each retrieval set-up.

    Shows that set-ups that retrieve evidence more often do not answer detectably better, with
    the oracle (the annotators' evidence) for reference.
    """
    base = json.loads((BASELINE / "summary.json").read_text())
    cands = {name: json.loads((folder / "summary.json").read_text()) for name, folder in CANDIDATES.items()}
    rows = [
        ("App: 1,000 characters, top 4", base["retrieval"]["hit_at_k"]["4"], base["conditions"]["fixed"]["correct"]),
        ("500 characters, top 8", cands["small"]["retrieval"]["hit_at_k"]["8"],
         cands["small"]["conditions"]["fixed"]["correct"]),
        ("500 characters, top 8, reranked", cands["small rerank"]["retrieval"]["hit_at_k"]["8"],
         cands["small rerank"]["conditions"]["fixed"]["correct"]),
        ("Annotators' evidence (oracle)", None, base["conditions"]["oracle"]["correct"]),
    ]
    fig, (left, right) = plt.subplots(1, 2, figsize=(6.3, 1.9), sharey=True, gridspec_kw={"wspace": 0.12})
    for i, (_, hit, correct) in enumerate(rows):
        y = len(rows) - 1 - i
        if hit:
            _dot(left, *_interval(hit)[:1], y, _interval(hit)[1], BLUE)
            left.annotate(f"{hit['pct']:.1f}", (hit["ci95"][1], y), xytext=(4, 0), textcoords="offset points",
                          va="center", fontsize=8, color=INK)
        else:
            left.text(42, y, "all evidence given", va="center", fontsize=8, color=MUTED)
        _dot(right, correct["pct"], y, _interval(correct)[1], BLUE)
        right.annotate(f"{correct['pct']:.1f}", (correct["ci95"][1], y), xytext=(4, 0), textcoords="offset points",
                       va="center", fontsize=8, color=INK)
    left.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)])
    left.set_xlim(40, 75)
    right.set_xlim(40, 75)
    left.set_xlabel("Gold evidence in the context (%)")
    right.set_xlabel("Answers judged correct (%)")
    for ax in (left, right):
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.tick_params(axis="y", length=0)
    fig.savefig(FIGURES / "answers.pdf", bbox_inches="tight")
    # A PNG copy for the repository README, which GitHub cannot show as PDF
    fig.savefig(FIGURES / "answers.png", bbox_inches="tight", dpi=200, facecolor="white")
    plt.close(fig)


def sweep_figure():
    """Figure: evidence in the context at equal prompt length for all retrieval set-ups.

    One panel per chunk size; retrievers in the same order in every panel, sorted by their mean
    over chunk sizes. The app's set-up is marked.
    """
    r = json.loads((SWEEP / "summary.json").read_text())["results"]
    sizes = ["c500", "c1000", "c1500"]
    retrievers = sorted({k.split("/")[1] for k in r}, key=lambda m: sum(r[f"{s}/{m}"]["hit@budget"][0] for s in sizes))
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 3.0), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, size in zip(axes, sizes, strict=True):
        for y, m in enumerate(retrievers):
            rate, low, high = r[f"{size}/{m}"]["hit@budget"]
            app = f"{size}/{m}" == "c1000/dense-minilm"
            _dot(ax, rate, y, [[rate - low], [high - rate]], ORANGE if app else BLUE)
            if app:
                ax.annotate("app", (high, y), xytext=(4, 0), textcoords="offset points", va="center",
                            fontsize=8, color=INK)
        k = r[f"{size}/dense-minilm"]["budget_k"]
        ax.set_title(f"{size[1:]} characters, top {k}", color=INK)
        ax.set_xlim(25, 75)
        ax.set_ylim(-0.6, len(retrievers) - 0.4)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(len(retrievers)), retrievers)
    axes[1].set_xlabel("Gold evidence in about 4,000 characters of context (%)")
    fig.savefig(FIGURES / "sweep.pdf", bbox_inches="tight")
    plt.close(fig)


def multidoc_figure():
    """Figure: evidence in the context as other papers are loaded, for the three retrieval rules."""
    s = json.loads((MULTIDOC / "summary.json").read_text())
    rules = [
        ("single", "top 4 of all papers", BLUE),
        ("multi", "top 4 + 1 per other paper (the app)", ORANGE),
        ("selected", "top 4 of the asked-about paper", AQUA),
    ]
    ns = sorted({int(k.split("/")[0]) for k in s})
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    offsets = {"single": -0.12, "multi": 0.0, "selected": 0.12}
    for rule, label, color in rules:
        rates = [s[f"{n}/{rule}"]["evidence_in_context"][2:] for n in ns]
        xs = [n + offsets[rule] for n in ns]
        ax.plot(xs, [v[0] for v in rates], color=color, lw=2, label=label)
        for x, (rate, low, high) in zip(xs, rates, strict=True):
            _dot(ax, x, rate, [[rate - low], [high - rate]], color, horizontal=False)
    ax.set_xticks(ns)
    ax.set_xlabel("Other papers loaded")
    ax.set_ylabel("Gold evidence in the context (%)")
    ax.set_ylim(20, 55)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, handlelength=1.5)
    fig.savefig(FIGURES / "multidoc.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    """Write every data figure."""
    FIGURES.mkdir(exist_ok=True)
    answers_figure()
    sweep_figure()
    multidoc_figure()
    print("figures written to", FIGURES)


if __name__ == "__main__":
    main()
