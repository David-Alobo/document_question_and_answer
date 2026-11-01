"""Compare retrieval set-ups on QASPER without calling a language model.

For every question, each set-up ranks the paper's chunks; we record the rank of the first
chunk that holds a gold evidence paragraph (matched as in run_eval.py). Set-ups vary:

    chunking   500/100, 1000/200 (the app's), 1500/300 characters (size/overlap)
    retriever  dense (MiniLM: the app's; BGE-small; E5-small), BM25 keywords,
               hybrid (dense + BM25 by reciprocal rank fusion), and MiniLM dense,
               BM25 and the three hybrids reranked with a cross-encoder over their
               top 20: 12 retrievers x 3 chunkings = 36 set-ups

Metrics are computed on the questions whose evidence is findable under every chunking,
so every set-up is scored on the same questions. "Budget" counts a hit within the first
~4000 characters of context (k = 4000 / chunk size), so chunk sizes compare at the same
prompt length. Paired exact McNemar tests compare each set-up with the app's (Holm).

Usage:
    python eval/qasper/retrieval_sweep.py run --data eval/data --out <folder>
    python eval/qasper/retrieval_sweep.py summarize --out <folder>
"""

import argparse
import csv
import gzip
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_eval import _load_split, _ngrams, _tokens, matches, references, wilson

CHUNKINGS = {"c500": (500, 100), "c1000": (1000, 200), "c1500": (1500, 300)}
DENSE = {
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", "", ""),
    "bge": ("BAAI/bge-small-en-v1.5", "Represent this sentence for searching relevant passages: ", ""),
    "e5": ("intfloat/e5-small-v2", "query: ", "passage: "),
}
RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_DEPTH = 20
RRF_K = 60
BUDGET_CHARS = 4000
BASELINE = "c1000/dense-minilm"
KS = (1, 3, 4, 5, 10)


def retrievers():
    """Names of the retrievers compared, in report order.

    Returns:
        list: Retriever names.
    """
    base = [f"dense-{m}" for m in DENSE] + ["bm25"] + [f"hybrid-{m}" for m in DENSE]
    return base + [f"rerank-{b}" for b in ("dense-minilm", "bm25", *[f"hybrid-{m}" for m in DENSE])]


def _rrf(*rankings):
    """Fuse rankings by reciprocal rank fusion.

    Args:
        *rankings: Arrays of chunk indices, best first.

    Returns:
        np.ndarray: Chunk indices, best first.
    """
    score = np.zeros(len(rankings[0]))
    for ranking in rankings:
        score[ranking] += 1.0 / (RRF_K + np.arange(1, len(ranking) + 1))
    return np.argsort(-score, kind="stable")


def run(args):
    """Rank every paper's chunks for every question under every set-up; append to ranks.csv.gz.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    import torch
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
    from sentence_transformers import CrossEncoder, SentenceTransformer

    torch.set_num_threads(args.threads)
    data_dir, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / ("ranks.csv.gz" if args.shard is None else f"ranks_{args.shard}.csv.gz")
    done = set()
    if path.exists():
        with gzip.open(path, "rt", newline="") as f:
            done = {row["paper_id"] for row in csv.DictReader(f)}
    papers = _load_split(data_dir, args.split, args.limit)
    models = {name: SentenceTransformer(spec[0]) for name, spec in DENSE.items()}
    reranker = CrossEncoder(RERANKER)
    names = retrievers()
    fields = ["paper_id", "question_id", "chunking", "retriever", "n_chunks", "findable", "first_hit"]
    new_file = not path.exists()

    for index, (paper_id, paper) in enumerate(papers.items()):
        if args.shard is not None and index % args.shards != args.shard:
            continue
        pdf = data_dir / "pdfs" / f"{paper_id}.pdf"
        if paper_id in done or not pdf.exists():
            continue
        start = time.time()
        try:
            pages = PyPDFLoader(str(pdf)).load()
        except Exception as e:  # noqa: BLE001 - such papers are left out, as in the baseline
            print(paper_id, "failed:", e, flush=True)
            continue
        questions = list(paper["qas"])
        texts_q = [q["question"] for q in questions]
        evidence = [[_ngrams(_tokens(e)) for e in sorted({e for r in references(q) for e in r["evidence"]})]
                    for q in questions]
        rows = []
        for chunking, (size, overlap) in CHUNKINGS.items():
            splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap, length_function=len)
            texts = [c.page_content for c in splitter.split_documents(pages)]
            if not texts:
                continue
            grams = [_ngrams(_tokens(t)) for t in texts]
            gold = [{i for i, g in enumerate(grams) if any(matches(g, pg) for pg in ev)} for ev in evidence]

            ranking = {}
            for name, (_, q_prefix, p_prefix) in DENSE.items():
                c_emb = models[name].encode([p_prefix + t for t in texts], normalize_embeddings=True, batch_size=64)
                q_emb = models[name].encode([q_prefix + t for t in texts_q], normalize_embeddings=True)
                scores = q_emb @ c_emb.T
                for qi in range(len(questions)):
                    ranking[(qi, f"dense-{name}")] = np.argsort(-scores[qi], kind="stable")
            bm25 = BM25Okapi([_tokens(t) or ["_"] for t in texts])
            for qi, q in enumerate(texts_q):
                ranking[(qi, "bm25")] = np.argsort(-bm25.get_scores(_tokens(q)), kind="stable")
                for name in DENSE:
                    ranking[(qi, f"hybrid-{name}")] = _rrf(ranking[(qi, f"dense-{name}")], ranking[(qi, "bm25")])

            # Cross-encoder scores, computed once per (question, chunk) and shared by every base
            for qi, q in enumerate(texts_q):
                bases = [n[len("rerank-"):] for n in names if n.startswith("rerank-")]
                pool = sorted({int(i) for b in bases for i in ranking[(qi, b)][:RERANK_DEPTH]})
                ce = dict(zip(pool, reranker.predict([(q, texts[i]) for i in pool], batch_size=64), strict=True))
                for b in bases:
                    top = list(ranking[(qi, b)][:RERANK_DEPTH])
                    top.sort(key=lambda i: -ce[int(i)])
                    ranking[(qi, f"rerank-{b}")] = np.array(top + list(ranking[(qi, b)][RERANK_DEPTH:]))

            for qi, q in enumerate(questions):
                for name in names:
                    order = ranking[(qi, name)]
                    hit = next((r for r, i in enumerate(order, 1) if int(i) in gold[qi]), None)
                    rows.append({
                        "paper_id": paper_id, "question_id": q["question_id"], "chunking": chunking,
                        "retriever": name, "n_chunks": len(texts), "findable": int(bool(gold[qi])),
                        "first_hit": "" if hit is None else hit,
                    })
        with gzip.open(path, "at", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            if new_file:
                writer.writeheader()
                new_file = False
            writer.writerows(rows)
        print(paper_id, len(questions), "questions", f"{time.time() - start:.0f}s", flush=True)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test from the discordant counts.

    Args:
        b (int): Pairs where only the first set-up succeeds.
        c (int): Pairs where only the second succeeds.

    Returns:
        float: The p-value.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def holm(pvalues):
    """Holm-adjust a dict of p-values.

    Args:
        pvalues (dict): Name to raw p-value.

    Returns:
        dict: Name to adjusted p-value.
    """
    order = sorted(pvalues, key=pvalues.get)
    adjusted, running = {}, 0.0
    for i, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - i) * pvalues[name]))
        adjusted[name] = running
    return adjusted


def summarize(args):
    """Hit rates, MRR and paired tests against the app's set-up; writes summary.json and summary.md.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    out = Path(args.out)
    rows = []
    for path in sorted(out.glob("ranks*.csv.gz")):
        with gzip.open(path, "rt", newline="") as f:
            rows += list(csv.DictReader(f))
    by_q = {}
    for r in rows:
        by_q.setdefault(r["question_id"], {})[f"{r['chunking']}/{r['retriever']}"] = r
    configs = [f"{c}/{r}" for c in CHUNKINGS for r in retrievers()]
    # Questions with evidence findable under every chunking, scored under every set-up
    common = [q for q, d in by_q.items() if all(c in d and d[c]["findable"] == "1" for c in configs)]

    def hit(qid, config, k):
        rank = by_q[qid][config]["first_hit"]
        return rank != "" and int(rank) <= k

    def budget_k(config):
        return max(1, round(BUDGET_CHARS / CHUNKINGS[config.split("/")[0]][0]))

    table = {}
    for config in configs:
        entry = {}
        for k in KS:
            p, lo, hi = wilson(sum(hit(q, config, k) for q in common), len(common))
            entry[f"hit@{k}"] = [round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1)]
        kb = budget_k(config)
        p, lo, hi = wilson(sum(hit(q, config, kb) for q in common), len(common))
        entry["budget_k"] = kb
        entry["hit@budget"] = [round(100 * p, 1), round(100 * lo, 1), round(100 * hi, 1)]
        ranks = [by_q[q][config]["first_hit"] for q in common]
        entry["mrr@10"] = round(sum(1 / int(r) for r in ranks if r != "" and int(r) <= 10) / len(common), 3)
        table[config] = entry

    tests = {}
    for metric, kfun in (("hit@4", lambda c: 4), ("hit@budget", budget_k)):
        raw, counts = {}, {}
        for config in configs:
            if config == BASELINE:
                continue
            b = sum(hit(q, config, kfun(config)) and not hit(q, BASELINE, kfun(BASELINE)) for q in common)
            c = sum(hit(q, BASELINE, kfun(BASELINE)) and not hit(q, config, kfun(config)) for q in common)
            raw[config], counts[config] = mcnemar_exact(b, c), (b, c)
        adjusted = holm(raw)
        tests[metric] = {
            c: {"gained": counts[c][0], "lost": counts[c][1], "p": raw[c], "p_holm": adjusted[c]} for c in raw
        }

    summary = {
        "questions_with_evidence_findable_in_all_chunkings": len(common),
        "questions_seen": len(by_q),
        "baseline": BASELINE,
        "budget_chars": BUDGET_CHARS,
        "rerank_depth": RERANK_DEPTH,
        "models": {**{k: v[0] for k, v in DENSE.items()}, "reranker": RERANKER},
        "results": table,
        "paired_vs_baseline": tests,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = [
        f"# Retrieval sweep: {len(common)} questions (evidence findable under every chunking)",
        "",
        (f"Baseline (the app): `{BASELINE}`. Budget = top k chunks with k = {BUDGET_CHARS} / chunk size. "
        "Gained/lost and Holm p: paired exact McNemar against the baseline at hit@4, over all set-ups."),
        "",
        "| Set-up | hit@1 | hit@4 | hit@10 | hit@budget (k) | MRR@10 | hit@4 gained/lost | Holm p |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for config in sorted(configs, key=lambda c: -table[c]["hit@4"][0]):
        e = table[config]
        t = tests["hit@4"].get(config)
        cmp = "baseline" if t is None else f"+{t['gained']} / -{t['lost']}"
        p = "" if t is None else f"{t['p_holm']:.2g}"
        lines.append(
            f"| `{config}` | {e['hit@1'][0]} | {e['hit@4'][0]} [{e['hit@4'][1]}, {e['hit@4'][2]}] | {e['hit@10'][0]} "
            f"| {e['hit@budget'][0]} ({e['budget_k']}) | {e['mrr@10']} | {cmp} | {p} |"
        )
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
    parser.add_argument("--shard", type=int, help="process only papers with index %% shards == shard")
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--threads", type=int, default=16, help="CPU threads for torch (shared server)")
    args = parser.parse_args()
    {"run": run, "summarize": summarize}[args.stage](args)


if __name__ == "__main__":
    main()
