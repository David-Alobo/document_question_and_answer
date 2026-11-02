"""How many of QASPER's section headings the section-aware splitter finds in the PDFs.

For each development paper, the headings found by src/utils/sections.heading in the PDF text
are compared with the section names in QASPER's full text. A QASPER name such as
"Experiments ::: Setup" counts by its last part ("Setup"). Run-in paragraph headings, whose
names end in a full stop ("Datasets."), are left out: the splitter does not treat them as
headings by design. Names match when they are equal after lowercasing and keeping only letters
and digits; the found heading's number ("3.1") is dropped first.

Usage, from the repo root on a machine with the QASPER data and PDFs:
    python eval/qasper/section_coverage.py --data eval/data
Writes eval/results/2026-11-01-qasper-dev-sections/section_coverage.json.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_eval import _load_split, _rate

from src.utils.sections import NUMBERED, heading

OUT = ROOT / "eval" / "results" / "2026-11-01-qasper-dev-sections"


def normalise(name):
    """Reduce a heading to lowercase letters and digits.

    Args:
        name (str): A heading.

    Returns:
        str: The normalised heading.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def found_headings(pdf):
    """The headings the splitter finds in a PDF, without their numbers.

    Args:
        pdf (Path): The paper's PDF.

    Returns:
        set: Normalised headings.
    """
    from langchain_community.document_loaders import PyPDFLoader

    out = set()
    for page in PyPDFLoader(str(pdf)).load():
        for line in page.page_content.splitlines():
            title = heading(line)
            if title:
                m = NUMBERED.match(title)
                out.add(normalise(m["title"] if m else title))
    return out


def main():
    """Count QASPER section names found by the splitter, over all papers with a PDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="eval/data")
    args = parser.parse_args()
    data = Path(args.data)

    names = found = papers = 0
    for pid, paper in _load_split(data, "dev").items():
        pdf = data / "pdfs" / f"{pid}.pdf"
        if not pdf.exists():
            continue
        papers += 1
        headings = found_headings(pdf)
        wanted = {normalise(s["section_name"].split(":::")[-1]) for s in paper["full_text"]
                  if s["section_name"] and not s["section_name"].rstrip().endswith(".")}
        wanted.discard("")
        names += len(wanted)
        found += len(wanted & headings)
    out = {"papers": papers, "section_names_found": _rate(found, names)}
    (OUT / "section_coverage.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
