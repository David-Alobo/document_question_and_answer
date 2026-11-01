"""Split a paper's text along its sections instead of at fixed lengths.

PDF text keeps no markup, so section headings are found from the lines themselves: numbered
headings ("3.1 Results", "IV. EXPERIMENTS", "B. Data") and a list of the usual unnumbered ones
("Introduction", "Related Work"). Chunks never cross a heading, and each chunk starts with the
title of the section it comes from, so a passage keeps its context when it is retrieved alone.
"""

import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Usual names of unnumbered sections, lower case
KNOWN_SECTIONS = {
    "abstract", "introduction", "related work", "related works", "background", "method", "methods",
    "methodology", "approach", "model", "models", "data", "dataset", "datasets", "experiments",
    "experiment", "experimental setup", "experimental results", "setup", "results", "evaluation",
    "analysis", "error analysis", "discussion", "conclusion", "conclusions", "conclusion and future work",
    "conclusions and future work", "future work", "limitations", "acknowledgments", "acknowledgements",
    "acknowledgment", "acknowledgement", "references", "appendix", "appendices",
}
TRAILING_WORDS = {"a", "an", "the", "of", "for", "to", "in", "on", "with", "and", "or", "by", "is", "are", "we"}
NUMBERED = re.compile(r"^(?P<num>\d{1,2}(?:\.\d{1,2}){0,2}\.?|[IVX]{1,5}\.|[A-H]\.)\s+(?P<title>\S.*)$")


def _unspace(line):
    """Undo letter-spacing such as "I N T RO D U CT IO N" in a line of capitals.

    Args:
        line (str): One line of PDF text.

    Returns:
        str: The line with spaced-out capitals joined into words where they were split.
    """
    tokens = line.split()
    if len(tokens) >= 4 and sum(len(t) <= 2 for t in tokens) >= 0.6 * len(tokens) and line.upper() == line:
        return "".join(tokens)
    return line


def heading(line):
    """The section title a line announces, if it is a heading.

    Args:
        line (str): One line of PDF text.

    Returns:
        str or None: The heading as written (number included), or None.
    """
    s = " ".join(line.split())
    if not 3 <= len(s) <= 80 or s[-1] in ".,;:" or "%" in s:
        return None
    if s.lower() in KNOWN_SECTIONS:
        return s
    m = NUMBERED.match(s)
    if not m:
        return None
    title = _unspace(m["title"])
    words = title.split()
    digits = sum(c.isdigit() for c in title)
    if not words or not title[0].isupper() or len(words) > 10 or digits > 0.2 * len(title):
        return None
    if not any(len(w) >= 4 and w.isalpha() for w in re.split(r"[\s\-]+", title)):
        return None
    if words[-1].lower() in TRAILING_WORDS:
        # A numbered list item broken across lines, not a heading
        return None
    return f"{m['num']} {title}"


def split_sections(pages, chunk_size, chunk_overlap):
    """Split page documents into chunks that stay inside one section and name it.

    Args:
        pages (list): LangChain Documents, one per page, with metadata["page"].
        chunk_size (int): Maximum characters per chunk, before the section title is added.
        chunk_overlap (int): Characters shared by consecutive chunks of one section.

    Returns:
        list: LangChain Documents. Each starts with "Section: <title>" and keeps the
            metadata of the page where its text starts, plus metadata["section"].
    """
    sections, title, lines = [], None, []  # lines: (page index, text)
    for i, page in enumerate(pages):
        for line in page.page_content.splitlines():
            found = heading(line)
            if found:
                if lines:
                    sections.append((title, lines))
                title, lines = found, []
            else:
                lines.append((i, line))
    if lines:
        sections.append((title, lines))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len, add_start_index=True
    )
    chunks = []
    for title, body in sections:
        text, starts = "", []  # starts: (offset, page index)
        for i, line in body:
            starts.append((len(text), i))
            text += line + "\n"
        for piece in splitter.create_documents([text]):
            offset = piece.metadata["start_index"]
            page_index = next(p for o, p in reversed(starts) if o <= offset) if starts else 0
            metadata = dict(pages[page_index].metadata)
            metadata["section"] = title
            prefix = f"Section: {title}\n" if title else ""
            chunks.append(Document(page_content=prefix + piece.page_content, metadata=metadata))
    return chunks
