from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.utils.utils import SYSTEM_PROMPT, Utils


@pytest.fixture
def mock_groq_client():
    """
    Mock the Groq API client, returning one fixed answer.
    """
    mock_client = MagicMock()
    message = SimpleNamespace(content="Mocked response")
    mock_client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=message)]
    )
    return mock_client


@pytest.fixture
def mock_uploaded_file():
    """
    Mock an uploaded file.
    """
    mock_file = MagicMock()
    mock_file.getvalue.return_value = b"Mock PDF content"
    mock_file.name = "paper.pdf"
    return mock_file


def test_initialize_groq():
    """
    The Groq client is created with the given key.
    """
    with patch("src.utils.utils.Groq") as MockGroq:
        result = Utils.initialize_groq("mock_api_key")
        MockGroq.assert_called_once_with(api_key="mock_api_key")
        assert result == MockGroq.return_value


def test_load_and_split_pdf(mock_uploaded_file):
    """
    The upload is written to a temporary file, split, and the file removed.
    """
    pages = [Document(page_content="Page 1 content", metadata={"page": 0})]
    with patch("src.utils.utils.PyPDFLoader") as MockLoader, \
         patch("src.utils.utils.RecursiveCharacterTextSplitter") as MockSplitter, \
         patch("src.utils.utils.os.remove") as mock_remove:
        MockLoader.return_value.load.return_value = pages
        chunks = [Document(page_content="Chunk 1", metadata={"source": "/tmp/x.pdf", "page": 0})]
        MockSplitter.return_value.split_documents.return_value = chunks

        result = Utils.load_and_split_pdf(mock_uploaded_file)

        MockSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200, length_function=len)
        MockSplitter.return_value.split_documents.assert_called_once_with(pages)
        mock_remove.assert_called_once()
        assert result == chunks
        # The source is the uploaded file's name, not the temporary path
        assert result[0].metadata == {"source": "paper.pdf", "page": 0}


def test_load_and_split_pdf_raises_on_unreadable_file(mock_uploaded_file):
    """
    An unreadable PDF raises ValueError (it used to return the error text as if it were chunks).
    """
    with patch("src.utils.utils.PyPDFLoader") as MockLoader, \
         patch("src.utils.utils.os.remove") as mock_remove:
        MockLoader.return_value.load.side_effect = RuntimeError("EOF marker not found")
        with pytest.raises(ValueError, match="EOF marker not found"):
            Utils.load_and_split_pdf(mock_uploaded_file)
        mock_remove.assert_called_once()


def test_build_messages_without_history_sends_context_and_question():
    """
    The first question carries the document context (it used to be sent with neither).
    """
    messages = Utils.build_messages("CONTEXT", "QUESTION?", [])
    assert [m["role"] for m in messages] == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "CONTEXT" in messages[1]["content"]
    assert "QUESTION?" in messages[1]["content"]


def test_build_messages_with_history_sends_question_once():
    """
    History comes before the current question, which appears exactly once.
    """
    history = [("q1", "a1"), ("q2", "a2")]
    messages = Utils.build_messages("CONTEXT", "QUESTION?", history)
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert [m["content"] for m in messages[1:5]] == ["q1", "a1", "q2", "a2"]
    assert sum("QUESTION?" in m["content"] for m in messages) == 1


def test_get_groq_response(mock_groq_client):
    """
    The client is called with the built messages and the answer text is returned.
    """
    result = Utils.get_groq_response(mock_groq_client, "CONTEXT", "QUESTION?", [], "llama-3.1-8b-instant")

    mock_groq_client.chat.completions.create.assert_called_once_with(
        messages=Utils.build_messages("CONTEXT", "QUESTION?", []),
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_tokens=1000,
    )
    assert result == "Mocked response"


def test_get_groq_response_propagates_errors(mock_groq_client):
    """
    API errors reach the caller, which shows rate-limit and context-length messages.
    """
    mock_groq_client.chat.completions.create.side_effect = RuntimeError("rate limit exceeded")
    with pytest.raises(RuntimeError, match="rate limit"):
        Utils.get_groq_response(mock_groq_client, "c", "q", [])


def test_format_context_labels_pages():
    """
    Each passage is labelled with its 1-based page.
    """
    docs = [
        Document(page_content="alpha", metadata={"page": 0}),
        Document(page_content="beta", metadata={}),
    ]
    assert Utils.format_context(docs) == "[p. 1] alpha\n\n[p. ?] beta"


def test_format_context_names_files_when_there_are_several():
    """
    With chunks from more than one file, each label names its file.
    """
    docs = [
        Document(page_content="alpha", metadata={"source": "a.pdf", "page": 0}),
        Document(page_content="beta", metadata={"source": "b.pdf", "page": 4}),
    ]
    assert Utils.format_context(docs) == "[a.pdf, p. 1] alpha\n\n[b.pdf, p. 5] beta"
    one_file = [Document(page_content="alpha", metadata={"source": "a.pdf", "page": 0})]
    assert Utils.format_context(one_file) == "[p. 1] alpha"


def test_source_label():
    """
    A chunk is described by its file and 1-based page.
    """
    doc = Document(page_content="x", metadata={"source": "a.pdf", "page": 2})
    assert Utils.source_label(doc) == "a.pdf, p. 3"
    assert Utils.source_label(doc, with_file=False) == "p. 3"
    assert Utils.source_label(Document(page_content="x")) == "p. ?"


def test_load_embedding_model():
    """
    The configured sentence-transformers model is loaded.
    """
    with patch("sentence_transformers.SentenceTransformer") as MockModel:
        result = Utils.load_embedding_model()
        MockModel.assert_called_once_with("all-MiniLM-L6-v2")
        assert result == MockModel.return_value


def test_manage_conversation_context_keeps_the_last_exchanges():
    """
    The history is trimmed to a list of the last exchanges (it used to return one tuple).
    """
    history = [(f"q{i}", f"a{i}") for i in range(12)]
    assert Utils.manage_conversation_context(history, max_exchanges=10) == history[-10:]
    assert Utils.manage_conversation_context(history[:3], max_exchanges=10) == history[:3]
    assert Utils.manage_conversation_context(history, max_exchanges=0) == []


def test_heading_detection():
    """Numbered, Roman, letter-spaced and named headings are found; sentences are not."""
    from src.utils.sections import heading

    assert heading("3.1 Results on MultiUN") == "3.1 Results on MultiUN"
    assert heading("IV. EXPERIMENTS") == "IV. EXPERIMENTS"
    assert heading("I. I N T RO D U CT IO N") == "I. INTRODUCTION"
    assert heading("Related Work") == "Related Work"
    assert heading("We train the model for 10 epochs.") is None
    assert heading("1. We first extract one-hot encoding vectors for") is None
    assert heading("2 1M 2,000 2,000") is None


def test_split_sections_keeps_sections_and_pages():
    """Chunks stay inside a section, start with its title and keep the page they start on."""
    from langchain_core.documents import Document

    from src.utils.sections import split_sections

    pages = [
        Document(page_content="Title\nAbstract\nWe study x.\n1 Introduction\n" + "intro text. " * 20,
                 metadata={"page": 0}),
        Document(page_content="2 Method\n" + "method text. " * 20, metadata={"page": 1}),
    ]
    chunks = split_sections(pages, chunk_size=120, chunk_overlap=20)
    sections = {c.metadata["section"] for c in chunks}
    assert {"Abstract", "1 Introduction", "2 Method"} <= sections
    for c in chunks:
        assert c.page_content.startswith(f"Section: {c.metadata['section']}\n") or c.metadata["section"] is None
        assert ("method" in c.page_content) == (c.metadata["section"] == "2 Method")
    assert all(c.metadata["page"] == 1 for c in chunks if c.metadata["section"] == "2 Method")
    assert all(c.metadata["page"] == 0 for c in chunks if c.metadata["section"] == "1 Introduction")


def test_cited_pages_reads_common_citation_forms():
    """
    Single pages, ranges, lists and file names are all read; implausible ranges are not.
    """
    answer = (
        "It uses BERT [p. 3]. Results improve [pp. 5-6] (p. 3, p. 8). "
        "Both agree [a.pdf, p. 2; b.pdf, p. 7] [p. 4, 9]. Overall [pp. 1-40]."
    )
    assert Utils.cited_pages(answer) == [
        (None, 3), (None, 5), (None, 6), (None, 8), ("a.pdf", 2), ("b.pdf", 7), (None, 4), (None, 9)
    ]
    assert Utils.cited_pages("No citations here, see Table 2 [1].") == []


def test_match_citations_pairs_pages_with_passages():
    """
    Cited pages are matched to the passages from them; pages no passage came from are flagged.
    """
    sources = [
        Document(page_content="alpha", metadata={"source": "a.pdf", "page": 2}),
        Document(page_content="beta", metadata={"source": "b.pdf", "page": 2}),
        Document(page_content="gamma", metadata={"source": "a.pdf", "page": 6}),
    ]
    cited, unmatched = Utils.match_citations("X [b.pdf, p. 3]. Y [p. 7] [p. 3]. Z [p. 12].", sources)
    assert [d.page_content for d in cited] == ["beta", "gamma", "alpha"]
    assert unmatched == ["p. 12"]
    assert Utils.match_citations("No citations.", sources) == ([], [])


def test_cited_pages_reads_page_references_in_prose():
    """
    The model often cites in the text itself; those references count, in order of position.
    """
    answer = "Table III on page 5 lists them [p. 2]; see also pages 7 and 9, and 1912.01214.pdf, page 4."
    assert Utils.cited_pages(answer) == [(None, 5), (None, 2), (None, 7), (None, 9), ("1912.01214.pdf", 4)]


def test_cited_files_finds_files_named_without_pages():
    """
    "[a.pdf]" names a file but no page.
    """
    assert Utils.cited_files("In [a.pdf] and [b.pdf], unlike [a.pdf, p. 3] or [a.pdf].") == ["a.pdf", "b.pdf"]
