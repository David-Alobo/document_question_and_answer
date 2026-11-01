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
        MockSplitter.return_value.split_documents.return_value = ["Chunk 1", "Chunk 2"]

        result = Utils.load_and_split_pdf(mock_uploaded_file)

        MockSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200, length_function=len)
        MockSplitter.return_value.split_documents.assert_called_once_with(pages)
        mock_remove.assert_called_once()
        assert result == ["Chunk 1", "Chunk 2"]


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
