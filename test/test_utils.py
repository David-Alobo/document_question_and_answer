import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import sys

# Get the parent directory of the current file
parent_dir = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(parent_dir))

from src.utils.utils import Utils

@pytest.fixture
def mock_groq_client():
    """
    Mock the Groq API client.
    """
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = {"response": "Mocked response"}
    return mock_client

@pytest.fixture
def mock_uploaded_file():
    """
    Mock an uploaded file.
    """
    mock_file = MagicMock()
    mock_file.read.return_value = b"Mock PDF content"
    return mock_file

@pytest.fixture
def mock_documents():
    """
    Mock documents returned by the PDF loader.
    """
    return [{"page_content": "Page 1 content"}, {"page_content": "Page 2 content"}]

def test_initialize_groq():
    """
    Test the initialize_groq method.
    """
    with patch("src.utils.utils.Groq") as MockGroq:
        mock_instance = MockGroq.return_value
        result = Utils.initialize_groq("mock_api_key")
        MockGroq.assert_called_once_with(api_key="mock_api_key")
        assert result == mock_instance

def test_load_and_split_pdf(mock_uploaded_file, mock_documents):
    """
    Test the load_and_split_pdf method.
    """
    with patch("src.utils.utils.PyPDFLoader") as MockLoader, \
         patch("src.utils.utils.RecursiveCharacterTextSplitter") as MockSplitter, \
         patch("os.remove") as mock_remove:
        
        # Mock the PDF loader
        mock_loader_instance = MockLoader.return_value
        mock_loader_instance.load.return_value = mock_documents

        # Mock the text splitter
        mock_splitter_instance = MockSplitter.return_value
        mock_splitter_instance.split_documents.return_value = ["Chunk 1", "Chunk 2"]

        # Call the method
        result = Utils.load_and_split_pdf(mock_uploaded_file)

        # Assertions
        MockLoader.assert_called_once()
        MockSplitter.assert_called_once_with(chunk_size=1000, chunk_overlap=200, length_function=len)
        mock_loader_instance.load.assert_called_once()
        mock_splitter_instance.split_documents.assert_called_once_with(mock_documents)
        mock_remove.assert_called_once()  
        assert result == ["Chunk 1", "Chunk 2"]

def test_get_groq_response(mock_groq_client):
    """
    Test the get_groq_response method.
    """
    context = "This is the context."
    question = "What is the question?"
    query = "Mock query"
    model_name = "llama-31-8b-instant"

    result = Utils.get_groq_response(mock_groq_client, context, question, query, model_name)

    # Assertions
    mock_groq_client.chat.completions.create.assert_called_once_with(
        query=query,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""
        Based on the following context, answer the following questions in a concise manner.
        Context: {context}
        Question: {question}
        Answer: Provide a clear, accurate answer based on the provided context. If the answer is not in the context, say "I don't know". 
        """}
        ],
        prompt=f"""
        Based on the following context, answer the following questions in a concise manner.
        Context: {context}
        Question: {question}
        Answer: Provide a clear, accurate answer based on the provided context. If the answer is not in the context, say "I don't know". 
        """,
        model_name=model_name,
        temperature=0.0,
        max_tokens=500,
    )
    assert result == {"response": "Mocked response"}

def test_load_embedding_model():
    """
    Test the load_embedding_model method.
    """
    with patch("src.utils.utils.SentenceTransformerEmbeddings") as MockEmbeddings:
        mock_instance = MockEmbeddings.return_value
        result = Utils.load_embedding_model()
        MockEmbeddings.assert_called_once_with(model_name="all-MiniLM-L6-v2")
        assert result == mock_instance