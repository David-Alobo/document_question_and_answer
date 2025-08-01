import pytest
import numpy as np
from unittest.mock import MagicMock
from pathlib import Path
import sys

# Get the parent directory of the current file
parent_dir = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(parent_dir))

from src.database.vectorstore import LocalVectorStore

class MockEmbeddingModel:
    """
    A mock embedding model for testing purposes.
    """
    def encode(self, texts):
        # Return a dummy embedding (e.g., a vector of ones with length 5 for each text)
        return np.ones((len(texts), 5))

class MockDocument:
    """
    A mock document class for testing purposes.
    """
    def __init__(self, page_content):
        self.page_content = page_content

@pytest.fixture
def mock_embedding_model():
    return MockEmbeddingModel()

@pytest.fixture
def mock_documents():
    return [MockDocument("Document 1 content"), MockDocument("Document 2 content")]

def test_add_document(mock_embedding_model, mock_documents):
    vector_store = LocalVectorStore(mock_embedding_model)
    vector_store.add_document(mock_documents)

    # Check if chunks are correctly stored
    assert vector_store.chunks == ["Document 1 content", "Document 2 content"]

    # Check if embeddings are correctly created
    assert vector_store.embeddings.shape == (2, 5)  # 2 documents, embedding size 5

    # Check if FAISS index is created and populated
    assert vector_store.index.ntotal == 2

def test_similarity_search(mock_embedding_model, mock_documents):
    vector_store = LocalVectorStore(mock_embedding_model)
    vector_store.add_document(mock_documents)

    # Perform a similarity search
    results = vector_store.similarity_search("Query", k=2)

    # Check if results are returned correctly
    assert len(results) == 2
    assert "Document 1 content" in results
    assert "Document 2 content" in results

def test_similarity_search_empty_index(mock_embedding_model):
    vector_store = LocalVectorStore(mock_embedding_model)

    # Attempt to search without adding documents
    with pytest.raises(ValueError, match="Vector store is empty"):
        vector_store.similarity_search("Query")