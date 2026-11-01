from typing import ClassVar

import numpy as np
import pytest
from langchain_core.documents import Document

from src.database.vectorstore import LocalVectorStore


class MockEmbeddingModel:
    """
    A mock embedding model: each text maps to a fixed unit vector by its first word.
    """

    VECTORS: ClassVar[dict] = {
        "cats": [1.0, 0.0, 0.0],
        "dogs": [0.0, 1.0, 0.0],
        "fish": [0.0, 0.0, 1.0],
    }

    def encode(self, texts):
        return np.array([self.VECTORS[t.split()[0].lower()] for t in texts])


@pytest.fixture
def mock_embedding_model():
    return MockEmbeddingModel()


@pytest.fixture
def mock_documents():
    return [
        Document(page_content="Cats purr", metadata={"page": 0}),
        Document(page_content="Dogs bark", metadata={"page": 1}),
        Document(page_content="Fish swim", metadata={"page": 2}),
    ]


def test_add_documents(mock_embedding_model, mock_documents):
    vector_store = LocalVectorStore(mock_embedding_model)
    vector_store.add_documents(mock_documents)

    assert vector_store.chunks == ["Cats purr", "Dogs bark", "Fish swim"]
    assert vector_store.embeddings.shape == (3, 3)
    assert vector_store.index.ntotal == 3


def test_add_documents_wraps_strings(mock_embedding_model):
    vector_store = LocalVectorStore(mock_embedding_model)
    vector_store.add_documents(["Cats purr", "Dogs bark"])
    assert all(isinstance(doc, Document) for doc in vector_store.documents)


@pytest.mark.parametrize("bad", [[], "Cats purr", [1, 2]])
def test_add_documents_rejects_bad_input(mock_embedding_model, bad):
    with pytest.raises(ValueError):
        LocalVectorStore(mock_embedding_model).add_documents(bad)


def test_similarity_search_ranks_and_keeps_metadata(mock_embedding_model, mock_documents):
    vector_store = LocalVectorStore(mock_embedding_model)
    vector_store.add_documents(mock_documents)

    results = vector_store.similarity_search("dogs", k=2)

    assert results[0].page_content == "Dogs bark"
    assert results[0].metadata == {"page": 1}
    assert len(results) == 2


def test_similarity_search_with_k_above_document_count(mock_embedding_model, mock_documents):
    """
    Asking for more chunks than exist returns each chunk once (FAISS pads with -1,
    which used to return the last chunk again).
    """
    vector_store = LocalVectorStore(mock_embedding_model)
    vector_store.add_documents(mock_documents[:2])

    results = vector_store.similarity_search("cats", k=4)

    assert [doc.page_content for doc in results] == ["Cats purr", "Dogs bark"]


def test_similarity_search_empty_index(mock_embedding_model):
    assert LocalVectorStore(mock_embedding_model).similarity_search("Query") == []
