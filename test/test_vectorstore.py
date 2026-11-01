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


def test_add_documents_appends(mock_embedding_model, mock_documents):
    vector_store = LocalVectorStore(mock_embedding_model)
    vector_store.add_documents(mock_documents[:1])
    vector_store.add_documents(mock_documents[1:])
    assert vector_store.chunks == ["Cats purr", "Dogs bark", "Fish swim"]
    assert vector_store.index.ntotal == 3
    assert vector_store.similarity_search("fish", k=1)[0].page_content == "Fish swim"


def test_add_documents_with_precomputed_embeddings(mock_embedding_model, mock_documents):
    vector_store = LocalVectorStore(mock_embedding_model)
    embeddings = vector_store.encode(mock_documents)
    vector_store.add_documents(mock_documents, embeddings)
    assert vector_store.similarity_search("dogs", k=1)[0].page_content == "Dogs bark"
    with pytest.raises(ValueError, match="one row per document"):
        vector_store.add_documents(mock_documents, embeddings[:2])


class OneAxisModel:
    """
    Embeds a text by its first word along one axis: the query "cats" ranks cats, then dogs, then fish.
    """

    AXIS: ClassVar[dict] = {"cats": 1.0, "dogs": 0.8, "fish": 0.0}

    def encode(self, texts):
        values = [self.AXIS[t.split()[0].lower()] for t in texts]
        return np.array([[v, 1.0 - v] for v in values])


def test_similarity_search_covers_every_file():
    """
    per_source adds the best chunk of each file missing from the top k, up to max_total.
    """
    docs = [
        Document(page_content="Cats one", metadata={"source": "a.pdf"}),
        Document(page_content="Cats two", metadata={"source": "a.pdf"}),
        Document(page_content="Dogs one", metadata={"source": "b.pdf"}),
        Document(page_content="Fish one", metadata={"source": "c.pdf"}),
    ]
    vector_store = LocalVectorStore(OneAxisModel())
    vector_store.add_documents(docs)
    assert vector_store.sources == ["a.pdf", "b.pdf", "c.pdf"]

    top = vector_store.similarity_search("cats", k=2)
    assert [d.metadata["source"] for d in top] == ["a.pdf", "a.pdf"]

    covered = vector_store.similarity_search("cats", k=2, per_source=1)
    assert [d.page_content for d in covered] == ["Cats one", "Cats two", "Dogs one", "Fish one"]

    capped = vector_store.similarity_search("cats", k=2, per_source=1, max_total=3)
    assert [d.page_content for d in capped] == ["Cats one", "Cats two", "Dogs one"]


def test_similarity_search_limited_to_chosen_files():
    """
    Restricting to one file gives exactly that file's ranking, as if it were alone.
    """
    docs = [
        Document(page_content="Cats one", metadata={"source": "a.pdf"}),
        Document(page_content="Dogs one", metadata={"source": "b.pdf"}),
        Document(page_content="Fish one", metadata={"source": "b.pdf"}),
    ]
    mixed = LocalVectorStore(OneAxisModel())
    mixed.add_documents(docs)
    alone = LocalVectorStore(OneAxisModel())
    alone.add_documents(docs[1:])

    chosen = mixed.similarity_search("cats", k=2, sources=["b.pdf"])
    assert chosen == alone.similarity_search("cats", k=2)
    assert [d.page_content for d in chosen] == ["Dogs one", "Fish one"]
    assert mixed.similarity_search("cats", k=2, sources=[]) == []


def test_similarity_search_empty_index(mock_embedding_model):
    assert LocalVectorStore(mock_embedding_model).similarity_search("Query") == []
