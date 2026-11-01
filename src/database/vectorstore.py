import faiss
import numpy as np
from langchain_core.documents import Document


class LocalVectorStore:
    """
    A local vector store using FAISS for similarity search

    This class:
    1. Stores document chunks and their embeddings
    2. Creates a FAISS index for fast similarity search
    3. Provides methods to add documents and search for similar content
    """

    def __init__(self, embedding_model):
        """
        Initialize the vector store

        Args:
            embedding_model: SentenceTransformer model for creating embeddings
        """
        self.embedding_model = embedding_model
        self.documents = []        # Store original chunks with their metadata
        self.embeddings = None     # Store embedding vectors
        self.index = None          # FAISS search index

    @property
    def chunks(self):
        """
        The text of every stored chunk.

        Returns:
            list: One string per chunk, in insertion order.
        """
        return [doc.page_content for doc in self.documents]

    def add_documents(self, documents):
        """
        Add documents to the vector store and create embeddings

        This method:
        1. Wraps plain strings in Document objects
        2. Creates embeddings for each chunk using the local model
        3. Builds a FAISS index for fast similarity search

        Args:
            documents (list): List of LangChain Document objects or strings

        Raises:
            ValueError: If documents is empty or holds anything else.
        """
        if not isinstance(documents, list) or not documents:
            raise ValueError("documents must be a non-empty list of strings or Document objects")
        wrapped = []
        for doc in documents:
            if isinstance(doc, str):
                doc = Document(page_content=doc)
            elif not hasattr(doc, "page_content"):
                raise ValueError("documents must be a list of strings or Document objects")
            wrapped.append(doc)
        self.documents = wrapped

        # Create embeddings locally (no API calls!)
        embeddings = self.embedding_model.encode(self.chunks)
        self.embeddings = np.asarray(embeddings, dtype="float32")

        # IndexFlatL2 uses L2 (Euclidean) distance; all-MiniLM-L6-v2 returns unit
        # vectors, so this ranks the same as cosine similarity
        dimension = self.embeddings.shape[1]  # 384 for all-MiniLM-L6-v2
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def similarity_search(self, query, k=4):
        """
        Find the most similar chunks to a query

        Args:
            query (str): User's question
            k (int): Number of similar chunks to return

        Returns:
            list: Up to k Document objects, most similar first; empty if nothing
                has been added
        """
        if self.index is None or not self.documents:
            return []

        query_embedding = np.asarray(self.embedding_model.encode([query]), dtype="float32")
        _, indices = self.index.search(query_embedding, min(k, len(self.documents)))

        # FAISS pads with -1 when it finds fewer than k results
        return [self.documents[i] for i in indices[0] if 0 <= i < len(self.documents)]
