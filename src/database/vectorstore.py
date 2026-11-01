import faiss
import numpy as np
from langchain_core.documents import Document


class LocalVectorStore:
    """
    A local vector store using FAISS for similarity search

    This class:
    1. Stores document chunks and their embeddings, from one or more files
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

    @property
    def sources(self):
        """
        The files the chunks come from, in the order they were added.

        Returns:
            list: Distinct metadata["source"] values (None for chunks without one).
        """
        return list(dict.fromkeys(doc.metadata.get("source") for doc in self.documents))

    @staticmethod
    def _as_documents(documents):
        """
        Check a list of chunks, wrapping plain strings in Document objects.

        Args:
            documents (list): Strings or Document objects.

        Returns:
            list: Document objects.

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
        return wrapped

    def encode(self, documents):
        """
        Embed chunks with the store's model, so the vectors can be cached per file.

        Args:
            documents (list): Strings or Document objects.

        Returns:
            np.ndarray: One float32 row per chunk.
        """
        texts = [doc.page_content for doc in self._as_documents(documents)]
        return np.asarray(self.embedding_model.encode(texts), dtype="float32")

    def add_documents(self, documents, embeddings=None):
        """
        Add documents to the vector store, after any already there

        This method:
        1. Wraps plain strings in Document objects
        2. Creates embeddings for each chunk using the local model, unless given
        3. Adds them to the FAISS index

        Args:
            documents (list): List of LangChain Document objects or strings
            embeddings (np.ndarray): Precomputed embeddings, one row per document

        Raises:
            ValueError: If documents is empty or holds anything else, or the
                embeddings do not match the documents.
        """
        documents = self._as_documents(documents)
        embeddings = self.encode(documents) if embeddings is None else np.asarray(embeddings, dtype="float32")
        if len(embeddings) != len(documents):
            raise ValueError("embeddings must have one row per document")

        # IndexFlatL2 uses L2 (Euclidean) distance; all-MiniLM-L6-v2 returns unit
        # vectors, so this ranks the same as cosine similarity
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])  # 384 for all-MiniLM-L6-v2
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        self.index.add(embeddings)
        self.documents.extend(documents)

    def similarity_search(self, query, k=4, per_source=0, max_total=None):
        """
        Find the most similar chunks to a query

        With several files, per_source makes sure each file is represented: after the
        top k overall, the best chunks of any file not yet among them are added, up to
        per_source per file and max_total in all. A comparison question then sees every
        file, while a question about one file still gets mostly that file's chunks.

        Args:
            query (str): User's question
            k (int): Number of most similar chunks to return
            per_source (int): Chunks to guarantee from each file
            max_total (int): Upper limit on the chunks returned (default: no limit)

        Returns:
            list: Document objects, most similar first, then the added chunks of other
                files in order of similarity; empty if nothing has been added
        """
        if self.index is None or not self.documents:
            return []

        query_embedding = np.asarray(self.embedding_model.encode([query]), dtype="float32")
        _, indices = self.index.search(query_embedding, len(self.documents))
        # FAISS pads with -1 when it finds fewer results than asked for
        ranking = [i for i in indices[0] if 0 <= i < len(self.documents)]

        chosen = ranking[:k]
        if per_source > 0:
            limit = max_total if max_total is not None else len(ranking)
            counts = {}
            for i in chosen:
                source = self.documents[i].metadata.get("source")
                counts[source] = counts.get(source, 0) + 1
            for i in ranking[k:]:
                if len(chosen) >= limit:
                    break
                source = self.documents[i].metadata.get("source")
                if counts.get(source, 0) < per_source:
                    chosen.append(i)
                    counts[source] = counts.get(source, 0) + 1
        return [self.documents[i] for i in chosen]
