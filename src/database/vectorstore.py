
from  pathlib import Path

# Get the parent directory of the current file
parent_dir = Path(__file__).resolve().parent

import faiss
import numpy as np

class LocalVectorStore:
    """
    A wrapper class for the FAISS vector store.
    """
    def __init__(self, embedding_model):
      self.embedding_model = embedding_model
      self.chunks = []
      self.embeddings = None
      self.index = None
      self.results = []

    def add_document(self, documents):
      """
      Add a document to the vector store.

      Args:
          documents: The documents to add.

      Returns:
          The response from the Groq API.

      """

      self.chunks = [doc.page_content for doc in documents]
      embeddings = self.embedding_model.encode(self.chunks)
      self.embeddings = np.array(embeddings).astype("float32")
      self.index = faiss.IndexFlatL2(embeddings.shape[1])
      self.index.add(self.embeddings)

    def similarity_search(self, query, k=4):
      """
      Search the vector store for similar documents.
      """
      if self.index is None:
        raise ValueError("Vector store is empty")
        return []

      query_embedding = self.embedding_model.encode([query])
      query_embedding = np.array(query_embedding).astype("float32")
      distances, indices = self.index.search(query_embedding, k)
     
      for i in indices[0]:
        if i < len(self.chunks):
          self.results.append(self.chunks[i])
      return self.results