
from  pathlib import Path

# Get the parent directory of the current file
parent_dir = Path(__file__).resolve().parent

import faiss
import numpy as np

from langchain.schema import Document

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
        self.chunks = []           # Store original text chunks
        self.embeddings = None     # Store embedding vectors
        self.index = None          # FAISS search index
    
    def add_documents(self, documents):
        """
        Add documents to the vector store and create embeddings
        
        This method:
        1. Extracts text content from document objects
        2. Creates embeddings for each chunk using the local model
        3. Builds a FAISS index for fast similarity search
        
        Args:
            documents (list): List of LangChain document objects
        """

        # Ensure that we are dealing with a list of Document objects
        if isinstance(documents, list):
            # If the list contains strings, wrap them in Document objects
            if isinstance(documents[0], str):
                documents = [Document(page_content=doc) for doc in documents]
            elif not isinstance(documents[0], Document):
                raise ValueError("documents must be a list of strings or Document objects")
        
        # Extract text content from LangChain document objects
        self.chunks = [doc.page_content for doc in documents]
        
        # Create embeddings locally (no API calls!)
        embeddings = self.embedding_model.encode(self.chunks)
        self.embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index for fast similarity search
        # IndexFlatL2 uses L2 (Euclidean) distance for similarity
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
            list: List of most similar text chunks
        """
        if self.index is None:
            return []
        
        # Create embedding for the query
        query_embedding = self.embedding_model.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search for similar chunks
        distances, indices = self.index.search(query_embedding, k)
        
        # Return the actual text chunks
        results = []
        for i in indices[0]:
            if i < len(self.chunks):
                results.append(self.chunks[i])
        
        return results