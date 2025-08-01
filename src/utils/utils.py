import streamlit as st
from groq import Groq
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import HuggingFaceHub
import faiss
import numpy as np
import tempfile
import os

class Utils:
    @staticmethod
    def initialize_groq(api_key: str):
        """
        Initialize the Groq API client.
        """
        return Groq(api_key=api_key)

    @staticmethod
    def load_and_split_pdf(uploaded_file):
        """
        Load a PDF file and return a list of documents.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
        try:
            loader = PyPDFLoader(temp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            split_docs = text_splitter.split_documents(documents)
            os.remove(temp_file_path)  # Clean up temporary file
            return split_docs
        except Exception as e:
            os.remove(temp_file_path)
            return str(e)

    @staticmethod
    def get_groq_response(groq_client, context, question, query: str, model_name="llama-31-8b-instant"):
        """
        Get a response from the Groq API.

        Args:
            groq_client: The Groq API client.
            context: The context to use for the query.
            question: The question to ask.
            query: The query to send to the Groq API.
            model_name: The name of the model to use.

        Returns:
            The response from the Groq API.
        """
        prompt = f"""
        Based on the following context, answer the following questions in a concise manner.
        Context: {context}
        Question: {question}
        Answer: Provide a clear, accurate answer based on the provided context. If the answer is not in the context, say "I don't know". 
        """
        try:
            response = groq_client.chat.completions.create(
                query=query,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                prompt=prompt,
                model_name=model_name,
                temperature=0.0,
                max_tokens=500,
            )
            return response
        except Exception as e:
            return str(e)

    @staticmethod
    def load_embedding_model():
        """
        Load the embedding model.
        """
        return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
