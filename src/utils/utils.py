import streamlit as st
from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from sentence_transformers import SentenceTransformer
# from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import HuggingFaceHub
import faiss
import numpy as np
import tempfile
import os


class Utils:

    def __init__(self):
        pass  

    @staticmethod
    def initialize_groq(api_key: str):
        """
            Initialize the Groq API client.

            Parameters:
            - api_key (str): Your Groq API key.

            Returns:
            - Groq: An initialized Groq client instance.
        """
        return Groq(api_key=api_key)

    @staticmethod
    @st.cache_data
    def load_and_split_pdf(uploaded_file):
        """
        Load a PDF file and return a list of documents.

        Args:
            upload_file: Streamlit uploaded file object

        Returns:
            list: List of text chunks as LangChain document objects
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        try:
            loader = PyPDFLoader(temp_file_path) # using LangChain's PyPDFLoader to load the PDF
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
    def get_groq_response(client, context, question, conversation_history, model_name="llama-3.1-8b-instant"):
        """
        Get response from Groq API using RAG pattern with conversation memory
        
        This function:
        1. Uses system prompts for better conversation awareness
        2. Includes previous Q&A pairs as context
        3. Handles references like "that", "it", "the topic we discussed"
        4. Maintains document grounding while being conversational
        
        Args:
            client (Groq): Initialized Groq client
            context (str): Relevant document chunks as context
            question (str): Current user question
            conversation_history (list): Previous Q&A pairs
            model_name (str): Groq model to use
            
        Returns:
            str: Generated answer with conversation awareness
        """
        
        # Build conversation messages for better context management
        messages = [
                    {
                        "role": "system",
                        "content": """You are a document analysis assistant with conversation memory. Your capabilities:

            1. DOCUMENT GROUNDING: Always base answers on the provided document context
            2. CONVERSATION AWARENESS: Remember and reference previous exchanges when relevant
            3. REFERENCE RESOLUTION: When users say "that", "it", "the topic", understand what they're referring to
            4. CLARITY: If a reference is ambiguous, ask for clarification
            5. ACCURACY: Never make up information not in the document or conversation

            You maintain context across the conversation while staying grounded in the document."""
                    }
                ]
        
        # Add recent conversation history (last 5 exchanges to manage tokens)
        for prev_q, prev_a in conversation_history[-5:]:
            messages.append({"role": "user", "content": prev_q})
            messages.append({"role": "assistant", "content": prev_a})
            
            # Add current question with document context
            current_message = f"""
                Document Context:
                    {context}
                        Current Question: 
                            {question}
                """
            
            messages.append({"role": "user", "content": current_message})
            
        try:
            # Make API call to Groq with conversation context
            response = client.chat.completions.create(
                        messages=messages,
                        model=model_name,  # Using Llama 3.1 8B for speed and quality
                        temperature=0.1,   # Low temperature for factual, consistent answers
                        max_tokens=1000    # Reasonable response length
                    )
            return response.choices[0].message.content
        except Exception as e:
            # Return user-friendly error message
            return f"Error getting response: {str(e)}"
        
    @staticmethod
    @st.cache_resource(key="embedding_model_all-MiniLM-L6-v2", hash_funcs={"sentence_transformers.SentenceTransformer": id})
    def load_embedding_model():
        """
        Load sentence transformer model for creating embeddings locally.

        Uses streamlit's cache_resource decoratore to load the model only once
        and reuse it across sessions for better performance.

        Returns:
            SentenceTransformer: Loaded embedding model
        """
        return SentenceTransformer('all-MiniLM-L6-v2')
    
    @staticmethod
    def manage_conversation_context(conversation_history, max_exchanges=10):
        """
        Manage conversation history to prevent token overflow

        Args:
            Conversation_history (list): List of question, answer) tuples
            max_exchanges (int): Maximum number of exchanges to keep
        
        Returns:
            list: Trimmed conversation history
        """
        if len(conversation_history) > max_exchanges:
            return conversation_history[-max_exchanges]
        return conversation_history