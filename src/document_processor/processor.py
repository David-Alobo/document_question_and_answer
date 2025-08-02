from pathlib import Path
import sys

parent_dir = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(parent_dir))


import streamlit as st
from utils.utils import Utils
from database.vectorstore import LocalVectorStore

class DocumentProcessor:
    def __init__(self):
        pass

    def process_documents(self, uploaded_file, groq_client, embedding_model):
        st.write("Processing documents...")

        # Process the uploaded file
        with st.spinner("Processing documents..."):
            documents = Utils.load_and_split_pdf(uploaded_file)

        # Check if documents were successfully processed
        if not documents:
            st.error("Failed to read or split the PDF document.")
            return

        st.success(f"Loaded {len(documents)} chunks from the documents")

        # Create the vector store
        with st.spinner("Creating vector store..."):
            vector_store = LocalVectorStore(embedding_model)
            vector_store.add_document(documents)

        st.success("Vector store created successfully!")

        # Save the vector store and Groq client in session state
        st.session_state.vector_store = vector_store
        st.session_state.groq_client = groq_client
        st.session_state.ready = True

        # If ready, display the question-asking interface
        if st.session_state.get("ready", False):
            st.header("Ask Your Questions")
            st.write("**Try Asking:**")

            col1, col2 = st.columns(2)

            with col1:
                if st.button("Ask"):
                    st.session_state.question = st.text_input("Enter your question here:")