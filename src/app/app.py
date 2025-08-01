from pathlib import Path

# Get the parent directory of the current file
parent_dir = Path(__file__).resolve().parent

import streamlit as st
from utils.utils import Utils


def process_documents(uploaded_file, groq_client, embedding_model):
   st.write("Processing documents...")
   
   with st.spinner("Processing documents..."):
    documents = Utils.load_and_split_pdf(uploaded_file)
    return
   
   if not documents:
    st.error("Failed to read or split the PDF document.")
    
    st.success(f"Loaded {len(documents)} chunks from the documents")
    
    with st.spiiner("Creating vector store..."):
      vector_store = LocalVectorStore(embedding_model)
      vector_store.add_document(documents)
      
    st.success("Vector store created successfully!")
      
    st.session_state.vector_store = vector_store
      
    st.session_state.gro_clinete = groq_client
    st.seession_state.ready = True
      
    if st.session_state.get("ready", False):
      st.header("Ask Your Questions")

    st.write("**Try Asking:**")
    col1, col2 = st.columns(2)

    with col1:
      if st.button(""):
        st.session_state.question