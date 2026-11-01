import sys
from pathlib import Path

parent_dir = Path(__file__).resolve(strict=True).parent
sys.path.append(str(parent_dir))

import os

import streamlit as st
from dotenv import load_dotenv

from src.document_processor.processor import FOOTNOTE, DocumentProcessor
from src.utils.utils import Utils

# Load environment variables; GROQ_API_KEY in .env is used if the sidebar field is empty
load_dotenv()

doc_processor = DocumentProcessor()

MODEL_OPTIONS = {
    "llama-3.1-8b-instant": "Llama 3.1 8B (Fast) - 131k context",
    "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Capable) - 131k context",
    "openai/gpt-oss-20b": "GPT-OSS 20B (Balanced) - 131k context",
}


@st.cache_resource
def load_embedding_model():
    """
    Load the embedding model once and share it across sessions.

    Returns:
        SentenceTransformer: Loaded embedding model
    """
    return Utils.load_embedding_model()


def main():
    """
    Main Streamlit application with conversation memory

    This function:
    1. Sets up the page configuration
    2. Creates the sidebar for API key and model selection
    3. Handles file upload
    4. Orchestrates the entire application flow with conversation context
    """
    st.set_page_config(
        page_title="Document Q&A with Memory",
        page_icon="📄",
        layout="centered"
    )

    st.title("📄 Document Q&A with Conversation Memory")
    st.write("Upload one or more PDFs and have a conversation about them. Every answer shows the passages it used.")

    with st.sidebar:
        st.header("🔧 Setup")
        st.write("Get a free Groq API key at: https://console.groq.com")

        groq_api_key = st.text_input("Groq API Key", type="password") or os.getenv("GROQ_API_KEY")

        selected_model = st.selectbox(
            "Choose Model:",
            options=list(MODEL_OPTIONS.keys()),
            format_func=lambda x: MODEL_OPTIONS[x],
            index=0
        )

        st.header("💬 Conversation Settings")
        max_history = st.slider(
            "Max conversation exchanges to remember:",
            min_value=0,
            max_value=20,
            value=5,
            help="Higher values provide more context but use more tokens"
        )

    if not groq_api_key:
        st.warning("⚠️ Enter a Groq API key in the sidebar, or set GROQ_API_KEY in a .env file.")
        st.markdown("""
        ### What it does
        - **Document Q&A**: Upload a PDF and ask questions about it
        - **Several documents**: Upload up to 10 PDFs and ask across them, or compare them
        - **Sources**: Every answer lists the passages, files and pages it was based on
        - **Conversation Memory**: Follow-up questions can refer to earlier answers
        - **Privacy**: Each PDF is split and embedded on this machine; only the passages retrieved for each question, and the conversation, are sent to Groq
        """)
        st.stop()

    groq_client = Utils.initialize_groq(groq_api_key)
    embedding_model = load_embedding_model()

    st.session_state.selected_model = selected_model
    st.session_state.max_history = max_history

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload one or more PDF documents (up to 10) to start asking questions about them"
    )

    if uploaded_files:
        doc_processor.process_documents(uploaded_files, groq_client, embedding_model)
    else:
        st.markdown("""
        ### 🚀 Getting Started
        1. **Upload one or more PDFs** using the file uploader above
        2. **Ask a question** or pick one of the examples
        3. **Check the sources** under each answer
        4. **Ask follow-ups**: the assistant remembers the conversation

        ### 💡 Example Questions to Try
        - "What is this document about?"
        - "Who are the main authors?"
        - "Can you elaborate on that?" (references previous answer)
        - "Compare these documents" (with several PDFs)
        """)
        st.markdown(FOOTNOTE, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
