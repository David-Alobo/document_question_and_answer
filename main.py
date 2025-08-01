from pathlib import Path
import sys

parent_dir = Path(__file__).resolve(strict=True).parent
sys.path.append(str(parent_dir))

from dotenv import load_dotenv
import os
import streamlit as st
from src.app.app import DocumentProcessor

# Load environment variables
load_dotenv()

# Get the parent directory of the current file
parent_dir = Path(__file__).resolve().parent

# Initialize the DocumentProcessor
groq_api_key = os.getenv("GROQ_API_KEY")  # load key from environment
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is not set in the environment variables.")
doc_processor = DocumentProcessor()

def main():
    st.set_page_config(page_title="Document Q&A App", page_icon=":books:", layout="centered", initial_sidebar_state="collapsed")
    st.header("Document Question & Answer App")
    st.title("Ask Questions About Your Documents")
    st.write("Upload a PDF document and ask questions about its content.")

    # Sidebar for settings
    st.sidebar.header("Settings")
    st.sidebar.write("Configure your API settings and model preferences. Get your free API key at groq.com")
    groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Get your free API key at groq.com")

    # Model selection dropdown
    model_options = {
        "Llama-2-7b-chat": "llama-2-7b-chat",
        "Llama-2-13b-chat": "llama-2-13b-chat",
        "Llama-2-70b-chat": "llama-2-70b-chat",
        "Llama 3,1-8b (Fast & Smart)": "llama-31-8b-fast",
        "Llama 3,1-8b (Slow & Dumb)": "llama-31-8b-instant",
        "Llama 3,1-8b (Fast & Dumb)": "llama-31-8b-fast-instant",
        "Llama 3,1-8b (Slow & Smart)": "llama-31-8b-instant",
    }
    selected_model = st.sidebar.selectbox("Select Model", list(model_options.keys()))

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file and groq_api_key:
        st.write("Processing your document...")
        embedding_model = model_options[selected_model]
        doc_processor.process_documents(uploaded_file, groq_api_key, embedding_model)
    elif not groq_api_key:
        st.error("Please provide a valid Groq API Key.")

    # footnote
    st.markdown("""
    ---
    <sub>© 2025 Monstrous. All rights reserved. For more information, visit [our website](https://monstrous.com.ng).</sub>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()