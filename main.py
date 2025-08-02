from pathlib import Path
import sys

parent_dir = Path(__file__).resolve(strict=True).parent
sys.path.append(str(parent_dir))

from dotenv import load_dotenv
import os
import streamlit as st
from src.document_processor.processor import DocumentProcessor
from src.utils.utils import Utils

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
    """
    Main Streamlit application with conversation memory
    
    This function:
    1. Sets up the page configuration
    2. Creates the sidebar for API key and model selection
    3. Handles file upload
    4. Orchestrates the entire application flow with conversation context
    """
    # Configure the Streamlit page
    st.set_page_config(
        page_title="Free Document Q&A with Memory", 
        page_icon="🆓",
        layout="centered"
    )
    
    st.title("🆓 Free Document Q&A with Conversation Memory")
    st.write("100% free APIs - Upload a PDF and have a conversation about it!")
    
    # Sidebar with collapsible UI (provided by Streamlit automatically)
    with st.sidebar:
        st.header("🔧 Setup (Free!)")
        st.write("Get your free Groq API key at: https://console.groq.com")

        # API key input
        groq_api_key = st.text_input("Groq API Key", type="password")

        # Model selection dropdown
        model_options = {
            "llama-3.1-8b-instant": "Llama 3.1 8B (Fast & Smart) - 131k context",
            "llama-3.3-70b-versatile": "Llama 3.3 70B (Most Capable) - 131k context",
            "gemma2-9b-it": "Gemma2 9B (Balanced) - 8k context"
        }

        selected_model = st.selectbox(
            "Choose Model:",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )

        st.header("💬 Conversation Settings")
        max_history = st.slider(
            "Max conversation exchanges to remember:",
            min_value=3,
            max_value=20,
            value=10,
            help="Higher values provide more context but use more tokens"
        )
        
    # Show helpful info if no API key
    if not groq_api_key:
        st.warning("⚠️ Get your free Groq API key at https://console.groq.com")
        st.info("💡 No credit card required - just sign up and start building!")
        
        # Show demo info
        st.markdown("""
        ### 🎯 What You'll Build Today
        - **Document Q&A**: Upload any PDF and ask questions
        - **Conversation Memory**: Reference previous answers naturally
        - **100% Free**: No hidden costs or credit cards needed
        - **Lightning Fast**: Groq's inference is typically under 1 second
        - **Privacy First**: Documents processed locally, only relevant chunks sent to API
        """)
        st.stop()
    
    # Initialize clients
    groq_client = Utils.initialize_groq(groq_api_key)
    embedding_model = Utils.load_embedding_model()
    
    # Store selected model and settings in session state
    st.session_state.selected_model = selected_model
    st.session_state.max_history = max_history
    
    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type="pdf",
        help="Upload any PDF document to start asking questions about it"
    )
    
    # Process uploaded file
    if uploaded_file is not None:
        doc_processor.process_document(uploaded_file, groq_client, embedding_model)
    else:
        # Show instructions when no file is uploaded
        st.markdown("""
        ### 🚀 Getting Started
        1. **Get your free Groq API key** at https://console.groq.com
        2. **Enter your API key** in the sidebar
        3. **Upload a PDF document** using the file uploader above
        4. **Start asking questions** - the AI remembers your conversation!
        
        ### 💡 Example Questions to Try
        - "What is this document about?"
        - "Who are the main authors?"
        - "Can you elaborate on that?" (references previous answer)
        - "How does it compare to what we discussed earlier?"
        """)

        # footnote
        st.markdown("""
        ---
        <sub>© 2025 Monstrous. All rights reserved. For more information, visit [our website](https://monstrous.com.ng).</sub>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()