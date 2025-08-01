from pathlib import Path
from dotenv import load_dotenv

# Get the parent directory of the current file
parent_dir = Path(__file__).resolve().parent

import streamlit as st
from app.app import DocumentProcessor

config = load_dotenv()
doc_processor = DocumentProcessor(groq_client=config.get('grop_api', []), embedding_model='')

def main():
  st.set_page_config(page_title="Document Q&A App", page_icon=":books:", layout="wide")
  st.header("Document Q&A App")
  st.title("Ask Questions About Your Documents")
  st.write("Upload a PDF document and ask questions about its content.")

  st.sidebar.header("Settings")
  st.sidebar.write("Configure your API settings and model preferences. get your free api at groq.com")  
  groq_api_key = st.sidebar.text_input("Groq API Key", type="password", help="Get your free api at groq.com")  
  model_options = {
      "Llama-2-7b-chat": "llama-2-7b-chat",
      "Llama-2-13b-chat": "llama-2-13b-chat",
      "Llama-2-70b-chat": "llama-2-70b-chat",
      "Llama 3,1-8b (Fast & Smart)": "llama-31-8b-fast",
      "Llama 3,1-8b (Slow & Dumb)": "llama-31-8b-instant",
      "Llama 3,1-8b (Fast & Dumb)": "llama-31-8b-fast-instant",
      "Llama 3,1-8b (Slow & Smart)": "llama-31-8b-instant",

  } 

if __name__ == "__main__":
    main()
