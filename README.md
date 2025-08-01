# Document - Question and Answer App

This is a Streamlit-based web application designed to assist researchers in analyzing and querying documents. The app leverages LangChain, Groq, and other advanced tools to provide intelligent insights and answers to user queries. It is intended for research purposes only.

## Features
- Document Upload: Upload PDF documents for analysis.
- Document Splitting: Automatically splits documents into manageable chunks for processing.
- Embedding and Vector Search: Uses embeddings to create a vector store for efficient similarity searches.
- Question Answering: Ask questions about the uploaded documents and get concise, context-aware answers.
- Powered by Groq and LangChain: Combines the power of Groq's API and LangChain for advanced natural language processing.

## Technologies Used
- Streamlit: For building the interactive web interface.
- LangChain: For managing and chaining language model tasks.
- Groq: For high-performance AI model inference.
- FAISS: For efficient similarity search and clustering.
- Sentence Transformers: For generating embeddings from text.

## Installation

1. Clone the repository:
```
git clone https://github.com/David-Alobo/document_question_and_answer.git
cd document_question_and_answer
```

2. Install uv (if not already installed):
```
pip install uv
```

3. Create a virtual environment and install dependencies using uv:
```
uv new env
uv pip install -r requirements.txt
```

4. Set up environment variables:

- Create a .env file in the root directory.
- Add your Groq API key
```
GROQ_API_KEY=your_groq_api_key
```

# Usage
1. Activate the virtual environment:
```
uv sunc
```
2. Run the streamlit app
```
uv run streamlit run main.py
```
3. Open the app in your browser.

4. Upload a PDF document, configure the settings, and start asking questions about the document.

## Workflow
- Upload a research paper or document in PDF format.
- The app processes the document, splits it into chunks, and creates a vector store.
- Ask questions about the document, and the app provides concise, context-aware answers.

## Limitations
- This app is intended for research purposes only.
- The accuracy of answers depends on the quality of the document and the underlying language model.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your changes.

Acknowledgments
- Streamlit
- LangChain
- Groq
- FAISS
- Sentence Transformers
- UV