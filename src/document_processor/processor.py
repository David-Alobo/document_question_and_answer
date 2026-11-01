import streamlit as st

from src.database.vectorstore import LocalVectorStore
from src.utils.utils import Utils

EXAMPLE_QUESTIONS = [
    ("📋", "What is this document about?"),
    ("👥", "Who are the main authors or people mentioned?"),
    ("🔍", "What are the key findings or conclusions?"),
    ("📊", "Can you elaborate on that?"),
]

FOOTNOTE = """
---
<sub>© 2025 Monstrous. For more information, visit [our website](https://monstrous.com.ng).</sub>
"""


class DocumentProcessor:
    def __init__(self):
        pass

    @staticmethod
    def _file_key(uploaded_file):
        """
        Identify an uploaded file, so it is only embedded once.

        Args:
            uploaded_file: Streamlit uploaded file object

        Returns:
            tuple: The file's name and size in bytes.
        """
        return (uploaded_file.name, uploaded_file.size)

    def _prepare(self, uploaded_file, embedding_model):
        """
        Split and embed a newly uploaded PDF, and start a fresh conversation.

        Streamlit reruns the script on every interaction; the vector store is kept in
        session state so the PDF is embedded once per upload, not once per question.

        Args:
            uploaded_file: Streamlit uploaded file object
            embedding_model: Loaded sentence transformer model

        Returns:
            bool: True if the document is ready for questions.
        """
        key = self._file_key(uploaded_file)
        if st.session_state.get("document_key") == key:
            return True

        with st.spinner("📖 Reading PDF..."):
            try:
                chunks = Utils.load_and_split_pdf(uploaded_file)
            except ValueError as e:
                st.error(f"❌ {e}")
                return False
        if not chunks:
            st.error("❌ Could not extract text from PDF. Scanned PDFs need OCR first.")
            return False

        with st.spinner("🧮 Embedding the document..."):
            vector_store = LocalVectorStore(embedding_model)
            vector_store.add_documents(chunks)

        st.session_state.document_key = key
        st.session_state.vector_store = vector_store
        st.session_state.conversation_history = []
        st.session_state.conversation_sources = []
        return True

    @staticmethod
    def _show_sources(sources):
        """
        Show the passages an answer was based on, with their pages.

        Args:
            sources (list): LangChain Document objects retrieved for the question.
        """
        with st.expander(f"📚 Sources ({len(sources)} passages)"):
            for doc in sources:
                page = doc.metadata.get("page")
                label = f"Page {page + 1}" if isinstance(page, int) else "Page unknown"
                text = doc.page_content
                st.markdown(f"**{label}**")
                st.text(text[:600] + "..." if len(text) > 600 else text)

    def _answer(self, question, groq_client):
        """
        Retrieve passages for a question, ask the model, and record the exchange.

        Args:
            question (str): The user's question.
            groq_client: Initialized Groq API client
        """
        sources = st.session_state.vector_store.similarity_search(question, k=4)
        if not sources:
            st.warning("🤷 No relevant information found. Try rephrasing your question.")
            return

        history = Utils.manage_conversation_context(
            st.session_state.conversation_history,
            max_exchanges=st.session_state.get("max_history", 10),
        )
        try:
            with st.spinner("🤔 Thinking..."):
                answer = Utils.get_groq_response(
                    groq_client,
                    Utils.format_context(sources),
                    question,
                    history,
                    st.session_state.get("selected_model", "llama-3.1-8b-instant"),
                )
        except Exception as e:  # noqa: BLE001 - any API error is shown to the user
            # Handle different types of errors gracefully
            if "rate limit" in str(e).lower():
                st.error("🕐 Rate limit reached. Please wait a moment and try again.")
            elif "context_length" in str(e).lower():
                st.error("📏 Conversation too long. Lower the number of exchanges to remember, or clear the conversation.")
            else:
                st.error(f"❌ Error: {e}")
                st.info("💡 Check your API key and the chosen model.")
            return

        st.session_state.conversation_history.append((question, answer))
        st.session_state.conversation_sources.append(sources)

    def process_document(self, uploaded_file, groq_client, embedding_model):
        """
        Main document processing pipeline with conversation memory

        This function orchestrates the entire RAG pipeline:
        1. Loads, splits and embeds the PDF (once per upload)
        2. Shows the conversation so far, with the sources of each answer
        3. Answers a new question with retrieved passages and the recent conversation

        Args:
            uploaded_file: Streamlit uploaded file object
            groq_client: Initialized Groq API client
            embedding_model: Loaded sentence transformer model
        """
        if not self._prepare(uploaded_file, embedding_model):
            return

        n_chunks = len(st.session_state.vector_store.documents)
        st.success(f"✅ Document ready: {n_chunks} chunks")
        st.header("💬 Ask Your Questions")

        # Example questions to help users get started
        cols = st.columns(2)
        for i, (icon, text) in enumerate(EXAMPLE_QUESTIONS):
            if cols[i % 2].button(f"{icon} {text}", key=f"example_{i}"):
                st.session_state.pending_question = text

        if st.session_state.conversation_history and st.button("🗑️ Clear Conversation History"):
            st.session_state.conversation_history = []
            st.session_state.conversation_sources = []
            st.rerun()

        question = st.chat_input("Ask anything about the document...")
        question = question or st.session_state.pop("pending_question", None)
        if question:
            self._answer(question, groq_client)

        exchanges = zip(st.session_state.conversation_history, st.session_state.conversation_sources)
        for (q, a), sources in exchanges:
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
                self._show_sources(sources)

        st.markdown(FOOTNOTE, unsafe_allow_html=True)
