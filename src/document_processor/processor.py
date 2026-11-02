import streamlit as st

from src.database.vectorstore import LocalVectorStore
from src.utils.utils import Utils

EXAMPLE_QUESTIONS = [
    ("📋", "What is this document about?"),
    ("👥", "Who are the main authors or people mentioned?"),
    ("🔍", "What are the key findings or conclusions?"),
    ("📊", "Can you elaborate on that?"),
]
COMPARE_QUESTION = ("⚖️", "Compare these documents: how do their aims, methods and findings differ?")

ALL_DOCUMENTS = "All documents"
COMPARE = "Compare all documents"
MAX_DOCUMENTS = 10
TOP_K = 4            # chunks retrieved by similarity alone
MAX_CONTEXT = 8      # with several files: up to this many chunks, at least one per file
COMPARE_EACH = 2     # when comparing: this many chunks from every file

CHECK_NOTE = (
    "Answers can be wrong: on a hand-checked sample of questions about research papers, fewer "
    "than half of Llama 3.1 8B's answers were fully correct. Check each point against the cited "
    "passage before relying on it."
)

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

    def _load(self, uploaded_file, embedding_model):
        """
        Split and embed one PDF.

        Args:
            uploaded_file: Streamlit uploaded file object
            embedding_model: Loaded sentence transformer model

        Returns:
            dict: The file's name and either its chunks and embeddings, or an error.
        """
        try:
            chunks = Utils.load_and_split_pdf(uploaded_file)
        except ValueError as e:
            return {"name": uploaded_file.name, "error": str(e)}
        if not chunks:
            return {"name": uploaded_file.name, "error": "no text could be extracted (scanned PDFs need OCR first)"}
        embeddings = LocalVectorStore(embedding_model).encode(chunks)
        return {"name": uploaded_file.name, "chunks": chunks, "embeddings": embeddings}

    def _prepare(self, uploaded_files, embedding_model):
        """
        Make the vector store match the uploaded files.

        Streamlit reruns the script on every interaction. Each file is split and embedded
        once and cached in session state; when files are added or removed, the store is
        rebuilt from the cache without embedding anything again. The conversation is kept,
        so a follow-up can compare a new file with what was discussed.

        Args:
            uploaded_files (list): Streamlit uploaded file objects
            embedding_model: Loaded sentence transformer model

        Returns:
            list: The files that could not be read, as (name, error) pairs.
        """
        cache = st.session_state.setdefault("file_cache", {})
        keys = [self._file_key(f) for f in uploaded_files]
        new = [(k, f) for k, f in zip(keys, uploaded_files, strict=True) if k not in cache]
        for i, (key, uploaded_file) in enumerate(new, 1):
            with st.spinner(f"📖 Reading and embedding {uploaded_file.name} ({i} of {len(new)})..."):
                cache[key] = self._load(uploaded_file, embedding_model)
        for key in list(cache):
            if key not in keys:
                del cache[key]

        st.session_state.setdefault("conversation_history", [])
        st.session_state.setdefault("conversation_sources", [])
        if st.session_state.get("document_keys") != keys:
            vector_store = LocalVectorStore(embedding_model)
            for key in keys:
                if "chunks" in cache[key]:
                    vector_store.add_documents(cache[key]["chunks"], cache[key]["embeddings"])
            st.session_state.vector_store = vector_store
            st.session_state.document_keys = keys
        return [(cache[k]["name"], cache[k]["error"]) for k in keys if "error" in cache[k]]

    @staticmethod
    def _show_sources(answer, sources, latest=False):
        """
        Show the passages an answer cites, then every passage it was based on.

        The cited passages are shown in full, so each point of the answer can be checked
        against the text it claims to come from. Pages the answer cites but no passage
        came from are flagged: the model never saw them.

        Args:
            answer (str): The model's answer.
            sources (list): LangChain Document objects retrieved for the question.
            latest (bool): Open the cited passages, as for the newest answer.
        """
        several = len({doc.metadata.get("source") for doc in sources}) > 1
        cited, unmatched = Utils.match_citations(answer, sources)
        if cited:
            with st.expander(f"🔎 Check the cited passages ({len(cited)})", expanded=latest):
                for doc in cited:
                    label = Utils.source_label(doc, several).replace("p. ", "page ")
                    st.markdown(f"**{label[0].upper()}{label[1:]}**")
                    st.text(doc.page_content)
        else:
            st.caption("This answer cites no pages; check it against the passages below.")
        if unmatched:
            st.warning(f"⚠️ Cited but not among the passages the model read: {', '.join(unmatched)}. "
                       "Treat those points with caution.")
        with st.expander(f"📚 All passages used ({len(sources)})"):
            for doc in sources:
                text = doc.page_content
                label = Utils.source_label(doc, several).replace("p. ", "page ")
                st.markdown(f"**{label[0].upper()}{label[1:]}**")
                st.text(text[:600] + "..." if len(text) > 600 else text)

    def _answer(self, question, groq_client, scope=None):
        """
        Retrieve passages for a question, ask the model, and record the exchange.

        Args:
            question (str): The user's question.
            groq_client: Initialized Groq API client
            scope (str): A file name to search only that file, COMPARE to take the same
                number of passages from every file, or None for all files.
        """
        store = st.session_state.vector_store
        if scope == COMPARE:
            sources = store.similarity_search(question, each=COMPARE_EACH)
        elif scope is not None:
            sources = store.similarity_search(question, k=TOP_K, sources=[scope])
        else:
            several = len(store.sources) > 1
            sources = store.similarity_search(
                question, k=TOP_K, per_source=1 if several else 0, max_total=MAX_CONTEXT if several else None
            )
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

    def process_documents(self, uploaded_files, groq_client, embedding_model):
        """
        Main document processing pipeline with conversation memory

        This function orchestrates the entire RAG pipeline:
        1. Loads, splits and embeds each PDF (once per upload)
        2. Shows the conversation so far, with the sources of each answer
        3. Answers a new question with retrieved passages and the recent conversation

        Args:
            uploaded_files (list): Streamlit uploaded file objects
            groq_client: Initialized Groq API client
            embedding_model: Loaded sentence transformer model
        """
        if len(uploaded_files) > MAX_DOCUMENTS:
            st.warning(f"⚠️ Only the first {MAX_DOCUMENTS} files are used.")
            uploaded_files = uploaded_files[:MAX_DOCUMENTS]

        failed = self._prepare(uploaded_files, embedding_model)
        for name, error in failed:
            st.error(f"❌ {name}: {error}")
        store = st.session_state.vector_store
        if not store.documents:
            return

        counts = {}
        for doc in store.documents:
            counts[doc.metadata.get("source")] = counts.get(doc.metadata.get("source"), 0) + 1
        if len(counts) == 1:
            st.success(f"✅ Document ready: {len(store.documents)} chunks")
        else:
            listed = ", ".join(f"{name} ({n} chunks)" for name, n in counts.items())
            st.success(f"✅ {len(counts)} documents ready: {listed}")
        st.header("💬 Ask Your Questions")

        # With several files, a question about one paper is best searched in that paper
        # alone: on QASPER, loading 9 other papers cut evidence retrieval from 48.8% to
        # 31.9% (eval/results/2026-11-01-multidoc-dev)
        scope = None
        if len(counts) > 1:
            choice = st.selectbox(
                "Ask about:",
                [ALL_DOCUMENTS, COMPARE, *counts],
                key="scope",
                help=(
                    "Pick one file when your question is about it: other files' passages can crowd it out. "
                    f"'{COMPARE}' gives every file an equal share of the passages."
                ),
            )
            scope = None if choice == ALL_DOCUMENTS else choice
            if scope is None:
                st.caption("💡 Asking about one paper? Choose it above, so the search finds its passages.")

        # Example questions to help users get started
        examples = EXAMPLE_QUESTIONS + ([COMPARE_QUESTION] if len(counts) > 1 else [])
        cols = st.columns(2)
        for i, (icon, text) in enumerate(examples):
            if cols[i % 2].button(f"{icon} {text}", key=f"example_{i}"):
                st.session_state.pending_question = text
                # The comparison example is searched as a comparison, whatever the scope
                st.session_state.pending_scope = COMPARE if (icon, text) == COMPARE_QUESTION else scope

        if st.session_state.conversation_history and st.button("🗑️ Clear Conversation History"):
            st.session_state.conversation_history = []
            st.session_state.conversation_sources = []
            st.rerun()

        question = st.chat_input("Ask anything about the documents..." if len(counts) > 1 else
                                 "Ask anything about the document...")
        pending_scope = st.session_state.pop("pending_scope", scope)
        if not question:
            question = st.session_state.pop("pending_question", None)
            scope = pending_scope
        if question:
            self._answer(question, groq_client, scope)

        exchanges = zip(st.session_state.conversation_history, st.session_state.conversation_sources, strict=True)
        last = len(st.session_state.conversation_history) - 1
        for n, ((q, a), sources) in enumerate(exchanges):
            with st.chat_message("user"):
                st.write(q)
            with st.chat_message("assistant"):
                st.write(a)
                self._show_sources(a, sources, latest=n == last)
        if st.session_state.conversation_history:
            st.caption(CHECK_NOTE)

        st.markdown(FOOTNOTE, unsafe_allow_html=True)
