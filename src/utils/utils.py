import os
import tempfile

from groq import Groq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

SYSTEM_PROMPT = """You are a document analysis assistant with conversation memory. Your capabilities:

1. DOCUMENT GROUNDING: Always base answers on the provided document context
2. CONVERSATION AWARENESS: Remember and reference previous exchanges when relevant
3. REFERENCE RESOLUTION: When users say "that", "it", "the topic", understand what they're referring to
4. CLARITY: If a reference is ambiguous, ask for clarification
5. ACCURACY: Never make up information not in the document or conversation. If the context does not contain the answer, say that the document does not say.

Each passage in the document context starts with its page number, e.g. [p. 3], or, when several documents are loaded, its file and page, e.g. [smith2020.pdf, p. 3]. Cite the pages you used, and when several documents are loaded, say which document each point comes from.

You maintain context across the conversation while staying grounded in the document."""


class Utils:

    def __init__(self):
        pass

    @staticmethod
    def initialize_groq(api_key: str):
        """
        Initialize the Groq API client.

        Args:
            api_key (str): Your Groq API key.

        Returns:
            Groq: An initialized Groq client instance.
        """
        return Groq(api_key=api_key)

    @staticmethod
    def split_pdf(path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """
        Load a PDF from disk and split it into overlapping chunks.

        Each chunk keeps the page metadata from PyPDFLoader, so answers can cite pages.

        Args:
            path (str): Path to the PDF file.
            chunk_size (int): Maximum characters per chunk.
            chunk_overlap (int): Characters shared by consecutive chunks.

        Returns:
            list: LangChain Document objects, one per chunk.
        """
        documents = PyPDFLoader(str(path)).load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        return text_splitter.split_documents(documents)

    @staticmethod
    def load_and_split_pdf(uploaded_file):
        """
        Load an uploaded PDF and return its chunks.

        Each chunk's metadata["source"] is the uploaded file's name (not the temporary
        path), so answers over several files can say where a passage came from.

        Args:
            uploaded_file: Streamlit uploaded file object (anything with getvalue() and name).

        Returns:
            list: LangChain Document objects, one per chunk.

        Raises:
            ValueError: If the PDF cannot be read.
        """
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name
        try:
            chunks = Utils.split_pdf(temp_file_path)
            for chunk in chunks:
                chunk.metadata["source"] = uploaded_file.name
            return chunks
        except Exception as e:
            raise ValueError(f"Could not read the PDF: {e}") from e
        finally:
            os.remove(temp_file_path)  # Clean up temporary file

    @staticmethod
    def format_context(documents):
        """
        Join retrieved chunks into one context string, each labelled with its page.

        When the chunks come from more than one file, the label also names the file.

        Args:
            documents (list): LangChain Document objects.

        Returns:
            str: The chunks separated by blank lines, each starting with "[p. N]" or
                "[file, p. N]".
        """
        several = len({doc.metadata.get("source") for doc in documents}) > 1
        return "\n\n".join(f"[{Utils.source_label(doc, several)}] {doc.page_content}" for doc in documents)

    @staticmethod
    def source_label(doc, with_file=True):
        """
        Describe where a chunk comes from.

        Args:
            doc: A LangChain Document object.
            with_file (bool): Include the file name, if the chunk has one.

        Returns:
            str: e.g. "p. 3" or "smith2020.pdf, p. 3" ("p. ?" if the page is unknown).
        """
        page = doc.metadata.get("page")
        label = f"p. {page + 1}" if isinstance(page, int) else "p. ?"
        source = doc.metadata.get("source")
        return f"{source}, {label}" if with_file and source else label

    @staticmethod
    def build_messages(context, question, conversation_history):
        """
        Build the chat messages for one question.

        The system prompt comes first, then the previous exchanges, then the current
        question with its document context.

        Args:
            context (str): Relevant document chunks as context.
            question (str): Current user question.
            conversation_history (list): Previous (question, answer) pairs to include.

        Returns:
            list: Messages in the chat-completions format.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for prev_q, prev_a in conversation_history:
            messages.append({"role": "user", "content": prev_q})
            messages.append({"role": "assistant", "content": prev_a})
        messages.append(
            {
                "role": "user",
                "content": f"Document Context:\n{context}\n\nCurrent Question:\n{question}",
            }
        )
        return messages

    @staticmethod
    def get_groq_response(client, context, question, conversation_history, model_name="llama-3.1-8b-instant"):
        """
        Get response from Groq API using RAG pattern with conversation memory

        Args:
            client (Groq): Initialized Groq client (or any client with the same
                chat.completions.create interface)
            context (str): Relevant document chunks as context
            question (str): Current user question
            conversation_history (list): Previous Q&A pairs, already trimmed
            model_name (str): Groq model to use

        Returns:
            str: Generated answer with conversation awareness

        Raises:
            Exception: Whatever the client raises, so the caller can tell rate limits
                and context-length errors apart.
        """
        response = client.chat.completions.create(
            messages=Utils.build_messages(context, question, conversation_history),
            model=model_name,
            temperature=0.1,  # Low temperature for factual, consistent answers
            max_tokens=1000,  # Reasonable response length
        )
        return response.choices[0].message.content

    @staticmethod
    def load_embedding_model():
        """
        Load sentence transformer model for creating embeddings locally.

        The import is local so that the rest of this module works without torch.

        Returns:
            SentenceTransformer: Loaded embedding model
        """
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL)

    @staticmethod
    def manage_conversation_context(conversation_history, max_exchanges=10):
        """
        Manage conversation history to prevent token overflow

        Args:
            conversation_history (list): List of (question, answer) tuples
            max_exchanges (int): Maximum number of exchanges to keep

        Returns:
            list: The last max_exchanges exchanges
        """
        if max_exchanges <= 0:
            return []
        return conversation_history[-max_exchanges:]
