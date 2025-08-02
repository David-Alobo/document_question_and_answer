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

    def process_document(self, uploaded_file, groq_client, embedding_model):
        """
        Main document processing pipeline with conversation memory
        
        This function orchestrates the entire RAG pipeline:
        1. Loads and splits the PDF
        2. Creates embeddings and vector store
        3. Sets up the conversational Q&A interface
        4. Handles user questions with conversation context
        
        Args:
            uploaded_file: Streamlit uploaded file object
            groq_client: Initialized Groq API client
            embedding_model: Loaded sentence transformer model
        """
        st.write("📄 Processing your document...")
        
        # Step 1: Load and split PDF
        with st.spinner("📖 Reading PDF..."):
            chunks = Utils.load_and_split_pdf(uploaded_file)
        
        if not chunks:
            st.error("❌ Could not extract text from PDF")
            return
        
        st.success(f"✅ Document loaded! Found {len(chunks)} chunks")
        
        # Step 2: Create vector store with embeddings
        with st.spinner("🧮 Creating embeddings (running locally)..."):
            vector_store = LocalVectorStore(embedding_model)
            vector_store.add_documents(chunks)
        
        st.success("✅ Document ready for questions!")
        
        # Step 3: Initialize conversation history if not exists
        if 'conversation_history' not in st.session_state:
            st.session_state.conversation_history = []
        
        # Step 4: Store everything in session state for persistence
        st.session_state.vector_store = vector_store
        st.session_state.groq_client = groq_client
        st.session_state.ready = True

        # Step 5: Conversational Q&A Interface
        if st.session_state.get('ready', False):
            st.header("💬 Ask Your Questions")
            
            # Show conversation status
            if st.session_state.conversation_history:
                st.info(f"💭 Conversation memory: {len(st.session_state.conversation_history)} exchanges")
            
            # Provide example questions to help users get started
            st.write("**Try asking:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("📋 What is this document about?"):
                    st.session_state.question = "What is this document about?"
                if st.button("👥 Who are the main authors or people mentioned?"):
                    st.session_state.question = "Who are the main authors or people mentioned?"
            with col2:
                if st.button("🔍 What are the key findings or conclusions?"):
                    st.session_state.question = "What are the key findings or conclusions?"
                if st.button("📊 Can you elaborate on that?"):
                    st.session_state.question = "Can you elaborate on that?"
            
            # Clear conversation button
            if st.session_state.conversation_history:
                if st.button("🗑️ Clear Conversation History"):
                    st.session_state.conversation_history = []
                    st.success("Conversation history cleared!")
                    st.rerun()
            
            # Main question input
            question = st.text_input(
                "Your question:", 
                value=st.session_state.get('question', ''),
                key="user_question",
                placeholder="Ask anything about the document... I remember our conversation!"
            )
            
            # Process question when user enters one
            if question:
                try:
                    with st.spinner("🤔 Thinking... (using conversation context + Groq's lightning-fast API)"):
                        # Step 5a: Find relevant chunks using similarity search
                        relevant_chunks = st.session_state.vector_store.similarity_search(question, k=4)
                        
                        if not relevant_chunks:
                            st.warning("🤷 No relevant information found. Try rephrasing your question.")
                            return
                        
                        # Step 5b: Combine chunks into context
                        context = "\n\n".join(relevant_chunks)
                        
                        # Step 5c: Get conversation history
                        conversation_history = Utils.manage_conversation_context(
                            st.session_state.conversation_history, 
                            max_exchanges=10
                        )
                        
                        # Step 5d: Get response with conversation memory
                        answer = Utils.get_groq_response(
                            st.session_state.groq_client, 
                            context, 
                            question, 
                            conversation_history,
                            st.session_state.get('selected_model', 'llama-3.1-8b-instant')
                        )
                        
                        # Step 5e: Store this Q&A in conversation history
                        st.session_state.conversation_history.append((question, answer))
                    
                    # Step 5f: Display results
                    st.write("**🎯 Answer:**")
                    st.write(answer)
                    
                    # Show performance info
                    st.success("⚡ Powered by Groq's blazing-fast inference + conversation memory!")
                    
                    # Show conversation history
                    if len(st.session_state.conversation_history) > 1:
                        with st.expander("💬 Conversation History"):
                            for i, (q, a) in enumerate(st.session_state.conversation_history[:-1]):  # Exclude current
                                st.write(f"**Q{i+1}:** {q}")
                                display_answer = a[:200] + "..." if len(a) > 200 else a
                                st.write(f"**A{i+1}:** {display_answer}")
                                st.write("---")
                    
                    # Show source chunks for transparency and debugging
                    with st.expander("📚 View source chunks"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.write(f"**Chunk {i+1}:**")
                            # Truncate long chunks for readability
                            display_chunk = chunk[:400] + "..." if len(chunk) > 400 else chunk
                            st.write(display_chunk)
                            st.write("---")
                    
                except Exception as e:
                    # Handle different types of errors gracefully
                    if "rate limit" in str(e).lower():
                        st.error("🕐 Rate limit reached. Please wait a moment and try again.")
                        st.info("💡 Free tier limits are generous but not unlimited!")
                    elif "context_length" in str(e).lower():
                        st.error("📏 Conversation too long. Clearing older messages...")
                        st.session_state.conversation_history = st.session_state.conversation_history[-5:]
                        st.info("💡 Try asking your question again!")
                    else:
                        st.error(f"❌ Error: {str(e)}")
                        st.info("💡 Try simplifying your question or check your API key.")