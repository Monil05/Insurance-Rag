import streamlit as st
import atexit
from test2 import RAGProcessor
import json
import time

# === Page Configuration ===
st.set_page_config(
    page_title="Document Q&A Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# === Initialize RAG Processor ===
@st.cache_resource
def get_rag_processor():
    try:
        return RAGProcessor()
    except ValueError as e:
        st.error(f"❌ Configuration Error: {str(e)}")
        st.info("💡 Please create a .env file with your GEMINI_API_KEY")
        return None

# === Cleanup function ===
def cleanup_resources():
    if 'rag_processor' in st.session_state:
        st.session_state.rag_processor.cleanup()

# Register cleanup function
atexit.register(cleanup_resources)

# === Main App ===
def main():
    # Initialize session state
    if 'rag_processor' not in st.session_state:
        st.session_state.rag_processor = get_rag_processor()
    if 'document_loaded' not in st.session_state:
        st.session_state.document_loaded = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # === Functions to handle button clicks ===
    def ask_question(question):
        """Processes a given question and adds the response to chat history."""
        if question.strip():
            with st.spinner("Processing your query..."):
                start_time = time.time()
                response_obj = st.session_state.rag_processor.process_query(question)
                end_time = time.time()
                response_time = end_time - start_time
                st.success(f"✅ Complete in {response_time:.1f}s")
                st.session_state.chat_history.append((question, response_obj))
                # Trigger a rerun to update the UI with the new chat message
                st.rerun()

    # === Header ===
    st.title("📄 Document Q&A Assistant")
    st.markdown("Upload your document and ask questions about it using Google Gemini 2.0 Flash!")
    st.markdown("---")

    # Check if RAG processor is available
    if not st.session_state.rag_processor:
        st.error("❌ Unable to initialize Gemini. Please check your .env file.")
        st.stop()

    # === Sidebar for Document Upload ===
    with st.sidebar:
        st.header("📁 Document Upload")
        
        st.success("🔑 API Key loaded from .env")
        
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=['pdf', 'docx', 'eml'],
            help="Upload PDF, DOCX, or EML files"
        )
        
        if uploaded_file is not None:
            if st.button("Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    success, message = st.session_state.rag_processor.load_document(uploaded_file)
                    
                    if success:
                        st.session_state.document_loaded = True
                        st.success(message)
                        st.session_state.chat_history = []  # Clear chat history for new document
                    else:
                        st.error(message)
        
        # Document status
    st.markdown("---")
    if st.session_state.document_loaded:
        st.success("✅ Document loaded successfully!")
        st.info(f"📄 **Current document:** {uploaded_file.name if uploaded_file else 'Unknown'}")
        st.info("🚀 **Powered by:** Google Gemini 2.0 Flash")
    else:
        st.warning("⚠️ No document loaded")
    
    # Clear chat history button
    if st.session_state.document_loaded and st.session_state.chat_history:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

    # === Main Content Area ===
    st.header("💬 Ask Questions")
    
    # Display chat history with structured output
    if st.session_state.chat_history:
        for i, (question, response) in enumerate(st.session_state.chat_history):
            with st.expander(f"Q{i+1}: {question[:60]}..." if len(question) > 60 else f"Q{i+1}: {question}", expanded=(i == len(st.session_state.chat_history) - 1)):
                st.markdown(f"**Question:** {question}")
                
                # Check if the response is a structured object
                if isinstance(response, dict):
                    st.markdown("### 📊 Decision & Justification")
                    if 'is_cached' in response and response['is_cached']:
                        st.info("⚡ This is a cached response.")

                    st.markdown(f"**Decision:** {response['decision']}")
                    st.markdown(f"**Amount:** {response['amount'] if response['amount'] is not None else 'Not Specified'}")
                    st.markdown(f"**Justification:** {response['justification']}")
                    
                    if 'query_details' in response:
                        with st.expander("🔍 View Parsed Query Details", expanded=False):
                            st.json(response['query_details'])

                    if 'sources' in response:
                        with st.expander("📄 View Source Chunks", expanded=False):
                            for source in response['sources']:
                                page_info = f"Page {source['page']}" if source['page'] != 'Unknown' else "Source document"
                                st.markdown(f"--- **Chunk {source['chunk_number']}** ({page_info}) ---")
                                st.text(source['content'])
                else:
                    st.error("Error: Could not retrieve a structured response.")
                    st.text(response)

    # Question input
    if st.session_state.document_loaded:
        question_input = st.text_input(
            "Enter your question:",
            placeholder="e.g., 46M, knee surgery, Pune, 3-month policy",
            key="question_input"
        )
        
        col_ask, col_example = st.columns([1, 2])
        
        with col_ask:
            # Main "Ask" button now uses the helper function
            if st.button("Ask Question", type="primary", disabled=not question_input.strip()):
                ask_question(question_input)

        with col_example:
            st.markdown("**Example questions:**")
            example_questions = [
                "What is the policy duration?",
                "What benefits are covered?",
                "Which section mentions tenure?",
                "45M, heart surgery, covered?",
                "What are the exclusions?"
            ]
            
            for eq in example_questions:
                # Example buttons now directly trigger the helper function via a callback
                st.button(eq, key=f"example_{eq}", help="Click to use this example", on_click=ask_question, args=(eq,))

    else:
        st.info("👆 Please upload and process a document first to start asking questions.")

    st.markdown("---")
    st.header("ℹ️ Instructions")
    st.markdown("""
    **Setup:**
    1. **Create .env file** in your project folder:
        ```
        GEMINI_API_KEY=your_actual_api_key_here
        ```
    2. **Get API Key**: https://aistudio.google.com/
    3. **Upload document and start chatting**
    
    **New Features:**
    - 🎯 **Structured JSON Output**: Returns a machine-readable JSON object.
    - 🔍 **Explicit Query Parsing**: The LLM is now instructed to explicitly parse and return key query details.
    - 🧠 **Rule-Based Decisions**: The system is designed to provide a clear decision with justification, directly referencing document clauses.
    - 📄 **Separate Sources**: Source chunks are now displayed separately from the main decision.
    
    **Supported documents:**
    - 📄 PDF files
    - 📝 Word documents (.docx)
    - 📧 Email files (.eml)
    """)
    st.markdown("---")
    
    # === Run App ===
if __name__ == "__main__":
    main()
