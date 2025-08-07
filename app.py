import streamlit as st
import atexit
from test2 import RAGProcessor

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

    # === Header ===
    st.title("📄 Document Q&A Assistant")
    st.markdown("Upload your document and ask questions about it using Google Gemini 2.5 Pro!")
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
            st.info("🚀 **Powered by:** Google Gemini 2.5 Pro")
            st.info("🎯 **Mode:** High Accuracy (4 chunks)")
        else:
            st.warning("⚠️ No document loaded")
        
        # Clear chat history button
        if st.session_state.document_loaded and st.session_state.chat_history:
            if st.button("🗑️ Clear Chat History"):
                st.session_state.chat_history = []
                st.rerun()

    # === Main Content Area ===
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Ask Questions")
        
        # Display chat history with improved formatting
        if st.session_state.chat_history:
            st.subheader("Chat History")
            for i, (question, answer) in enumerate(st.session_state.chat_history):
                with st.expander(f"Q{i+1}: {question[:60]}..." if len(question) > 60 else f"Q{i+1}: {question}", expanded=(i == len(st.session_state.chat_history) - 1)):
                    st.markdown(f"**Question:** {question}")
                    st.markdown("**Answer:**")
                    
                    # Split answer and source chunks for better display
                    if "="*50 in answer:
                        main_answer, source_section = answer.split("="*50, 1)
                        st.markdown(main_answer)
                        
                        # Show source chunks in a separate expandable section
                        with st.expander("📄 View Source Chunks", expanded=False):
                            st.text(source_section)
                    else:
                        st.text_area("", value=answer, height=300, key=f"answer_{i}", disabled=True)

        # Question input
        if st.session_state.document_loaded:
            question = st.text_input(
                "Enter your question:",
                placeholder="e.g., What is the duration of this policy?",
                key="question_input"
            )
            
            col_ask, col_example = st.columns([1, 2])
            
            with col_ask:
                if st.button("Ask Question", type="primary", disabled=not question.strip()):
                    if question.strip():
                        # Add progress tracking
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        try:
                            status_text.text("🔍 Searching document...")
                            progress_bar.progress(25)
                            
                            status_text.text("🧠 Processing with AI...")
                            progress_bar.progress(50)
                            
                            status_text.text("📄 Gathering comprehensive context...")
                            progress_bar.progress(75)
                            
                            # Set a timeout for the query
                            import time
                            start_time = time.time()
                            
                            answer = st.session_state.rag_processor.process_query(question)
                            
                            end_time = time.time()
                            response_time = end_time - start_time
                            
                            progress_bar.progress(100)
                            status_text.text(f"✅ Complete in {response_time:.1f}s")
                            
                            st.session_state.chat_history.append((question, answer))
                            
                            # Clear progress indicators after a moment
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.rerun()
                            
                        except Exception as e:
                            progress_bar.empty()
                            status_text.empty()
                            st.error(f"Error: {str(e)}")
                            st.info("💡 Try a simpler question or check your API key.")
            
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
                    if st.button(eq, key=f"example_{eq}", help="Click to use this example"):
                        st.session_state.question_input = eq
                        st.rerun()
        else:
            st.info("👆 Please upload and process a document first to start asking questions.")

    with col2:
        st.header("ℹ️ Instructions")
        st.markdown("""
        **Setup:**
        1. **Create .env file** in your project folder:
           ```
           GEMINI_API_KEY=your_actual_api_key_here
           ```
        2. **Get API Key**: https://aistudio.google.com/
        3. **Student Plan**: https://gemini.google/students/
        4. **Upload document and start chatting**
        
        **Improved Features:**
        - 🎯 **Higher Accuracy**: Uses 4 context chunks
        - 🔍 **Synonym Recognition**: Finds related terms automatically
        - 📄 **Full Source Display**: Shows complete chunks used
        - 🧠 **Better Understanding**: Improved prompts for comprehensive answers
        
        **Question formats:**
        - Direct: "What is the policy duration?"
        - Section-based: "Which section mentions benefits?"
        - Conversational: "46M, knee surgery, covered?"
        - Comprehensive: "What are all the covered benefits?"
        
        **Supported documents:**
        - 📄 PDF files
        - 📝 Word documents (.docx)
        - 📧 Email files (.eml)
        
        **Best for:**
        - Insurance policies
        - Legal documents
        - HR policies
        - Contracts
        - Compliance documents
        
        **Performance:**
        - ⚡ **Speed**: ~3-6 seconds per query
        - 🎯 **Accuracy**: Enhanced with better context
        - 💾 **Memory**: Smart caching for repeat queries
        
        **Tips:**
        - Ask about specific sections or headings
        - Use synonyms (duration = tenure = term)
        - Request comprehensive coverage details
        - Check source chunks for verification
        """)
        
        # Document info
        if st.session_state.document_loaded and uploaded_file:
            st.markdown("---")
            st.subheader("📊 Document Info")
            st.write(f"**Name:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.write(f"**Type:** {uploaded_file.type}")

# === Run App ===
if __name__ == "__main__":
    main()