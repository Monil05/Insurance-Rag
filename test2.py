import os
import tempfile
import hashlib
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.email import UnstructuredEmailLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# === Load .env file if needed (optional) ===
load_dotenv()

class RAGProcessor:
    def __init__(self):
        self.vector_store = None
        self.retriever = None
        self.rag_chain = None
        self.temp_file_path = None
        self.query_cache = {}  # Simple query cache
        self.setup_components()
    
    def setup_components(self):
        """Initialize embeddings, LLM, and prompt template"""
        # === Embed & Store Vectors ===
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # === Enhanced Prompt Template for Better Accuracy ===
        self.enhanced_prompt = ChatPromptTemplate.from_template("""
You are an expert document analyst. Your job is to provide accurate, comprehensive answers based strictly on the provided context.

Context from document:
{context}

User Query: {question}

CRITICAL INSTRUCTIONS:
1. READ ALL PROVIDED CONTEXT CAREFULLY - Multiple sections may contain relevant information
2. For queries about duration, tenure, term, period - these are often synonymous in policy documents
3. When asked about benefits or coverage, provide comprehensive details from ALL relevant sections
4. If asked about headings/sections, mention the specific section names and numbers
5. Always quote exact text from the document when possible
6. If information spans multiple sections, combine all relevant details
7. For coverage questions: Look for both what IS covered and what is NOT covered
8. Don't assume - if something isn't explicitly stated, say so
9. Be thorough - users prefer complete answers over brief ones

Structure your response as:
- Direct answer to the question
- Supporting details from the document (with exact quotes when relevant)
- Section/heading references where information was found

Answer comprehensively based on ALL provided context:""")

        # === Initialize Gemini LLM with API key from .env ===
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file. Please add your API key to .env file.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=512,  # Increased for more comprehensive answers
            convert_system_message_to_human=True
        )
    
    def load_document(self, uploaded_file):
        """Load document from uploaded file"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                self.temp_file_path = tmp_file.name
            
            # Determine file type and load accordingly
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'pdf':
                loader = PyPDFLoader(self.temp_file_path)
            elif file_extension == 'docx':
                loader = Docx2txtLoader(self.temp_file_path)
            elif file_extension == 'eml':
                loader = UnstructuredEmailLoader(self.temp_file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
            
            docs = loader.load()
            
            # Improved text splitting with better overlap
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,  # Slightly larger chunks for more context
                chunk_overlap=300,  # Better overlap to catch cross-boundary information
                separators=["\n\n", "\n", ".", "!", "?", ";", " ", ""]
            )
            pages = text_splitter.split_documents(docs)
            
            # Create vector store
            self.vector_store = InMemoryVectorStore.from_documents(pages, self.embeddings)
            
            # Improved retriever - balance between speed and accuracy
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 4}  # Increased to 4 for better context coverage
            )
            
            # Build RAG chain
            self.rag_chain = (
                {"context": self.retriever | self.format_docs_with_analysis, "question": RunnablePassthrough()}
                | self.enhanced_prompt
                | self.llm
                | StrOutputParser()
            )
            
            return True, f"Document loaded successfully! Found {len(pages)} text chunks."
            
        except Exception as e:
            return False, f"Error loading document: {str(e)}"
    
    def format_docs_with_analysis(self, docs):
        """Format retrieved docs with clear separation and metadata"""
        if not docs:
            return "No relevant context found."
        
        # Sort docs by relevance score if available
        formatted_context = ""
        for i, doc in enumerate(docs, 1):
            # Include more context information
            page_info = f"Page {doc.metadata.get('page', 'Unknown')}" if doc.metadata.get('page') is not None else "Source document"
            formatted_context += f"\n=== CONTEXT CHUNK {i} ({page_info}) ===\n{doc.page_content}\n"
        
        return formatted_context

    def get_source_info(self, docs):
        """Extract detailed chunk information from retrieved documents"""
        source_info = []
        for i, doc in enumerate(docs, 1):
            page_num = doc.metadata.get('page', 'Unknown')
            if page_num != 'Unknown' and page_num is not None:
                page_num = page_num + 1  # Add 1 because PDF pages are 0-indexed
            
            source_info.append({
                'chunk_number': i,
                'page': page_num,
                'content': doc.page_content.strip(),
                'content_length': len(doc.page_content)
            })
        return source_info

    def create_synonym_expanded_query(self, question):
        """Expand query with common synonyms without hardcoding specific terms"""
        query_lower = question.lower()
        expanded_parts = [question]  # Start with original query
        
        # Common synonym patterns in documents
        synonym_pairs = [
            ('duration', 'tenure term period validity'),
            ('benefits', 'coverage advantages entitlements'),
            ('policy', 'plan scheme contract agreement'),
            ('covered', 'included eligible payable'),
            ('exclusions', 'exceptions limitations restrictions'),
            ('premium', 'cost price payment'),
            ('claim', 'settlement payment reimbursement')
        ]
        
        # Add synonyms if base word found
        for base_word, synonyms in synonym_pairs:
            if base_word in query_lower:
                expanded_parts.append(synonyms)
        
        return ' '.join(expanded_parts)

    def get_cache_key(self, question):
        """Generate cache key for question"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    def process_query(self, question):
        """Process user query with improved accuracy"""
        if not self.rag_chain:
            return "Please upload a document first."
        
        try:
            # Check cache first
            cache_key = self.get_cache_key(question)
            if cache_key in self.query_cache:
                cached_response = self.query_cache[cache_key]
                return cached_response + "\n\n⚡ (Cached response)"
            
            # Create synonym-expanded query for better retrieval
            enhanced_question = self.create_synonym_expanded_query(question)
            
            # Get relevant chunks using enhanced query
            relevant_docs = self.retriever.invoke(enhanced_question)
            
            # If we have docs, proceed with the chain
            if relevant_docs:
                # Use original question for the LLM
                response = self.rag_chain.invoke(question)
                
                # Get detailed source information
                sources = self.get_source_info(relevant_docs)
                
                # Format the complete response with full chunk information
                complete_response = response.strip()
                complete_response += "\n\n" + "="*50
                complete_response += "\n📄 SOURCE CHUNKS USED:\n"
                
                for source in sources:
                    page_info = f"Page {source['page']}" if source['page'] != 'Unknown' else "Source document"
                    complete_response += f"\n--- CHUNK {source['chunk_number']} ({page_info}) ---\n"
                    complete_response += f"{source['content']}\n"
                
                # Cache the response (without source chunks to save memory)
                cache_response = response.strip() + f"\n\n📄 Referenced {len(sources)} chunks from document"
                self.query_cache[cache_key] = cache_response
                
                # Limit cache size
                if len(self.query_cache) > 30:  # Reduced cache size due to longer responses
                    oldest_keys = list(self.query_cache.keys())[:10]
                    for key in oldest_keys:
                        del self.query_cache[key]
                
                return complete_response
            else:
                return "I couldn't find relevant information in the document to answer your question."
                
        except Exception as e:
            return f"An error occurred while processing your query: {str(e)}"
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except Exception as e:
                pass  # Ignore cleanup errors