import os
import tempfile
import hashlib
from dotenv import load_dotenv
import json
from pydantic import BaseModel, Field, ValidationError

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_community.document_loaders.email import UnstructuredEmailLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate

# === Define the structured output model ===
class ClaimDecision(BaseModel):
    """Structured output for a claim decision."""
    decision: str = Field(description="The decision of the claim, e.g., 'Approved', 'Rejected', or 'Needs More Info'.")
    amount: float = Field(description="The final payout amount in INR. Should be 0 if the claim is rejected.")
    justification: str = Field(description="A detailed explanation for the decision, referencing specific clauses or rules from the document context.")
    # Add a field for the raw extracted query details for debugging/audit
    query_details: dict = Field(description="Structured details extracted from the user's query.")

# === Load .env file if needed (optional) ===
load_dotenv()

class RAGProcessor:
    def __init__(self):
        # FIX: Moved parser initialization to the start of the method
        self.parser = JsonOutputParser(pydantic_object=ClaimDecision)
        
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
        
        # === Define the structured prompt template ===
        # The prompt now explicitly instructs the LLM to output JSON and provides the schema.
        self.structured_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert document analyst. Your job is to analyze user queries and document context to provide a structured, justified decision in JSON format. Your decisions MUST be based strictly on the provided context. If a detail is not in the document, you cannot make a decision on it."),
                ("user", """
Context from document:
{context}

User Query: {question}

CRITICAL INSTRUCTIONS:
1. First, analyze the 'User Query' and extract key details like age, procedure, location, and policy duration. Put these into a 'query_details' dictionary.
2. Based on ALL the provided 'Context from document', evaluate the user's query against the rules and clauses.
3. Determine a 'decision' (e.g., 'Approved' or 'Rejected'). If you cannot make a definitive decision due to missing information, use 'Needs More Info'.
4. Determine the 'amount'. If the claim is rejected, the amount must be 0. If it is approved and the document specifies an amount, provide that. If no amount is specified, you can state 'As per document, amount not specified'.
5. Provide a detailed 'justification' that directly references the specific clauses, sections, or rules from the context that led to your decision.
6. The final output must be a valid JSON object matching this schema:
{format_instructions}

Answer comprehensively based on ALL provided context.
""")
            ]
        ).partial(format_instructions=self.parser.get_format_instructions())


        # === Initialize Gemini LLM with API key from .env ===
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file. Please add your API key to .env file.")
        
        # Using a more suitable model for this task
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-exp",
            google_api_key=api_key,
            temperature=0.1,
            max_output_tokens=1024,  # Increased for comprehensive JSON output
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
                search_kwargs={"k": 5}  # Increased to 5 for better context coverage
            )
            
            # Build RAG chain
            self.rag_chain = (
                {"context": self.retriever, "question": RunnablePassthrough()}
                | self.structured_prompt
                | self.llm
                | self.parser
            )
            
            return True, f"Document loaded successfully! Found {len(pages)} text chunks."
            
        except Exception as e:
            return False, f"Error loading document: {str(e)}"

    def get_source_info(self, docs):
        """Extract detailed chunk information from retrieved documents"""
        source_info = []
        for i, doc in enumerate(docs, 1):
            # Pages are 0-indexed in metadata, so add 1 for display
            page_num = doc.metadata.get('page', 'Unknown')
            if page_num != 'Unknown' and isinstance(page_num, int):
                page_num += 1
            
            source_info.append({
                'chunk_number': i,
                'page': page_num,
                'content': doc.page_content.strip()
            })
        return source_info

    def get_cache_key(self, question):
        """Generate cache key for question"""
        return hashlib.md5(question.lower().strip().encode()).hexdigest()

    def process_query(self, question):
        """Process user query and return a structured JSON response."""
        if not self.rag_chain:
            return "Please upload a document first."
        
        try:
            # Check cache first
            cache_key = self.get_cache_key(question)
            if cache_key in self.query_cache:
                cached_response = self.query_cache[cache_key]
                # Add a flag to indicate it's a cached response for the UI
                cached_response['is_cached'] = True
                return cached_response
            
            # Get relevant chunks using the retriever separately to pass to the UI
            relevant_docs = self.retriever.invoke(question)
            
            if not relevant_docs:
                return "I couldn't find relevant information in the document to answer your question."
                
            # Invoke the RAG chain
            response_json = self.rag_chain.invoke(question)

            # Pydantic will parse the response, so no need for manual JSON parsing
            # The chain already outputs the Pydantic object
            structured_response = response_json

            # Return the structured response and the source info
            sources = self.get_source_info(relevant_docs)

            # Add sources to the response object before caching
            structured_response['sources'] = sources

            # Cache the complete response object
            self.query_cache[cache_key] = structured_response
            
            # Limit cache size
            if len(self.query_cache) > 30:
                oldest_keys = list(self.query_cache.keys())[:10]
                for key in oldest_keys:
                    del self.query_cache[key]
            
            return structured_response
                
        except Exception as e:
            return f"An error occurred while processing your query: {str(e)}"
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_file_path and os.path.exists(self.temp_file_path):
            try:
                os.unlink(self.temp_file_path)
            except Exception as e:
                pass  # Ignore cleanup errors


