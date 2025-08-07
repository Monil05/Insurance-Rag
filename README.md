LLM Document Processing System
An intelligent, self-contained application that uses a Large Language Model (LLM) to process natural language queries and extract structured information from unstructured documents. The system is built entirely with Streamlit, providing a single, user-friendly interface that handles all processing logic.

🚀 Features

Intelligent Document Retrieval: Uses a Retrieval-Augmented Generation (RAG) pipeline with an in-memory vector store to find and retrieve relevant clauses from documents based on semantic understanding, not just keyword matching.

Structured Output: Provides a consistent, machine-readable JSON response for each query, containing a clear decision, amount, and justification.

Explicit Query Parsing: The system parses natural language queries to extract key details like age, procedure, and policy duration before processing.

Flexible Document Support: Supports a variety of document types, including PDF (.pdf), Word documents (.docx), and emails (.eml).

⚙️ Tech Stack

Streamlit: For building the interactive, web-based user interface and handling all backend logic.

LangChain: Orchestrates the RAG pipeline.

Google Gemini 2.0 Flash (gemini-2.0-flash-exp): The Large Language Model used for document analysis and response generation.

HuggingFace Embeddings: Used to create the numerical representations of document chunks.

Pydantic: Defines the structured data model for the JSON output, ensuring consistency.

📦 Installation & Setup

Follow these steps to set up and run the project locally.

1. Clone the repository
First, clone your project's repository from GitHub.

git clone [your-repository-url]
cd [your-project-folder]

2. Create a virtual environment
It is highly recommended to use a virtual environment to manage dependencies.

python -m venv venv

Activate the virtual environment
For macOS/Linux:
source venv/bin/activate

For Windows:
venv\Scripts\activate

3. Install dependencies

Install all the required libraries from the requirements.txt file.

pip install -r requirements.txt

4. Configure your API key

The system requires a Google Gemini API key. Create a .env file in your project's root directory and add your key.

.env
GEMINI_API_KEY=your_actual_api_key_here

5. Run the application

Run the Streamlit application with a single command.

streamlit run app.py

Your Streamlit application should now open in your browser.

🖥️ How to Use

Upload a Document: Use the "Choose a file" button in the left sidebar to upload a PDF, DOCX, or EML file.

Process the Document: Click the "Process Document" button to have the app ingest the file and create the in-memory vector store.

Ask a Question: Once the document is processed, type your natural language query into the text box and click "Ask Question," or select one of the example questions.

View the Answer: The application will display a structured response, including the decision, amount, justification, and the source chunks used by the LLM.

🌐 Architecture Overview

This project uses a single, monolithic architecture:

The Streamlit app is a self-contained application that handles both the user interface and all backend processing, including document ingestion, RAG, and communication with the LLM.

🔗 Deployment

This application can be easily deployed to the Streamlit Community Cloud platform by connecting your GitHub repository.