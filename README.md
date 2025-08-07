# 📄 Document Q&A Assistant — Gemini RAG App

Ask questions about your PDFs, Word docs, and emails using Google Gemini 2.5.

Built with:
- 🧠 Google Gemini (via `langchain-google-genai`)
- 🗃️ LangChain vector search
- 💬 Streamlit UI
- 🧩 HuggingFace embeddings
- 📂 PDF/DOCX/EML support

---

## ⚙️ Setup Instructions

1. Rename the ".env_exmaple" file to ".env"

2. Paste your gemini api key (Get your api key: https://aistudio.google.com/app/apikey)

3. Create a virtual environment (optional but recommended) :
    python -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate

4. Install required packages :
    pip install -r requirements.txt

5. Run Streamlit app :
    streamlit run app.py

