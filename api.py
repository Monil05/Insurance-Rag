import os
import tempfile
from flask import Flask, request, jsonify
from test2 import RAGProcessor

# Initialize the Flask app
app = Flask(__name__)

# Initialize the RAG processor as a global object
# It's a good practice to handle initialization outside of the request
# context to avoid re-creating the object for every request.
# The RAGProcessor can load the document and keep the vector store in memory.
# Note: In a production environment, you would want to handle multiple documents
# and potentially a shared persistent vector database.
rag_processor = RAGProcessor()

@app.route('/upload', methods=['POST'])
def upload_document():
    """
    API endpoint to handle document uploads.
    The uploaded file is saved to a temporary location, then processed
    by the RAGProcessor to create the in-memory vector store.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Pass the uploaded file directly to the RAGProcessor
        # Using a temporary file is a good way to handle file data from Flask
        success, message = rag_processor.load_document(uploaded_file)
        
        if success:
            return jsonify({"status": "success", "message": message}), 200
        else:
            return jsonify({"status": "error", "message": message}), 500

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/query', methods=['POST'])
def process_query():
    """
    API endpoint to process a user's query against the loaded document.
    It expects a JSON payload with a 'question' key.
    """
    if not rag_processor.rag_chain:
        return jsonify({"error": "No document loaded. Please upload a document first."}), 400
    
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({"error": "Missing 'question' in request body"}), 400

    try:
        # Process the query using the RAGProcessor
        response = rag_processor.process_query(question)
        
        # The response from process_query can be a string or a dict.
        # We need to handle both cases for JSON serialization.
        if isinstance(response, str):
            return jsonify({"error": response}), 500
        
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app on a specific port, e.g., 8000
    # In a real deployment, you would use a production WSGI server like Gunicorn
    app.run(host='0.0.0.0', port=8000, debug=True)
