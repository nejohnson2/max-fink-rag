# Route to handle PDF uploads and add them to ChromaDB
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify,send_from_directory
import os
from config import logger

#from rag_system import RAGSystem
from rag_system_v3 import RAGSystem

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Limit upload size to 500MB
#app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# This works
PDF_DIR = os.path.join(os.getcwd(), 'uploads')  # Ensure uploads directory exists

@app.route('/')
def index():
    """Redirect to home page"""
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    """Chat interface for interacting with the RAG system"""
    user_query = request.form.get('question', '').strip()
    if not user_query:
        logger.error('Query cannot be empty')
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        #response = rag.query(user_query)
        chat_session_id = 'nb_test_session'  # For testing, use a fixed session ID
        answer = rag.ask(user_query, chat_session_id=chat_session_id)
        logger.info(f'Answer: {answer}')
        # Return only the answer and sources if present
        result = {
            'answer': answer.get('answer', ''),
            'sources': answer.get('sources', [])
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f'Error processing query: {str(e)}')
        return jsonify({'error': 'Error processing query'}), 500
    

if __name__ == '__main__':
    logger.info("Starting Ask the Archive RAG System")
    # Initialize RAGSystem (adjust constructor as needed)
    STORE_DIR = "./fink_archive"

    rag = RAGSystem(
        store_dir=STORE_DIR,
        chroma_collection="rag_collection",
        enable_bm25=True,          # can set False to simplify
        k_recall=30,
        k_ensemble=20,
        k_after_rerank=6,
    )    

    # Run the app
    app.run(debug=False, host='0.0.0.0', port=5000)