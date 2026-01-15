# Route to handle PDF uploads and add them to ChromaDB
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify,send_from_directory
import os
from config import logger
from pathlib import Path

#from rag_system import RAGSystem
from rag_system_v3 import RAGSystem

#app = Flask(__name__)
app = Flask(__name__, static_folder="static", static_url_path="/static")
app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1, x_host=1, x_proto=1)
# BASE_DIR = Path(__file__).resolve().parent
# app = Flask(
#     __name__,
#     static_folder=str(BASE_DIR / "static"),
#     template_folder=str(BASE_DIR / "templates"),
# )
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Limit upload size to 500MB
app.config["APPLICATION_ROOT"] = "/max.fink"
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
    # Accept JSON instead of form data
    data = request.get_json()

    user_query = data.get('question', '').strip()
    session_id = data.get('session_id', 'default')  # Get from client
    excluded_parent_ids = data.get('excluded_parent_ids', [])  # Optional exclusion list

    # Removing the CV document
    #excluded_parent_ids = ["item_6794"]

    if not user_query:
        logger.error('Query cannot be empty')
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        answer = rag.ask(
            user_query,
            chat_session_id=session_id,
            excluded_parent_ids=excluded_parent_ids if excluded_parent_ids else None
        )
        logger.info(f'Answer for session {session_id}: {answer.get("answer", "")[:100]}...')

        result = {
            'answer': answer.get('answer', ''),
            'sources': answer.get('sources', [])
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f'Error processing query for session {session_id}: {str(e)}')
        return jsonify({'error': 'Error processing query'}), 500

@app.route('/cleanup_session', methods=['POST'])
def cleanup_session():
    """Clean up session history when tab closes"""
    session_id = request.form.get('session_id')

    if session_id:
        try:
            rag.cleanup_session(session_id)
            logger.info(f'Cleaned up session: {session_id}')
            return '', 204  # No content response
        except Exception as e:
            logger.error(f'Error cleaning up session {session_id}: {str(e)}')
            return jsonify({'error': 'Cleanup failed'}), 500

    return jsonify({'error': 'No session_id provided'}), 400

if __name__ == '__main__':
    logger.info("Starting Ask the Archive RAG System")
    # Initialize RAGSystem (adjust constructor as needed)
    STORE_DIR = "./fink_archive"

    rag = RAGSystem(
        store_dir=STORE_DIR,
        chroma_collection="rag_collection",
        enable_bm25=True,          # can set False to simplify
        k_recall=15,
        k_ensemble=10,
        k_after_rerank=6,
    )    

    # Run the app
    app.run(debug=False, host='0.0.0.0', port=5067)