# Route to handle PDF uploads and add them to ChromaDB
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_from_directory, Response
import os
from config import logger, URL_PREFIX, CHAT_LOG_PATH, DEBUG_RETRIEVAL, print_debug_config
from pathlib import Path

from rag_system import RAGSystem

app = Flask(__name__, static_folder="static", static_url_path="/static")

# Only configure proxy middleware if URL_PREFIX is set
if URL_PREFIX:
    app.wsgi_app = ProxyFix(app.wsgi_app, x_prefix=1, x_host=1, x_proto=1)
    app.config["APPLICATION_ROOT"] = URL_PREFIX

app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # Limit upload size to 500MB

# This works
PDF_DIR = os.path.join(os.getcwd(), 'uploads')  # Ensure uploads directory exists

@app.route('/')
def index():
    """Redirect to home page"""
    return render_template('index.html', url_prefix=URL_PREFIX)

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

        # Log interaction for data collection
        metadata = answer.get('_metadata', {})
        RAGSystem.log_interaction(
            log_path=CHAT_LOG_PATH,
            session_id=session_id,
            question=user_query,
            answer=answer.get('answer', ''),
            sources=answer.get('sources', []),
            intent=metadata.get('intent'),
            retrieval_time=metadata.get('retrieval_time'),
            answer_time=metadata.get('answer_time'),
            total_time=metadata.get('total_time'),
            excluded_parent_ids=metadata.get('excluded_parent_ids'),
            num_sources=metadata.get('num_sources'),
        )

        result = {
            'answer': answer.get('answer', ''),
            'sources': answer.get('sources', [])
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f'Error processing query for session {session_id}: {str(e)}')
        return jsonify({'error': 'Error processing query'}), 500

@app.route('/query_stream', methods=['POST'])
def query_stream():
    """Streaming chat interface using Server-Sent Events (SSE)"""
    import json as json_module

    data = request.get_json()
    user_query = data.get('question', '').strip()
    session_id = data.get('session_id', 'default')
    excluded_parent_ids = data.get('excluded_parent_ids', [])

    if not user_query:
        logger.error('Query cannot be empty')
        return jsonify({'error': 'Query cannot be empty'}), 400

    def generate():
        """Generator that yields SSE events"""
        try:
            full_answer = []
            sources = []
            metadata = {}

            for event_type, event_data in rag.ask_streaming(
                user_query,
                chat_session_id=session_id,
                excluded_parent_ids=excluded_parent_ids if excluded_parent_ids else None
            ):
                if event_type == "sources":
                    sources = event_data.get("sources", [])
                    intent = event_data.get("intent")
                    # Send sources as first event
                    yield f"data: {json_module.dumps({'type': 'sources', 'sources': sources, 'intent': intent})}\n\n"

                elif event_type == "token":
                    token = event_data.get("token", "")
                    full_answer.append(token)
                    # Send each token
                    yield f"data: {json_module.dumps({'type': 'token', 'token': token})}\n\n"

                elif event_type == "done":
                    metadata = event_data.get("_metadata", {})
                    # Log interaction for data collection
                    RAGSystem.log_interaction(
                        log_path=CHAT_LOG_PATH,
                        session_id=session_id,
                        question=user_query,
                        answer="".join(full_answer),
                        sources=sources,
                        intent=metadata.get('intent'),
                        retrieval_time=metadata.get('retrieval_time'),
                        answer_time=metadata.get('answer_time'),
                        total_time=metadata.get('total_time'),
                        excluded_parent_ids=metadata.get('excluded_parent_ids'),
                        num_sources=metadata.get('num_sources'),
                    )
                    # Send done event
                    yield f"data: {json_module.dumps({'type': 'done'})}\n\n"

        except Exception as e:
            logger.error(f'Error in streaming query for session {session_id}: {str(e)}')
            yield f"data: {json_module.dumps({'type': 'error', 'error': 'Error processing query'})}\n\n"

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
        }
    )


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

    if DEBUG_RETRIEVAL:
        # Print all configuration settings first
        print_debug_config()
        logger.info("=" * 80)
        logger.info("DEBUG MODE: Flask Application Startup")
        logger.info("=" * 80)
        logger.info("Application Settings:")
        logger.info("  Store directory: %s", os.path.abspath(STORE_DIR))
        logger.info("  Max upload size: %d MB", app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024))
        logger.info("  URL prefix: '%s'", URL_PREFIX if URL_PREFIX else "(empty - local mode)")
        logger.info("  Chat log path: %s", CHAT_LOG_PATH)
        logger.info("  Static folder: %s", app.static_folder)
        logger.info("=" * 80)

    rag = RAGSystem(
        store_dir=STORE_DIR,
        chroma_collection="rag_collection",
        enable_bm25=True,          # can set False to simplify
        k_recall=30,
        k_ensemble=10,
        k_after_rerank=6,
    )

    if DEBUG_RETRIEVAL:
        logger.info("=" * 80)
        logger.info("DEBUG MODE: Server Ready")
        logger.info("  Listening on: http://0.0.0.0:5067")
        logger.info("=" * 80)

    # Run the app
    app.run(debug=False, host='0.0.0.0', port=5067)