# Route to handle PDF uploads and add them to ChromaDB
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import os
from config import logger
#from werkzeug.exceptions import NotFound

from rag_system import RAGSystem

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

@app.route('/')
def index():
    """Redirect to home page"""
    return render_template('index.html')

@app.route('/home')
def home():
    """Home page showing all documents"""
    try:
        # Get all documents from ChromaDB
        documents = rag.get_all_documents()
        return render_template('home.html', documents=documents)
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        return render_template('home.html', documents=[])

@app.route('/document/<doc_id>')
def view_document(doc_id):
    """View a specific document"""
    try:
        document = rag.get_document(int(doc_id))

        if not document:
            logger.error('Document not found')
            return redirect(url_for('home'))
            
        return render_template('view_document.html', document=document)
    except Exception as e:
        logger.error(f'Error loading document: {str(e)}')
        return redirect(url_for('home'))

@app.route('/query', methods=['POST'])
def query():
    """Chat interface for interacting with the RAG system"""
    user_query = request.form.get('question', '').strip()
    if not user_query:
        logger.error('Query cannot be empty')
        return jsonify({'error': 'Query cannot be empty'}), 400

    try:
        response = rag.query(user_query)
        # Return only the answer and sources if present
        result = {
            'answer': response.get('answer', ''),
            'sources': response.get('sources', [])
        }
        return jsonify(result)
    except Exception as e:
        logger.error(f'Error processing query: {str(e)}')
        return jsonify({'error': 'Error processing query'}), 500

@app.route('/upload', methods=['GET'])
def upload():
    """Render the upload page"""
    return render_template('upload.html')

@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    """Accepts uploaded PDF files and adds them to ChromaDB via RAGSystem."""
    logger.info(request)
    if 'pdf_files' not in request.files:
        logger.error('No files part in the request')
        return jsonify({'error': 'No files part in the request'}), 400

    files = request.files.getlist('pdf_files')
    if not files or len(files) == 0:
        logger.error('No PDF files selected')
        return jsonify({'error': 'No PDF files selected'}), 400

    saved_paths = []
    upload_folder = os.path.join(os.getcwd(), 'uploads')
    os.makedirs(upload_folder, exist_ok=True)

    for file in files:
        if file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            file_path = os.path.join(upload_folder, filename)
            file.save(file_path)
            saved_paths.append(file_path)
        else:
            logger.warning(f"Skipped non-PDF file: {file.filename}")

    if saved_paths:
        rag.add_documents_from_files(saved_paths)
        logger.info(f"Uploaded and processed {len(saved_paths)} PDF files.")
        return redirect(url_for('home'))
        #return jsonify({'success': True, 'files': saved_paths}), 200
    else:
        logger.error('No valid PDF files uploaded')
        return jsonify({'error': 'No valid PDF files uploaded'}), 400
# @app.route('/document/<doc_id>/edit', methods=['GET', 'POST'])
# def edit_document(doc_id):
#     """Edit a specific document"""
#     if request.method == 'GET':
#         try:
#             # document = rag_system.get_document_by_id(doc_id)
#             document = None  # Placeholder - replace with actual call
            
#             if not document:
#                 flash('Document not found', 'error')
#                 return redirect(url_for('home'))
                
#             return render_template('edit_document.html', document=document)
#         except Exception as e:
#             flash(f'Error loading document: {str(e)}', 'error')
#             return redirect(url_for('home'))
    
#     elif request.method == 'POST':
#         try:
#             title = request.form.get('title', '').strip()
#             content = request.form.get('content', '').strip()
            
#             if not title or not content:
#                 flash('Title and content are required', 'error')
#                 return redirect(url_for('edit_document', doc_id=doc_id))
            
#             # Update document in ChromaDB
#             # success = rag_system.update_document(doc_id, title, content)
#             success = True  # Placeholder - replace with actual call
            
#             if success:
#                 flash('Document updated successfully', 'success')
#                 return redirect(url_for('view_document', doc_id=doc_id))
#             else:
#                 flash('Failed to update document', 'error')
#                 return redirect(url_for('edit_document', doc_id=doc_id))
                
#         except Exception as e:
#             flash(f'Error updating document: {str(e)}', 'error')
#             return redirect(url_for('edit_document', doc_id=doc_id))

# @app.route('/document/new', methods=['GET', 'POST'])
# def new_document():
#     """Create a new document"""
#     if request.method == 'GET':
#         return render_template('new_document.html')
    
#     elif request.method == 'POST':
#         try:
#             title = request.form.get('title', '').strip()
#             content = request.form.get('content', '').strip()
            
#             if not title or not content:
#                 flash('Title and content are required', 'error')
#                 return render_template('new_document.html')
            
#             # Add document to ChromaDB
#             # doc_id = rag_system.add_document(title, content)
#             doc_id = "placeholder_id"  # Placeholder - replace with actual call
            
#             if doc_id:
#                 flash('Document created successfully', 'success')
#                 return redirect(url_for('view_document', doc_id=doc_id))
#             else:
#                 flash('Failed to create document', 'error')
#                 return render_template('new_document.html')
                
#         except Exception as e:
#             flash(f'Error creating document: {str(e)}', 'error')
#             return render_template('new_document.html')

@app.route('/document/<doc_id>/delete', methods=['POST'])
def delete_document(doc_id):
    """Delete a specific document"""
    try:
        success = rag.delete_document(int(doc_id))
        # success = True  # Placeholder - replace with actual call

        if success:
            logger.info('Document deleted successfully')
        else:
            logger.error('Failed to delete document')

    except Exception as e:
        logger.error(f'Error deleting document: {str(e)}')

    return redirect(url_for('home'))

# @app.route('/search')
# def search_documents():
#     """Search documents"""
#     query = request.args.get('q', '').strip()
    
#     if not query:
#         return redirect(url_for('home'))
    
#     try:
#         # results = rag_system.search_documents(query)
#         results = []  # Placeholder - replace with actual call
#         return render_template('search_results.html', query=query, results=results)
#     except Exception as e:
#         flash(f'Search error: {str(e)}', 'error')
#         return redirect(url_for('home'))

# @app.errorhandler(404)
# def not_found(error):
#     return render_template('404.html'), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return render_template('500.html'), 500

if __name__ == '__main__':
    logger.info("Starting ChromaDB Document Manager")
    # Initialize RAGSystem (adjust constructor as needed)
    #rag_system = RAGSystem()
    rag = RAGSystem(
        persist_directory="./chroma_db",
        collection_name="my_documents",
        model_name="Remote_Ollama",  # examples include "Ollama", "Remote_Ollama", "OpenAI"
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Run the app
    app.run(debug=False, host='0.0.0.0', port=5000)