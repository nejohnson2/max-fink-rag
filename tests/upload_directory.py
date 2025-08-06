import os
import sys
import logging

# Add the app directory to the path
sys.path.append('../app')

from rag_system import RAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def upload_pdfs_from_folder(folder_path):
    """Uploads all PDFs in the given folder to the vector database using RAGSystem."""
    if not os.path.isdir(folder_path):
        logger.error(f"Provided path is not a directory: {folder_path}")
        return

    # Find all PDF files in the folder
    pdf_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.lower().endswith('.pdf') and os.path.isfile(os.path.join(folder_path, f))
    ]

    if not pdf_files:
        logger.warning(f"No PDF files found in {folder_path}")
        return

    logger.info(f"Found {len(pdf_files)} PDF files in {folder_path}: {pdf_files}")

    # Initialize RAGSystem
    rag = RAGSystem(
        persist_directory="../chroma_db",
        collection_name="rag_documents",
        model_name="Remote_Ollama",  # or "Ollama" or "OpenAI" as appropriate
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Upload PDFs
    rag.add_documents_from_files(pdf_files)
    logger.info("Upload complete.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload all PDFs in a folder to the vector database.")
    parser.add_argument("folder", help="Path to the folder containing PDF files")
    args = parser.parse_args()