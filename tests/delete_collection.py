import os
import sys
import logging

sys.path.append('../app')

from rag_system import RAGSystem

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Starting RAG system...")
rag = RAGSystem(
    persist_directory="../chroma_db",
    collection_name="rag_documents",
    model_name="Remote_Ollama",  # examples include "Ollama", "Remote_Ollama", "OpenAI"
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)
logger.info("RAG system initialized.")
num_docs = rag.vector_store.get(include=['documents'])['ids']  # Retrieve all documents in the collection

logger.info(f"Number of documents in collection: {len(num_docs)}")
rag.delete_collection()  # Delete the collection

num_docs = rag.vector_store.get(include=['documents'])['ids']  # Verify the collection is deleted
logger.info(f"Number of documents in collection after deletion: {len(num_docs)}")

logger.info("Collection deleted successfully.")