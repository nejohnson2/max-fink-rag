import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

# Set up the environment variables for Ollama
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL")
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# RAG system configuration
ENABLE_MULTI_QUERY = os.getenv("ENABLE_MULTI_QUERY", "false").lower() in ("true", "1", "yes")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)

# Create logger for the application
logger = logging.getLogger("rag_app")