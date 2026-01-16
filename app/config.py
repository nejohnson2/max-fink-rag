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

# Deployment configuration
URL_PREFIX = os.getenv("URL_PREFIX", "")

# Data collection configuration
CHAT_LOG_PATH = os.getenv("CHAT_LOG_PATH", "logs/chat_interactions.jsonl")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout
)

# Create logger for the application
logger = logging.getLogger("rag_app")

# Load prompts from prompts.py
try:
    from prompts import SYSTEM_PROMPT, INTENT_CLASSIFICATION_PROMPT
    logger.info("Loaded system prompts from prompts.py")
except ImportError as e:
    logger.warning(f"Could not load prompts.py: {e}. Using default prompts.")
    SYSTEM_PROMPT = "You are a helpful assistant."
    INTENT_CLASSIFICATION_PROMPT = "Classify this question."