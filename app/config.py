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
ENABLE_INTENT_CLASSIFICATION = os.getenv("ENABLE_INTENT_CLASSIFICATION", "true").lower() in ("true", "1", "yes")
ENABLE_BIOGRAPHY = os.getenv("ENABLE_BIOGRAPHY", "false").lower() in ("true", "1", "yes")
ENABLE_PARENT_CHILD = os.getenv("ENABLE_PARENT_CHILD", "true").lower() in ("true", "1", "yes")
DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "false").lower() in ("true", "1", "yes")

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

# Load biography context from biography.md (if enabled)
# This content is included in every query to provide foundational knowledge
from pathlib import Path

def print_debug_config():
    """Print all configuration values when DEBUG_RETRIEVAL is enabled."""
    logger.info("=" * 80)
    logger.info("DEBUG MODE: Configuration Settings")
    logger.info("=" * 80)
    logger.info("LLM Configuration:")
    logger.info("  OLLAMA_URL: %s", OLLAMA_URL)
    logger.info("  OLLAMA_MODEL: %s", OLLAMA_MODEL)
    logger.info("  OLLAMA_API_KEY: %s", "***" + OLLAMA_API_KEY[-4:] if OLLAMA_API_KEY and len(OLLAMA_API_KEY) > 4 else "[not set]")
    logger.info("-" * 40)
    logger.info("RAG System Configuration:")
    logger.info("  ENABLE_MULTI_QUERY: %s", ENABLE_MULTI_QUERY)
    logger.info("  ENABLE_INTENT_CLASSIFICATION: %s", ENABLE_INTENT_CLASSIFICATION)
    logger.info("  ENABLE_BIOGRAPHY: %s", ENABLE_BIOGRAPHY)
    logger.info("  ENABLE_PARENT_CHILD: %s", ENABLE_PARENT_CHILD)
    logger.info("  DEBUG_RETRIEVAL: %s", DEBUG_RETRIEVAL)
    logger.info("-" * 40)
    logger.info("Deployment Configuration:")
    logger.info("  URL_PREFIX: '%s'", URL_PREFIX if URL_PREFIX else "(empty - local mode)")
    logger.info("  CHAT_LOG_PATH: %s", CHAT_LOG_PATH)
    logger.info("-" * 40)
    logger.info("System Prompt (%d chars):", len(SYSTEM_PROMPT))
    # Print system prompt with indentation, line by line
    for line in SYSTEM_PROMPT.split("\n"):
        logger.info("  %s", line)
    logger.info("-" * 40)
    logger.info("Intent Classification Prompt (%d chars):", len(INTENT_CLASSIFICATION_PROMPT))
    for line in INTENT_CLASSIFICATION_PROMPT.split("\n"):
        logger.info("  %s", line)
    logger.info("=" * 80)

BIOGRAPHY_PATH = Path(__file__).parent / "biography.md"
if ENABLE_BIOGRAPHY:
    try:
        BIOGRAPHY_CONTEXT = BIOGRAPHY_PATH.read_text(encoding="utf-8")
        logger.info(f"Loaded biography context from {BIOGRAPHY_PATH}")
    except FileNotFoundError:
        logger.warning(f"Biography file not found at {BIOGRAPHY_PATH}. No biography context will be used.")
        BIOGRAPHY_CONTEXT = ""
    except Exception as e:
        logger.warning(f"Could not load biography.md: {e}. No biography context will be used.")
        BIOGRAPHY_CONTEXT = ""
else:
    BIOGRAPHY_CONTEXT = ""
    logger.info("Biography context DISABLED")