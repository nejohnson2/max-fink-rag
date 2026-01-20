# Max Fink RAG System

A Retrieval-Augmented Generation (RAG) application for exploring the Max Fink Digital Collection. This system provides an intelligent chat interface that answers questions about Max Fink's life, work, and contributions using documents from the Stony Brook University Libraries Special Collections.

**Tech Stack:** Flask + LangChain + ChromaDB + Ollama (remote) + HuggingFace Embeddings

---

## Features

### Core Capabilities
- **Intelligent Document Search**: Query the Max Fink digital archive using natural language questions
- **Hybrid Retrieval System**:
  - Dense vector search using `BAAI/bge-small-en-v1.5` embeddings
  - Sparse BM25 search for keyword matching
  - Ensemble fusion of both retrieval methods
- **Parent-Child Chunking**:
  - Small "child" chunks (300 chars) for precise retrieval
  - Larger "parent" documents (1000+ chars) for coherent context
- **Advanced Reranking**: Cross-encoder reranking with `BAAI/bge-reranker-base`
- **Remote LLM Integration**: Uses Ollama for answer generation with conversation history
- **Session Management**: Maintains chat history per browser session
- **Source Citations**: Every answer includes links to original source materials

### User Interface
- Modern chat interface with markdown rendering
- Real-time typing indicators
- Clickable source references
- Responsive design for mobile and desktop

---

## Architecture

### RAG Pipeline (`app/rag_system.py`)

The main RAG implementation is the `RAGSystem` class:

```python
class RAGSystem:
    def __init__(
        self,
        store_dir: str = "./fink_archive",
        chroma_collection: str = "rag_collection",
        embeddings_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        enable_bm25: bool = True,
        k_recall: int = 15,
        k_ensemble: int = 10,
        k_after_rerank: int = 6,
    )
```

**Key Methods:**
- `ask(question, chat_session_id, excluded_parent_ids)`: Main query interface
- `cleanup_session(session_id)`: Clean up chat history when sessions end
- `log_interaction(...)`: Log queries for analytics and improvement

### Data Ingestion (`app/ingest.py`)

Batch ingestion script for processing JSONL datasets:

```python
def ingest_jsonl(
    jsonl_path: str,
    store_dir: str = "./rag_store",
    chroma_collection: str = "rag_collection",
    embeddings_model: str = "BAAI/bge-small-en-v1.5",
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 40,
    excluded_titles: Optional[List[str]] = None,
)
```

**Features:**
- Incremental updates (only processes changed documents)
- Stable document IDs using content fingerprints
- Title-based exclusion filtering
- Maintains `parents.jsonl` for full document text lookup

---

## Installation

### Prerequisites
- Python 3.10+ (tested with 3.13)
- Ollama instance (local or remote)
- ~2GB disk space for models and embeddings

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd max-fink-rag
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**

Create a `.env` file in the project root:

```bash
# Ollama Configuration
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:latest
OLLAMA_API_KEY=your-api-key-if-needed

# Optional: RAG System Configuration
ENABLE_MULTI_QUERY=false

# Optional: Deployment Configuration
URL_PREFIX=

# Optional: Logging Configuration
CHAT_LOG_PATH=logs/chat_interactions.jsonl
```

5. **Initialize data directories**
```bash
mkdir -p fink_archive logs
```

---

## Quick Start Workflow

Once installed, follow these steps to get the system running:

1. **Prepare your data**: Create a JSONL file with your documents (see JSONL Format below)

2. **Ingest documents** (run from project root):
   ```bash
   python app/ingest.py --jsonl-path data/compiled_dataset.jsonl
   ```
   This creates `./fink_archive/` with the vector database

3. **Start the application**:
   ```bash
   python app/app.py
   ```
   The app will automatically load data from `./fink_archive/`

4. **Access the interface**: Open `http://localhost:5067` in your browser

**Important:** Both the ingest script and the app must use the same `store_dir` (`./fink_archive` by default). They are already configured to work together out of the box.

---

## Usage

### Running the Application

**Development Mode:**
```bash
python app/app.py
```

The app will start on `http://0.0.0.0:5067` by default.

**Production Deployment:**

For production, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5067 app.app:app
```

### Ingesting Documents

The ingestion script processes JSONL files and creates the vector database and metadata files that the application uses.

**Markdown-Aware Processing:**
The system uses `MarkdownTextSplitter` which respects markdown structure when creating chunks. This means:
- Chunks break at natural boundaries (headers, paragraphs, lists)
- Related content stays together (a header with its content)
- Code blocks and tables remain intact when possible
- Better context preservation for the LLM

**Important:** Run the script from the project root directory (not from inside the `app/` directory) so that the output directory `./fink_archive` is created in the correct location.

**Basic usage (recommended):**
```bash
# Run from project root - outputs to ./fink_archive by default
python app/ingest.py --jsonl-path data/compiled_dataset.jsonl
```

This will create:
```
./fink_archive/
├── chroma/              # Vector database
│   └── [ChromaDB files]
└── parents.jsonl        # Parent document metadata
```

**With additional options:**
```bash
# Customize chunking parameters
python app/ingest.py \
  --jsonl-path data/compiled_dataset.jsonl \
  --chunk-size 300 \
  --chunk-overlap 40
```

**Exclude specific documents by title:**
```bash
python app/ingest.py \
  --jsonl-path data/compiled_dataset.jsonl \
  --exclude-titles "CV" "Resume" "Draft Document"
```

**Use a custom output directory:**
```bash
# Only needed if you want to store data elsewhere
python app/ingest.py \
  --jsonl-path data/compiled_dataset.jsonl \
  --store-dir ./custom_archive

# Make sure to update app.py to use the same directory
```

**JSONL Format:**

Each line should be a JSON object with at least:
```json
{
  "markdown_text": "Document content here...",
  "Title": "Document Title",
  "Date": "2024-01-01",
  "Collection": "Max Fink Papers",
  "item_url": "https://example.com/item",
  "item_id": "item_123"
}
```

### Programmatic Usage

```python
from app.rag_system import RAGSystem

# Initialize the RAG system
rag = RAGSystem(
    store_dir="./fink_archive",
    chroma_collection="rag_collection",
    enable_bm25=True,
    k_recall=15,
    k_ensemble=10,
    k_after_rerank=6,
)

# Ask a question
result = rag.ask(
    question="What was Max Fink's contribution to ECT research?",
    chat_session_id="user_session_123"
)

print(result['answer'])
print(result['sources'])
```

---

## Configuration

### System Prompts

The RAG assistant's behavior is controlled by prompts defined in [app/prompts.py](app/prompts.py). You can customize the assistant without touching the core code.

**To modify the assistant's behavior:**

1. Edit [app/prompts.py](app/prompts.py)
2. Modify the `SYSTEM_PROMPT` variable
3. Restart the application

**Available prompts:**
- `SYSTEM_PROMPT`: Main system prompt for the RAG assistant (defines personality and behavior)
- `INTENT_CLASSIFICATION_PROMPT`: Prompt for classifying user queries by intent
- `ALTERNATIVE_PROMPTS`: Pre-defined alternative prompts you can use (scholarly, conversational, educational)

**Example - Switching to a scholarly tone:**
```python
# In app/prompts.py, replace SYSTEM_PROMPT with:
SYSTEM_PROMPT = ALTERNATIVE_PROMPTS["scholarly"]
```

Changes take effect immediately on restart (no code recompilation needed).

### RAG System Parameters

- `k_recall`: Number of documents to retrieve initially (default: 15)
- `k_ensemble`: Number of documents after ensemble fusion (default: 10)
- `k_after_rerank`: Final number of documents after reranking (default: 6)
- `enable_bm25`: Enable BM25 sparse retrieval (default: True)
- `child_chunk_size`: Size of retrieval chunks in characters (default: 300)
- `child_chunk_overlap`: Overlap between chunks (default: 40)

### Text Chunking

The ingestion process uses **markdown-aware text splitting** via LangChain's `MarkdownTextSplitter`. This intelligently splits documents based on markdown structure rather than just character count.

**Benefits:**
- Preserves document hierarchy (headers stay with their content)
- Respects semantic boundaries (paragraphs, lists, code blocks)
- Improves retrieval quality by keeping related information together
- Better context for the LLM to understand and answer questions

**How it works:**
1. Text is split at markdown structural elements (headers, blank lines, etc.)
2. Chunks are sized to fit within `child_chunk_size` while respecting boundaries
3. Overlaps between chunks preserve context continuity
4. The original markdown syntax is preserved (bold, links, etc.)

### Data Directory Structure

The system stores all persistent data in the `./fink_archive/` directory (relative to project root):

```
./fink_archive/
├── chroma/              # ChromaDB vector database
│   ├── chroma.sqlite3   # Metadata database
│   └── [UUID]/          # Vector embeddings
└── parents.jsonl        # Full text of parent documents (for retrieval)
```

**Configuration:**
- **Collection name**: `rag_collection` (set in both `app.py` and `ingest.py`)
- **Embeddings model**: `BAAI/bge-small-en-v1.5` (set in `ingest.py`)
- **Store directory**: `./fink_archive` (default in both `app.py` and `ingest.py`)

**Important Notes:**
1. The `app.py` and `ingest.py` must use the **same `store_dir` and `chroma_collection`** values
2. If you change the embeddings model, delete the existing ChromaDB directory and re-ingest:
   ```bash
   rm -rf ./fink_archive/chroma
   python app/ingest.py --jsonl-path data/compiled_dataset.jsonl
   ```
3. The `fink_archive/` directory is gitignored - your data won't be committed to the repository

---

## API Endpoints

### `POST /query`

Query the RAG system.

**Request:**
```json
{
  "question": "What was Max Fink's educational background?",
  "session_id": "session_12345",
  "excluded_parent_ids": ["item_6794"]  // optional
}
```

**Response:**
```json
{
  "answer": "Max Fink received his medical degree from...",
  "sources": [
    {
      "source": "https://example.com/item/123",
      "title": "Biography",
      "collection": "Max Fink Papers"
    }
  ]
}
```

### `POST /cleanup_session`

Clean up chat history for a session.

**Request:**
```
session_id=session_12345
```

---

## Project Structure

```
max-fink-rag/
├── app/
│   ├── app.py                 # Flask application and routes
│   ├── config.py              # Configuration and environment variables
│   ├── prompts.py             # System prompts (easily editable!)
│   ├── rag_system.py          # Main RAG implementation
│   ├── ingest.py              # Batch document ingestion script
│   ├── remote_ollama.py       # Custom LLM for remote Ollama with headers
│   ├── test_performance.py    # Performance profiling utilities
│   ├── static/
│   │   ├── css/
│   │   │   └── chat_interface.css
│   │   ├── js/
│   │   │   └── main.js        # Frontend chat interface logic
│   │   └── images/
│   └── templates/
│       └── index.html         # Main chat interface
├── scripts/
│   └── analyze_logs.py        # Utility for analyzing chat logs
├── archive/                   # Archived old code (gitignored)
├── fink_archive/              # RAG data storage (gitignored)
│   ├── chroma/                # Vector database
│   └── parents.jsonl          # Parent document metadata
├── logs/                      # Application logs (gitignored)
├── .env                       # Environment configuration (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Development

### Useful Scripts

**Performance Testing:**
```bash
python app/test_performance.py
```

This will profile:
- System initialization time
- Intent classification time
- Query processing time
- Cache performance

**Analyze Chat Logs:**
```bash
# View statistics and recent queries
python scripts/analyze_logs.py logs/chat_interactions.jsonl

# Export to CSV for detailed analysis
python scripts/analyze_logs.py logs/chat_interactions.jsonl --csv output.csv
```

This utility provides:
- Total interactions and unique sessions
- Intent distribution statistics
- Average timing metrics (retrieval, answer, total)
- Average sources per query
- Recent query history

### Customizing the Assistant

**Modify System Prompts:**

The easiest way to customize the assistant's behavior is to edit [app/prompts.py](app/prompts.py):

```python
# app/prompts.py

# Main system prompt - edit this to change assistant behavior
SYSTEM_PROMPT = """You are a helpful research assistant...
[Your custom instructions here]
"""

# Or use one of the pre-defined alternatives
SYSTEM_PROMPT = ALTERNATIVE_PROMPTS["scholarly"]  # More formal
SYSTEM_PROMPT = ALTERNATIVE_PROMPTS["conversational"]  # More friendly
```

After editing, restart the app:
```bash
python app/app.py
```

### Adding New Features

1. **Modify RAG pipeline**: Edit `app/rag_system.py`
2. **Update ingestion**: Edit `app/ingest.py`
3. **Change UI**: Edit `app/templates/index.html` and `app/static/js/main.js`
4. **Add routes**: Edit `app/app.py`
5. **Customize prompts**: Edit `app/prompts.py` (no code changes needed!)

### Testing Changes

After making changes, test with:
```bash
# Start the app
python app/app.py

# In another terminal, test a query
curl -X POST http://localhost:5067/query \
  -H "Content-Type: application/json" \
  -d '{"question": "test question", "session_id": "test"}'
```

---

## Troubleshooting

### Data Directory Issues

**Error: Collection not found / No documents retrieved**
- **Cause**: The app can't find the vector database
- **Solution**:
  1. Check that `./fink_archive/` exists and contains `chroma/` and `parents.jsonl`
  2. Verify you ran `ingest.py` from the project root (not from inside `app/`)
  3. Ensure `app.py` and `ingest.py` use the same `store_dir` (default: `./fink_archive`)

**Error: Store directory not in expected location**
- If you ran `python ingest.py` from inside the `app/` directory, it created `./app/fink_archive/` instead of `./fink_archive/`
- **Solution**: Move the directory to the correct location:
  ```bash
  mv app/fink_archive ./fink_archive
  ```

### ChromaDB Issues

**Error: Dimension mismatch**
- **Cause**: You changed the embeddings model without recreating the database
- **Solution**: Delete `./fink_archive/chroma/` and re-ingest:
  ```bash
  rm -rf ./fink_archive/chroma
  python app/ingest.py --jsonl-path data/compiled_dataset.jsonl
  ```

**Error: Collection not found**
- **Cause**: The collection name in `app.py` doesn't match the one used during ingestion
- **Solution**: Ensure both use `rag_collection` (the default)

### Ollama Issues

**Error: Connection refused**
- Check that Ollama is running: `ollama list`
- Verify `OLLAMA_URL` in `.env` is correct

**Error: Model not found**
- Pull the model: `ollama pull llama3.1:latest`

### Performance Issues

**Slow queries**
- Reduce `k_recall`, `k_ensemble`, or `k_after_rerank`
- Disable BM25: `enable_bm25=False`
- Use a smaller embeddings model
- Increase chunk sizes to reduce total chunks

---

## Data Privacy

This system logs all queries and responses to `logs/chat_interactions.jsonl` for analytics and improvement. The logs include:
- User questions
- System answers
- Retrieved sources
- Performance metrics
- Session IDs (not personally identifiable)

To disable logging, remove the `RAGSystem.log_interaction()` call in `app/app.py`.

---

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## License

MIT License - See LICENSE file for details

---

## Acknowledgments

- **Max Fink Digital Collection**: Stony Brook University Libraries Special Collections
- **Embeddings**: BAAI BGE models
- **LLM**: Ollama/Llama models
- **Vector DB**: ChromaDB
- **Framework**: LangChain
