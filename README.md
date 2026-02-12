# Max Fink RAG System

A Retrieval-Augmented Generation (RAG) application for exploring the Max Fink Digital Collection. This system provides an intelligent chat interface that answers questions about Max Fink's life, work, and contributions using documents from the Stony Brook University Libraries Special Collections.

**Tech Stack:** Flask + LangChain + ChromaDB + Ollama (remote) + HuggingFace Embeddings

---

## Features

### Core Capabilities
- **Intent-Based Retrieval**: Rules-based pattern matching classifies queries to determine which collections to search
  - Biographical queries: Always searches biographical files
  - Research queries: Also searches published works and research files
  - Correspondence queries: Also searches correspondence collection
- **Hybrid Retrieval System**:
  - Dense vector search using `BAAI/bge-small-en-v1.5` embeddings
  - Sparse BM25 search for keyword matching
  - Ensemble fusion using Reciprocal Rank Fusion (RRF)
- **Header-Aware Markdown Chunking**:
  - Splits documents at markdown header boundaries
  - Prepends header hierarchy to each chunk for context
  - LaTeX and citation normalization for cleaner text
- **Biographical Guarantee**: At least one biographical chunk is always included in the context
- **Advanced Reranking**: Cross-encoder reranking with `BAAI/bge-reranker-base`
- **Remote LLM Integration**: Uses Ollama for answer generation
- **Session Management**: Maintains independent sessions per browser tab
- **Source Citations**: Every answer includes links to original source materials
- **Interaction Logging**: All queries logged in JSONL format for analytics

### User Interface
- Modern chat interface with markdown rendering
- Animated thinking indicator with cycling archive-themed messages
- Clickable source references with expandable titles
- Responsive design for mobile and desktop

---

## Architecture

### Intent-Based Hybrid RAG Pipeline

```
Query → Intent Classification
              ↓
┌─────────────────────────────────────────────────┐
│  BIOGRAPHICAL RETRIEVAL (always runs)           │
│  ┌─────────────┐    ┌─────────────┐            │
│  │   Chroma    │    │    BM25     │            │
│  │  (dense)    │    │  (sparse)   │            │
│  └──────┬──────┘    └──────┬──────┘            │
│         └────────┬─────────┘                    │
│                  ↓                              │
│         EnsembleRetriever                       │
│         (RRF weights: 0.5/0.5)                  │
└─────────────────────────────────────────────────┘
              +
┌─────────────────────────────────────────────────┐
│  SUPPLEMENTAL RETRIEVAL (based on intent)       │
│  Same hybrid structure (Chroma + BM25)          │
│  - research → Published Works, Research Files   │
│  - correspondence → Correspondence              │
│  - biographical → (no supplemental)             │
└─────────────────────────────────────────────────┘
              ↓
      Combine all documents
              ↓
      Cross-encoder reranking
              ↓
      Biographical guarantee check
              ↓
      Answer generation (LLM)
```

### RAG System (`app/rag_system.py`)

The main RAG implementation is the `RAGSystem` class:

```python
class RAGSystem:
    # Collection configuration
    BIOGRAPHICAL_COLLECTION = "Biographical Files"
    SUPPLEMENTAL_COLLECTIONS = {
        "biographical": [],  # No supplemental
        "research": ["Published Works", "Research Files and Unpublished Works"],
        "correspondence": ["Correspondence"],
    }
    MIN_BIOGRAPHICAL_CHUNKS = 1  # Guaranteed minimum

    def __init__(
        self,
        store_dir: str = "./fink_archive",
        chroma_collection: str = "rag_collection",
        embeddings_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        enable_bm25: bool = True,
        k_recall: int = 40,
        k_ensemble: int = 20,
        k_after_rerank: int = 6,
    )
```

**Key Methods:**
- `ask(question, chat_session_id, excluded_parent_ids)`: Main query interface
- `classify_intent(question)`: Classifies query as biographical/research/correspondence
- `cleanup_session(session_id)`: Clean up when sessions end
- `log_interaction(...)`: Log queries for analytics

### Data Ingestion (`app/ingest.py`)

Batch ingestion script with header-aware chunking:

```python
def ingest_jsonl(
    jsonl_path: str,
    store_dir: str = "./fink_archive",
    chroma_collection: str = "rag_collection",
    embeddings_model: str = "BAAI/bge-small-en-v1.5",
    child_chunk_size: int = 1000,
    child_chunk_overlap: int = 200,
    excluded_titles: Optional[List[str]] = None,
    normalize_latex: bool = True,
    only_items: Optional[List[str]] = None,
)
```

**Features:**
- **Header-aware markdown splitting**: Chunks break at header boundaries
- **LaTeX normalization**: Converts LaTeX math and symbols to readable text
- **Citation normalization**: Standardizes citation formats
- **Incremental updates**: Only processes changed documents
- **Stable document IDs**: Uses content fingerprints
- **Title-based exclusion**: Filter out specific documents
- **Item filtering**: Ingest only specific item IDs

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

# RAG System Configuration
ENABLE_MULTI_QUERY=false      # Multi-query expansion (adds latency)
ENABLE_PARENT_CHILD=false     # Return parent docs instead of chunks
DEBUG_RETRIEVAL=false         # Verbose logging of retrieval steps

# Deployment Configuration
URL_PREFIX=                   # Set to "/max.fink" for proxy deployment

# Logging Configuration
CHAT_LOG_PATH=logs/chat_interactions.jsonl
```

5. **Initialize data directories**
```bash
mkdir -p fink_archive logs
```

---

## Quick Start Workflow

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

**Important:** Both the ingest script and the app must use the same `store_dir` (`./fink_archive` by default).

---

## Usage

### Running the Application

**Development Mode:**
```bash
python app/app.py
```

The app will start on `http://0.0.0.0:5067` by default.

**With Debug Logging:**
```bash
DEBUG_RETRIEVAL=true python app/app.py
```

This shows detailed logs for each query including:
- Intent classification results
- Retrieval timing for each collection
- Document counts and sources
- Biographical guarantee checks

**Production Deployment:**

For production, use a WSGI server like Gunicorn:

```bash
gunicorn -w 4 -b 0.0.0.0:5067 app.app:app
```

### Ingesting Documents

**Basic usage:**
```bash
python app/ingest.py --jsonl-path data/compiled_dataset.jsonl
```

**With custom chunking:**
```bash
python app/ingest.py \
  --jsonl-path data/compiled_dataset.jsonl \
  --chunk-size 1000 \
  --chunk-overlap 200
```

**Exclude specific documents:**
```bash
python app/ingest.py \
  --jsonl-path data/compiled_dataset.jsonl \
  --exclude-titles "CV" "Resume" "Draft Document"
```

**Ingest only specific items:**
```bash
python app/ingest.py \
  --jsonl-path data/compiled_dataset.jsonl \
  --only-items item_6761 item_6762
```

**Disable LaTeX normalization:**
```bash
python app/ingest.py \
  --jsonl-path data/compiled_dataset.jsonl \
  --no-normalize
```

**JSONL Format:**

Each line should be a JSON object with at least:
```json
{
  "markdown_text": "# Document Title\n\nDocument content here...",
  "Title": "Document Title",
  "Date": "2024-01-01",
  "collection": "Biographical Files",
  "item_url": "https://example.com/item",
  "item_id": "item_123",
  "pdf_filename": "document.pdf"
}
```

The `collection` field is used for intent-based routing:
- `"Biographical Files"` - Always searched
- `"Published Works"` - Searched for research queries
- `"Research Files and Unpublished Works"` - Searched for research queries
- `"Correspondence"` - Searched for correspondence queries

### Programmatic Usage

```python
from app.rag_system import RAGSystem

# Initialize the RAG system
rag = RAGSystem(
    store_dir="./fink_archive",
    chroma_collection="rag_collection",
    enable_bm25=True,
    k_recall=40,
    k_ensemble=20,
    k_after_rerank=6,
)

# Ask a question
result = rag.ask(
    question="What was Max Fink's contribution to ECT research?",
    chat_session_id="user_session_123"
)

print(result['answer'])
print(result['sources'])
print(result['_metadata']['intent'])  # biographical, research, or correspondence
```

---

## Configuration

### System Prompts

The RAG assistant's behavior is controlled by prompts in [app/prompts.py](app/prompts.py):

- `SYSTEM_PROMPT`: Main system prompt defining assistant personality
- `INTENT_CLASSIFICATION_PROMPT`: Classifies queries into biographical/research/correspondence
- `ALTERNATIVE_PROMPTS`: Pre-defined alternatives (scholarly, conversational, educational)

**To customize:**
```python
# In app/prompts.py
SYSTEM_PROMPT = ALTERNATIVE_PROMPTS["scholarly"]  # More formal tone
```

### RAG System Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k_recall` | 40 | Documents retrieved from each collection |
| `k_ensemble` | 20 | Documents after ensemble fusion |
| `k_after_rerank` | 6 | Final documents after cross-encoder reranking |
| `enable_bm25` | True | Enable hybrid retrieval (Chroma + BM25) |
| `MIN_BIOGRAPHICAL_CHUNKS` | 1 | Guaranteed biographical docs in context |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_URL` | - | Ollama server URL |
| `OLLAMA_MODEL` | - | Model name (e.g., `llama3.1:latest`) |
| `OLLAMA_API_KEY` | - | API key for authenticated endpoints |
| `ENABLE_MULTI_QUERY` | false | Generate query variants for better recall |
| `ENABLE_PARENT_CHILD` | false | Return full parent documents |
| `DEBUG_RETRIEVAL` | false | Verbose retrieval logging |
| `URL_PREFIX` | "" | URL prefix for proxy deployment |
| `CHAT_LOG_PATH` | logs/chat_interactions.jsonl | Query log file path |

### Ingestion Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--chunk-size` | 1000 | Target chunk size in characters |
| `--chunk-overlap` | 200 | Overlap between chunks |
| `--exclude-titles` | - | Titles to exclude from ingestion |
| `--no-normalize` | false | Disable LaTeX/citation normalization |
| `--only-items` | - | Only ingest specific item IDs |

---

## API Endpoints

### `POST /query`

Query the RAG system.

**Request:**
```json
{
  "question": "What was Max Fink's educational background?",
  "session_id": "session_12345",
  "excluded_parent_ids": ["item_6794"]
}
```

**Response:**
```json
{
  "answer": "Max Fink received his medical degree from...",
  "sources": [
    {
      "parent_id": "item_123",
      "source": "https://example.com/files/document.pdf",
      "title": "Biography",
      "collection": "Biographical Files",
      "text": "Chunk content used..."
    }
  ]
}
```

### `POST /cleanup_session`

Clean up session when tab closes.

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
│   ├── remote_ollama.py       # Custom LLM for remote Ollama
│   ├── test_performance.py    # Performance profiling utilities
│   ├── static/
│   │   ├── css/
│   │   │   └── chat_interface.css
│   │   ├── js/
│   │   │   └── main.js        # Frontend chat interface
│   │   └── images/
│   └── templates/
│       └── index.html         # Main chat interface
├── notebooks/
│   ├── 01_sample_outputs.ipynb        # Overview of logged data
│   ├── 02_session_explorer.ipynb      # Session-level exploration
│   └── 03_retrieval_relevance.ipynb   # Query-source relevance analysis
├── scripts/
│   └── analyze_logs.py        # Utility for analyzing chat logs
├── fink_archive/              # RAG data storage (gitignored)
│   ├── chroma/                # Vector database
│   ├── parents.jsonl          # Parent document metadata
│   └── logs/                  # Ingestion logs
├── logs/                      # Application logs (gitignored)
├── .env                       # Environment configuration (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Development

### Debug Mode

Enable detailed logging to see what's happening:

```bash
DEBUG_RETRIEVAL=true python app/app.py
```

Output includes:
- Configuration summary at startup
- Intent classification for each query
- Retrieval counts per collection (biographical vs supplemental)
- Timing breakdown (intent, retrieval, reranking, answer generation)
- Document details including titles and relevance scores

### Performance Testing

```bash
python app/test_performance.py
```

Profiles:
- System initialization time
- Intent classification time
- Query processing time
- Cache performance

### Analyze Chat Logs

```bash
# View statistics
python scripts/analyze_logs.py logs/chat_interactions.jsonl

# Export to CSV
python scripts/analyze_logs.py logs/chat_interactions.jsonl --csv output.csv
```

Statistics include:
- Total interactions and unique sessions
- Intent distribution (biographical/research/correspondence)
- Average timing metrics
- Sources per query

### Jupyter Notebooks

Two notebooks in `notebooks/` provide interactive exploration of chat logs:

- **`01_sample_outputs.ipynb`** - High-level overview: schema inspection, intent distribution, timing metrics, most-cited sources, activity over time
- **`02_session_explorer.ipynb`** - Session deep dives: full conversation threads, per-session timing, source analysis, cross-session comparison, keyword search
- **`03_retrieval_relevance.ipynb`** - Measures how well retrieved sources relate to queries using TF-IDF similarity, semantic embedding similarity (BGE), and keyword overlap, with qualitative side-by-side inspection of the weakest and strongest retrievals

#### Syncing Logs from a Remote Server

If the application is running on a remote machine, use `rsync` to copy the log file locally before opening the notebooks:

```bash
# One-time sync
rsync -avz user@remote-host:/path/to/max-fink-rag/logs/chat_interactions.jsonl ./logs/

# Subsequent syncs (only transfers new data)
rsync -avz user@remote-host:/path/to/max-fink-rag/logs/chat_interactions.jsonl ./logs/

# With a custom SSH port
rsync -avz -e 'ssh -p 2222' user@remote-host:/path/to/max-fink-rag/logs/chat_interactions.jsonl ./logs/

# Dry run to preview what will be transferred
rsync -avzn user@remote-host:/path/to/max-fink-rag/logs/chat_interactions.jsonl ./logs/
```

Once the file is synced, open the notebooks and run from the `notebooks/` directory — they reference `../logs/chat_interactions.jsonl` by relative path.

### Adding New Features

1. **Modify RAG pipeline**: Edit `app/rag_system.py`
2. **Update ingestion**: Edit `app/ingest.py`
3. **Change UI**: Edit `app/templates/index.html` and `app/static/js/main.js`
4. **Add routes**: Edit `app/app.py`
5. **Customize prompts**: Edit `app/prompts.py`

---

## Troubleshooting

### Data Directory Issues

**Error: Collection not found / No documents retrieved**
- Check that `./fink_archive/` exists with `chroma/` and `parents.jsonl`
- Verify you ran `ingest.py` from the project root
- Ensure `app.py` and `ingest.py` use the same `store_dir`

### ChromaDB Issues

**Error: Dimension mismatch**
- You changed the embeddings model without recreating the database
- Solution: Delete `./fink_archive/chroma/` and re-ingest

### Ollama Issues

**Error: Connection refused**
- Check Ollama is running: `ollama list`
- Verify `OLLAMA_URL` in `.env`

**Error: Model not found**
- Pull the model: `ollama pull llama3.1:latest`

### Performance Issues

**Slow queries**
- Reduce `k_recall`, `k_ensemble`, or `k_after_rerank`
- Disable BM25: `enable_bm25=False`
- Increase chunk sizes to reduce total chunks

---

## Data Privacy

This system logs queries to `logs/chat_interactions.jsonl` including:
- User questions and system answers
- Retrieved sources with full text
- Intent classification
- Performance metrics
- Session IDs (not personally identifiable)

To disable logging, remove `RAGSystem.log_interaction()` in `app/app.py`.

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
