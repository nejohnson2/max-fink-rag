# Max Fink RAG

Retrieval-Augmented Generation (RAG) app for exploring the work of Max Fink.  
Backend: Flask + LangChain + ChromaDB + Ollama (remote).  
Frontend: Web chat interface with markdown-rendered responses and PDF upload.

---

## Features

- Upload one or more PDFs and index them into ChromaDB.
- Hybrid retrieval:
  - Dense vector search with `BAAI/bge-small-en-v1.5`.
  - Sparse BM25 search over the same chunks.
  - Ensemble fusion of both signals.
- Parent/child chunking:
  - Small “child” chunks for retrieval.
  - “Parent” documents preserved for coherent context.
- Cross-encoder reranking with `BAAI/bge-reranker-base`.
- Remote Ollama LLM for answering questions with context.
- Chat history per session via LangChain’s `RunnableWithMessageHistory`.
- Markdown support in the chat UI (for bot responses).

---

## Architecture Overview

### RAG Pipeline (`rag_system_v2.py`)

The main RAG implementation lives in:

```python
# filepath: app/rag_system_v2.py
class RAGSystem:
    def __init__(
        self,
        persist_dir: str = "./chroma_rag",
        ollama_model: str = "llama3.1:latest",
        embeddings_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        k_recall: int = 30,
        k_ensemble: int = 20,
        k_after_rerank: int = 6,
        child_chunk_size: int = 300,
        child_chunk_overlap: int = 40,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        chroma_collection: str = "rag_collection",
    ):
        ...
```

Key parts:

- **Indexing**:
  - `index_documents(docs, clear_existing=False)`:  
    - Normalizes inputs into parent `Document`s with a stable `parent_id`.
    - Splits parents into smaller “child” chunks using `child_splitter`.
    - Builds a Chroma vector store on child docs (dense embeddings).
    - Builds a BM25 retriever over the same child docs (sparse).
    - Stores a `_parent_lookup` map (`parent_id -> parent Document`).

- **Incremental updates**:
  - `add_documents(docs)`: Adds new docs into existing Chroma + BM25, and updates `_parent_lookup`.

- **PDF ingestion**:
  - `add_pdfs(pdf_paths, metadata_list=None)`:  
    - Uses `PyPDF2` to extract text from each PDF.
    - Wraps each PDF as a parent `Document` with `metadata["source"] = pdf_path`.
    - Delegates to `add_documents(...)`.

- **Question answering**:
  - `ask(question: str, chat_session_id: str = "default") -> str`:
    1. Vector retriever (`Chroma.as_retriever`) over child chunks.
    2. BM25 retriever over child chunks.
    3. `EnsembleRetriever` to fuse results (`k_ensemble`).
    4. Map child hits to parent docs via `_ParentFromChildRetriever`.
    5. `MultiQueryRetriever` to expand the query using the LLM.
    6. `ContextualCompressionRetriever` with a cross-encoder reranker.
    7. Build final `context` string from top-k documents and call the LLM with history.

---

## Requirements

Install Python dependencies (inside a virtualenv is recommended):

```bash
pip install -r requirements.txt
```

You’ll need:

- Python 3.10+ (project currently using 3.13 in `.venv`).
- ChromaDB
- LangChain & langchain-community extras
- `langchain_ollama` (or `langchain_community` ChatOllama fallback)
- PyPDF2
- Flask
- `sentence-transformers` models (HuggingFaceEmbeddings and cross-encoder)

---

## Configuration

The RAG system reads Ollama config from `config.py`:

```python
# filepath: app/config.py (example)
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.1:latest"
OLLAMA_API_KEY = "your-ollama-api-key-if-needed"
```

Adjust `OLLAMA_URL`, `OLLAMA_MODEL`, and `OLLAMA_API_KEY` for your environment (e.g., a remote Ollama instance).

---

## Running the App

From the project root:

```bash
export FLASK_APP=app.app
export FLASK_ENV=development  # optional, for debug

flask run
```

This will:

- Start the Flask server.
- Mount routes defined in `app/app.py`:
  - `/` or `/home` – main chat UI.
  - `/upload` – upload form for PDFs.
  - `/upload_pdfs` – POST endpoint to handle PDF uploads and indexing.
  - `/query` – chat query endpoint.

`app.config['MAX_CONTENT_LENGTH']` is set to `500 MB`, so large PDFs are accepted at the HTTP layer, but be mindful of memory and indexing time.

---

## Using the RAG System Programmatically

Example: indexing PDFs and asking questions from a Python shell:

```python
from langchain.schema import Document
from app.rag_system_v2 import RAGSystem

rag = RAGSystem()

# Index one or more PDFs
rag.add_pdfs([
    "uploads/max_fink_paper1.pdf",
    "uploads/max_fink_paper2.pdf",
])

# Ask a question
answer = rag.ask("What did Max Fink say about the use of ECT in depression?")
print(answer)
```

If you want to index arbitrary text:

```python
docs = [
    Document(page_content="Some text about Max Fink...", metadata={"title": "Notes A"}),
    Document(page_content="More text...", metadata={"title": "Notes B"}),
]
rag.index_documents(docs, clear_existing=True)
```

---

## Frontend: Chat Interface

The chat UI lives under `app/templates` and `app/static`.

- Input element: `#questionInput` (CSS in `app/static/css/chat_interface.css`).
- Bot messages are rendered with markdown (via `marked.js`) in `app/static/js/main.js`:
  - For bot messages, `innerHTML = marked.parse(content)` is used.
  - For user messages, plain text is used.

If you want to adjust the size of the chat input:

```css
/* filepath: app/static/css/chat_interface.css */
.chat-input-container {
    max-width: 900px;      /* widen the whole input area */
    width: 100%;
}

#questionInput {
    width: 100%;           /* fills the container above */
    height: 48px;          /* adjust for taller input */
    font-size: 16px;
}
```

---

## ChromaDB Storage

- Default persistence directory: `./chroma_rag`.
- Collection name: `rag_collection`.
- The embedding model is `BAAI/bge-small-en-v1.5`, so the Chroma collection is configured with that embedding dimension.  
  If you change the embeddings model, delete the existing Chroma directory:

```bash
rm -rf ./chroma_rag
```

and re-index documents to avoid dimension mismatch errors.

---

## Large Files and Git

Uploaded PDFs are stored under `uploads/`.  
These can easily exceed GitHub’s 100 MB limit, so you should:

- **Ignore them in Git**:

```bash
echo "uploads/*.pdf" >> .gitignore
git add .gitignore
git commit -m "Ignore uploaded PDFs"
```

- Or use Git LFS if you truly need them in the repo.

---

## Notes

- `RAGSystem` in `rag_system_v2.py` is the newer, hybrid + reranked implementation.  
  If `rag_system.py` still exists, consider migrating routes and usage to the v2 system and deprecating the old class.
- Error like `Request Entity Too Large` during indexing usually indicate:
  - Very large PDFs, or
  - Extremely large chunks or many chunks at once.  
  Adjust `child_chunk_size`, `parent_chunk_size`, or limit number of documents per batch if needed.

---

## License

MIT