# wow-rag

A complete Retrieval-Augmented Generation (RAG) system using HuggingFace embeddings, Weaviate vector database, and GPT-4o for generation.

## Features
- Document ingestion from text and PDF files
- Chunking and embedding with HuggingFace models
- Storage and retrieval using Weaviate
- Question answering with GPT-4o
- Customizable chunk size, overlap, and embedding model

## Installation

### Prerequisites
- Python 3.8+
- [Weaviate](https://weaviate.io/) running locally or remotely
- OpenAI API key (for GPT-4o)

### Clone the repository
```bash
git clone <your-repo-url>
cd wow-rag
```

### Install dependencies
It is recommended to use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install dotenv langchain langchain_community sentence-transformers PyPDF2 chromadb
#pip install -r requirements.txt
```

#### Example `requirements.txt`
```
langchain
sentence-transformers
openai
PyPDF2
```

## Usage

See `rag_system.py` for example usage. Basic workflow:

```python
from rag_system import RAGSystem

rag = RAGSystem(
    weaviate_url="http://localhost:8080",
    openai_api_key="your-openai-api-key",
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Add documents
rag.add_documents([
    "Machine learning is a subset of artificial intelligence...",
    "Deep learning uses neural networks..."
], sources=["ml_doc", "dl_doc"])

# Query
response = rag.query("What is machine learning?")
print(response["answer"])
```

## Adding Documents from Files
```python
rag.add_documents_from_files(["path/to/file.txt", "path/to/file.pdf"])
```

## System Statistics
```python
stats = rag.get_stats()
print(stats)
```

## License
MIT
