from __future__ import annotations

import os, json, hashlib, logging, argparse

from typing import Dict, Any, Iterable, List, Optional, Tuple

# Prefer central app logger if present; else fall back to module logger
try:
    from config import logger  # type: ignore
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

#from langchain.schema import Document
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

#from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def stable_parent_id(rec: Dict[str, Any]) -> str:
    """
    Derive a stable parent_id for a record.

    Priority:
      1. Existing identifiers in the data (item_id, Identifier, item_url, filenames).
      2. Fallback: SHA1 hash of the full record JSON.

    This keeps IDs stable across ingestion runs so we can update / de-duplicate.
    """
    for key in ["item_id", "Identifier", "item_url", "pdf_filename", "markdown_filename"]:
        v = rec.get(key)
        if v:
            return str(v)
    raw = json.dumps(rec, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def record_fingerprint(rec: Dict[str, Any]) -> str:
    """
    Compute a fingerprint for a record based on:
      - item_id, Identifier, item_url
      - markdown_text content

    This lets us skip re-embedding unchanged records on subsequent runs.
    """
    base = {
        "item_id": rec.get("item_id"),
        "Identifier": rec.get("Identifier"),
        "item_url": rec.get("item_url"),
        "markdown_text": rec.get("markdown_text", ""),
    }
    raw = json.dumps(base, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()


def load_parent_store(path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load the parent store from parents.jsonl.

    Structure:
      {
        parent_id: {
          "parent_id": ...,
          "fingerprint": ...,
          "text": ...,
          "metadata": {...}
        },
        ...
      }
    """    
    store: Dict[str, Dict[str, Any]] = {}
    if not os.path.exists(path):
        logger.info("No existing parent store at %s; starting fresh", path)
        return store
    
    logger.info("Loading existing parent store from %s", path)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            store[rec["parent_id"]] = rec
    logger.info("Loaded %d parent entries from %s", len(store), path)
    return store


def write_parent_store(path: str, store: Dict[str, Dict[str, Any]]) -> None:
    """
    Atomically write the parent store back to disk as JSONL.
    """    
    tmp = path + ".tmp"
    logger.info("Writing parent store with %d entries to %s", len(store), path)
    with open(tmp, "w", encoding="utf-8") as f:
        for rec in store.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    logger.info("Parent store written to %s", path)


# def infer_doc_type(rec: Dict[str, Any]) -> str:
#     """
#     Optional, but useful for routing.
#     Use Collection/Tags/Subject as hints.
#     """
#     collection = (rec.get("Collection") or "").strip().lower()
#     tags = (rec.get("Tags") or "").strip().lower()
#     subject = (rec.get("Subject") or "").strip().lower()
#     title = (rec.get("Title") or "").strip().lower()

#     text = " ".join([collection, tags, subject, title])

#     if "bibliograph" in text:
#         return "bibliographic"
#     if "correspond" in text:
#         return "correspondence"
#     if "research" in text or "publication" in text or "paper" in text:
#         return "published_research"
#     if "bio" in text or "biograph" in text:
#         return "bio"
#     return "other"


def build_child_chunks(
    parent_docs: List[Document],
    child_splitter: RecursiveCharacterTextSplitter,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Split each parent document into smaller "child" chunks for retrieval.

    Each child chunk gets:
      - id       = f"{parent_id}:{chunk_index}"
      - text     = chunk text
      - metadata = copy of parent metadata (including parent_id)

    Returns:
        ids, texts, metadatas
    """
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for p in parent_docs:
        pid = p.metadata["parent_id"]
        chunks = child_splitter.split_text(p.page_content)
        for i, chunk in enumerate(chunks):
            ids.append(f"{pid}:{i}")
            texts.append(chunk)
            metadatas.append(dict(p.metadata))

    logger.info(
        "Built %d child chunks from %d parent docs",
        len(texts),
        len(parent_docs),
    )
    return ids, texts, metadatas


def ingest_jsonl(
    jsonl_path: str,
    *,
    store_dir: str = "./rag_store",
    chroma_collection: str = "rag_collection",
    embeddings_model: str = "BAAI/bge-small-en-v1.5",
    child_chunk_size: int = 300,
    child_chunk_overlap: int = 40,
) -> None:
    """
    Ingest a JSONL file of records and index them into Chroma + parents.jsonl.

    - Reads JSONL line by line.
    - For each record with non-empty markdown_text:
        * Computes stable parent_id and fingerprint.
        * Skips records that haven't changed since last run.
        * Normalizes some metadata (title, date, collection, tags, item_url).
        * Accumulates parent Documents for any new or changed records.
    - Splits parents into smaller child chunks and upserts them into Chroma.
    - Updates parents.jsonl with full parent text and metadata.

    Args:
        jsonl_path: Path to the compiled_dataset.jsonl (or similar).
        store_dir: Root directory for Chroma + parents.jsonl.
        chroma_collection: Name of the Chroma collection.
        embeddings_model: HuggingFace model name for embeddings.
        child_chunk_size: Character size of each retrieval chunk.
        child_chunk_overlap: Character overlap between consecutive chunks.
    """
    logger.info("Starting ingestion from %s", jsonl_path)
    logger.info("store_dir=%s, chroma_collection=%s", store_dir, chroma_collection)

    if not os.path.exists(jsonl_path):
        logger.error("JSONL file not found: %s", jsonl_path)
        raise FileNotFoundError(jsonl_path)
    
    os.makedirs(store_dir, exist_ok=True)
    chroma_dir = os.path.join(store_dir, "chroma")
    os.makedirs(chroma_dir, exist_ok=True)

    parents_path = os.path.join(store_dir, "parents.jsonl")
    parent_store = load_parent_store(parents_path)

    logger.info("Initializing embeddings model: %s", embeddings_model)

    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
    logger.info(
        "Connecting to Chroma collection '%s' at %s",
        chroma_collection,
        chroma_dir,
    )
    vs = Chroma(
        collection_name=chroma_collection,
        embedding_function=embeddings,
        persist_directory=chroma_dir,
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
    )

    parent_docs: List[Document] = []
    total_lines = 0
    skipped_no_text = 0
    skipped_unchanged = 0
    added_or_updated = 0

    logger.info("Reading JSONL records from %s", jsonl_path)
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)

            text = rec.get("markdown_text", "")
            if not text or not text.strip():
                continue

            pid = stable_parent_id(rec)
            fp = record_fingerprint(rec)

            existing = parent_store.get(pid)
            if existing and existing.get("fingerprint") == fp:
                skipped_unchanged += 1
                continue

            #doc_type = infer_doc_type(rec)

            # Keep metadata mostly as-is but normalize a few fields for retrieval/UI
            metadata = dict(rec)
            metadata["parent_id"] = pid
            metadata["fingerprint"] = fp
            #metadata["doc_type"] = doc_type

            # It can help to standardize a few lower_snake_case duplicates
            metadata["title"] = rec.get("Title")
            metadata["date"] = rec.get("Date")
            metadata["collection"] = rec.get("Collection")
            metadata["tags"] = rec.get("Tags")
            metadata["item_url"] = rec.get("item_url")

            parent_docs.append(Document(page_content=text, metadata=metadata))
            added_or_updated += 1

            # Persist parent doc so the runtime system can map parent_id -> full text + metadata
            parent_store[pid] = {
                "parent_id": pid,
                "fingerprint": fp,
                "text": text,
                "metadata": metadata,
            }
    logger.info(
        "Finished scanning JSONL: total_lines=%d, added_or_updated=%d, "
        "skipped_no_text=%d, skipped_unchanged=%d",
        total_lines,
        added_or_updated,
        skipped_no_text,
        skipped_unchanged,
    )
    if not parent_docs:
        logger.info("No new or updated parent docs to ingest; nothing to do.")
        return

    ids, texts, metadatas = build_child_chunks(parent_docs, child_splitter)

    # Upsert child chunks into Chroma with stable ids
    # Chroma has a max batch size; so we chunk the inputs
    BATCH_SIZE = 5000  # must be <= max batch size (5461 from error)
    logger.info(
        "Upserting %d child chunks into Chroma in batches of %d",
        len(texts),
        BATCH_SIZE,
    )
    for start in range(0, len(texts), BATCH_SIZE):
        end = start + BATCH_SIZE
        batch_texts = texts[start:end]
        batch_metas = metadatas[start:end]
        batch_ids = ids[start:end]
        logger.debug("Upserting batch %d:%d", start, end)
        vs.add_texts(texts=batch_texts, metadatas=batch_metas, ids=batch_ids)
    logger.info("Persisting Chroma collection to disk")
    vs.persist()

    write_parent_store(parents_path, parent_store)
    logger.info("Ingestion complete for %s", jsonl_path)

def main() -> None:
    """
    CLI entry point for ingestion.

    Usage:
        python ingest.py --jsonl-path /path/to/file.jsonl --store-dir /path/to/store
    """
    parser = argparse.ArgumentParser(description="Ingest a JSONL file into Chroma + parents.jsonl")
    parser.add_argument(
        "--jsonl-path",
        required=True,
        help="Path to the input JSONL file (e.g. compiled_dataset.jsonl)",
    )
    parser.add_argument(
        "--store-dir",
        required=True,
        help="Directory where Chroma and parents.jsonl will be written (e.g. ./fink_archive)",
    )
    parser.add_argument(
        "--collection",
        default="rag_collection",
        help="Chroma collection name (default: rag_collection)",
    )
    parser.add_argument(
        "--embeddings-model",
        default="BAAI/bge-small-en-v1.5",
        help="HuggingFace embeddings model name",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=300,
        help="Child chunk size in characters (default: 300)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=40,
        help="Child chunk overlap in characters (default: 40)",
    )

    args = parser.parse_args()

    ingest_jsonl(
        jsonl_path=args.jsonl_path,
        store_dir=args.store_dir,
        chroma_collection=args.collection,
        embeddings_model=args.embeddings_model,
        child_chunk_size=args.chunk_size,
        child_chunk_overlap=args.chunk_overlap,
    )

if __name__ == "__main__":
    main()