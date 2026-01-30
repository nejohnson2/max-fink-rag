from __future__ import annotations

import os, json, hashlib, logging, argparse, re
from datetime import datetime, timezone

from typing import Dict, Any, Iterable, List, Optional, Tuple

# Prefer central app logger if present; else fall back to module logger
try:
    from config import logger  # type: ignore
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        logging.basicConfig(level=logging.INFO)

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# ---------------------------------------------------------------------------
# Text Normalization (LaTeX and Citations)
# ---------------------------------------------------------------------------

def normalize_latex_and_citations(text: str) -> str:
    """
    Normalize LaTeX escapes and citation formats in markdown text.

    This function cleans up:
    1. LaTeX math delimiters (inline $...$ and display $$...$$)
    2. Common LaTeX commands and escapes
    3. Citation formats (e.g., [1], (Author, Year), etc.)
    4. Special characters and escape sequences

    Args:
        text: Raw markdown text possibly containing LaTeX

    Returns:
        Cleaned text with LaTeX simplified and citations normalized
    """
    if not text:
        return text

    # Store the original for logging if needed
    result = text

    # --- LaTeX Math Handling ---

    # Remove display math blocks ($$...$$) - replace with placeholder or simplify
    # This handles multi-line display math
    result = re.sub(
        r'\$\$\s*(.+?)\s*\$\$',
        r'[mathematical expression: \1]',
        result,
        flags=re.DOTALL
    )

    # Remove inline math ($...$) - be careful not to match currency
    # Only match if there's actual LaTeX-like content (letters, commands, etc.)
    result = re.sub(
        r'\$([^$\n]+?)\$',
        lambda m: f'[math: {m.group(1)}]' if '\\' in m.group(1) or re.search(r'[a-zA-Z]{2,}', m.group(1)) else m.group(0),
        result
    )

    # --- Common LaTeX Commands ---

    # Remove \textbf{...}, \textit{...}, \emph{...} - keep content
    result = re.sub(r'\\text(?:bf|it|rm|sf|tt)\{([^}]*)\}', r'\1', result)
    result = re.sub(r'\\emph\{([^}]*)\}', r'\1', result)

    # Remove \cite{...} commands - replace with [citation]
    result = re.sub(r'\\cite[pt]?\{([^}]*)\}', r'[citation: \1]', result)

    # Remove \ref{...} and \label{...}
    result = re.sub(r'\\(?:ref|label|eqref)\{[^}]*\}', '', result)

    # Remove \begin{...} and \end{...} environment markers
    result = re.sub(r'\\(?:begin|end)\{[^}]*\}', '', result)

    # Common LaTeX symbols to readable text
    latex_symbols = {
        r'\\alpha': 'α', r'\\beta': 'β', r'\\gamma': 'γ', r'\\delta': 'δ',
        r'\\epsilon': 'ε', r'\\theta': 'θ', r'\\lambda': 'λ', r'\\mu': 'μ',
        r'\\pi': 'π', r'\\sigma': 'σ', r'\\omega': 'ω', r'\\phi': 'φ',
        r'\\psi': 'ψ', r'\\rho': 'ρ', r'\\tau': 'τ', r'\\chi': 'χ',
        r'\\pm': '±', r'\\times': '×', r'\\div': '÷', r'\\approx': '≈',
        r'\\neq': '≠', r'\\leq': '≤', r'\\geq': '≥', r'\\infty': '∞',
        r'\\sum': 'Σ', r'\\prod': 'Π', r'\\int': '∫',
        r'\\rightarrow': '→', r'\\leftarrow': '←', r'\\leftrightarrow': '↔',
        r'\\Rightarrow': '⇒', r'\\Leftarrow': '⇐',
        r'\\ldots': '...', r'\\cdots': '...', r'\\dots': '...',
        r'\\degree': '°', r'\\%': '%',
    }
    for latex_cmd, replacement in latex_symbols.items():
        result = re.sub(latex_cmd + r'(?![a-zA-Z])', replacement, result)

    # Remove remaining backslash commands (but not escaped characters)
    # This catches things like \hspace, \vspace, \newline, etc.
    result = re.sub(r'\\(?:hspace|vspace|newline|newpage|clearpage|pagebreak)\*?\{[^}]*\}', ' ', result)
    result = re.sub(r'\\(?:hspace|vspace|newline|newpage|clearpage|pagebreak)\*?', ' ', result)

    # Clean up common escape sequences
    result = result.replace(r'\\', ' ')  # Double backslash (LaTeX line break)
    result = result.replace(r'\&', '&')
    result = result.replace(r'\#', '#')
    result = result.replace(r'\$', '$')
    result = result.replace(r'\_', '_')
    result = result.replace(r'\{', '{')
    result = result.replace(r'\}', '}')

    # --- Citation Normalization ---

    # Normalize numbered citations [1], [1,2], [1-3] to [citation]
    result = re.sub(r'\[(\d+(?:\s*[,;-]\s*\d+)*)\]', r'[ref \1]', result)

    # Normalize author-year citations (Author, Year) or (Author et al., Year)
    result = re.sub(
        r'\(([A-Z][a-z]+(?:\s+(?:et\s+al\.?|and\s+[A-Z][a-z]+))?),?\s*(\d{4}[a-z]?)\)',
        r'[citation: \1 \2]',
        result
    )

    # --- Cleanup ---

    # Remove any remaining empty brackets or parentheses from removed content
    result = re.sub(r'\(\s*\)', '', result)
    result = re.sub(r'\[\s*\]', '', result)
    result = re.sub(r'\{\s*\}', '', result)

    # Normalize multiple spaces to single space
    result = re.sub(r'  +', ' ', result)

    # Normalize multiple newlines to max 2
    result = re.sub(r'\n{3,}', '\n\n', result)

    # Strip trailing whitespace from lines
    result = '\n'.join(line.rstrip() for line in result.split('\n'))

    return result


# ---------------------------------------------------------------------------
# Header-aware Markdown Chunking
# ---------------------------------------------------------------------------

# Regex to match markdown headers (# through ######)
HEADER_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def split_markdown_by_headers(text: str) -> List[Tuple[List[str], str]]:
    """
    Split markdown text by headers, tracking the header hierarchy.

    Returns a list of tuples: (header_stack, content)
    where header_stack is the list of parent headers leading to this section,
    and content is the text content of that section.

    Example:
        "# Title\nIntro\n## Section\nBody" ->
        [
            (["# Title"], "Intro"),
            (["# Title", "## Section"], "Body")
        ]
    """
    if not text or not text.strip():
        return []

    # Find all headers and their positions
    headers = []
    for match in HEADER_PATTERN.finditer(text):
        level = len(match.group(1))  # Number of # characters
        header_text = match.group(0)  # Full header line (e.g., "## Section")
        start = match.start()
        end = match.end()
        headers.append({
            "level": level,
            "header": header_text,
            "start": start,
            "end": end,
        })

    # If no headers found, return the whole text as one section
    if not headers:
        return [([], text.strip())]

    sections: List[Tuple[List[str], str]] = []
    header_stack: List[Tuple[int, str]] = []  # (level, header_text)

    # Handle content before the first header (if any)
    if headers[0]["start"] > 0:
        pre_content = text[:headers[0]["start"]].strip()
        if pre_content:
            sections.append(([], pre_content))

    # Process each header and its content
    for i, h in enumerate(headers):
        level = h["level"]
        header_text = h["header"]

        # Update header stack - pop headers at same or deeper level
        while header_stack and header_stack[-1][0] >= level:
            header_stack.pop()

        # Push current header
        header_stack.append((level, header_text))

        # Get content between this header and the next (or end of text)
        content_start = h["end"]
        content_end = headers[i + 1]["start"] if i + 1 < len(headers) else len(text)
        content = text[content_start:content_end].strip()

        # Only add if there's actual content (not just whitespace)
        if content:
            # Build the header hierarchy list
            hierarchy = [hdr for _, hdr in header_stack]
            sections.append((hierarchy, content))

    return sections


def build_header_aware_chunks(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    normalize_text: bool = True,
) -> List[str]:
    """
    Split markdown text into chunks that respect header boundaries.

    1. Optionally normalizes LaTeX and citations in the text
    2. First splits by headers to create logical sections
    3. Prepends the full header hierarchy to each section
    4. If a section exceeds chunk_size, splits it further with RecursiveCharacterTextSplitter

    Each chunk will start with its header hierarchy for context:
        "# Document Title\n## Section Name\n## Subsection\nActual content..."

    Args:
        text: The markdown text to split
        chunk_size: Target size for each chunk (default 800)
        chunk_overlap: Overlap between chunks when splitting large sections (default 100)
        normalize_text: Whether to normalize LaTeX and citations (default True)

    Returns:
        List of chunk strings, each with its header hierarchy prepended
    """
    # Normalize LaTeX and citations before splitting
    if normalize_text:
        text = normalize_latex_and_citations(text)

    sections = split_markdown_by_headers(text)

    if not sections:
        return [text.strip()] if text.strip() else []

    # Splitter for sections that are too large
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks: List[str] = []

    for header_stack, content in sections:
        # Build the header prefix
        header_prefix = "\n".join(header_stack)

        # Calculate available space for content after headers
        prefix_len = len(header_prefix) + 2 if header_prefix else 0  # +2 for \n\n
        available_size = chunk_size - prefix_len

        if available_size < 100:
            # Headers are very long, use minimum content size
            available_size = 100

        # Check if content fits in one chunk
        if len(content) <= available_size:
            if header_prefix:
                chunk_text = f"{header_prefix}\n\n{content}"
            else:
                chunk_text = content
            chunks.append(chunk_text)
        else:
            # Split large content into smaller pieces
            sub_chunks = recursive_splitter.split_text(content)
            for sub_chunk in sub_chunks:
                if header_prefix:
                    chunk_text = f"{header_prefix}\n\n{sub_chunk}"
                else:
                    chunk_text = sub_chunk
                chunks.append(chunk_text)

    return chunks


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


def _write_ingest_log(
    log_dir: str,
    *,
    jsonl_path: str,
    store_dir: str,
    chroma_collection: str,
    embeddings_model: str,
    child_chunk_size: int,
    child_chunk_overlap: int,
    excluded_titles: Optional[List[str]],
    total_lines: int,
    added_or_updated: int,
    skipped_no_text: int,
    skipped_unchanged: int,
    skipped_excluded: int,
    excluded_title_list: Optional[List[str]],
    child_chunks_created: int,
) -> str:
    """Write a JSON log file documenting this ingestion run.

    Returns the path to the written log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    filename = f"ingest_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    log_path = os.path.join(log_dir, filename)

    log_entry = {
        "timestamp": timestamp.isoformat(),
        "settings": {
            "source_file": os.path.abspath(jsonl_path),
            "store_dir": os.path.abspath(store_dir),
            "chroma_dir": os.path.abspath(os.path.join(store_dir, "chroma")),
            "parents_file": os.path.abspath(os.path.join(store_dir, "parents.jsonl")),
            "chroma_collection": chroma_collection,
            "embeddings_model": embeddings_model,
            "child_chunk_size": child_chunk_size,
            "child_chunk_overlap": child_chunk_overlap,
        },
        "exclusions": {
            "excluded_titles_config": excluded_titles or [],
            "excluded_documents": excluded_title_list or [],
            "excluded_count": skipped_excluded,
        },
        "results": {
            "total_lines_read": total_lines,
            "added_or_updated": added_or_updated,
            "skipped_no_text": skipped_no_text,
            "skipped_unchanged": skipped_unchanged,
            "skipped_excluded": skipped_excluded,
            "child_chunks_created": child_chunks_created,
        },
    }

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_entry, f, indent=2, ensure_ascii=False)

    logger.info("Ingestion log written to %s", log_path)
    return log_path


def build_child_chunks(
    parent_docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    normalize_text: bool = True,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Split each parent document into smaller "child" chunks for retrieval.

    Uses header-aware markdown splitting:
    1. Splits by markdown headers to create logical sections
    2. Prepends full header hierarchy to each chunk for context
    3. Further splits large sections with RecursiveCharacterTextSplitter

    Each child chunk gets:
      - id       = f"{parent_id}:{chunk_index}"
      - text     = chunk text with header hierarchy prepended
      - metadata = copy of parent metadata (including parent_id)

    Returns:
        ids, texts, metadatas
    """
    ids: List[str] = []
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []

    for p in parent_docs:
        pid = p.metadata["parent_id"]
        chunks = build_header_aware_chunks(
            p.page_content,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            normalize_text=normalize_text,
        )
        for i, chunk in enumerate(chunks):
            ids.append(f"{pid}:{i}")
            texts.append(chunk)
            metadatas.append(dict(p.metadata))

    logger.info(
        "Built %d child chunks from %d parent docs (chunk_size=%d, overlap=%d)",
        len(texts),
        len(parent_docs),
        chunk_size,
        chunk_overlap,
    )
    return ids, texts, metadatas


def ingest_jsonl(
    jsonl_path: str,
    *,
    store_dir: str = "./fink_archive",
    chroma_collection: str = "rag_collection",
    embeddings_model: str = "BAAI/bge-small-en-v1.5",
    child_chunk_size: int = 800,
    child_chunk_overlap: int = 100,
    excluded_titles: Optional[List[str]] = None,
    normalize_latex: bool = True,
    only_items: Optional[List[str]] = None,
) -> None:
    """
    Ingest a JSONL file of records and index them into Chroma + parents.jsonl.

    - Reads JSONL line by line.
    - For each record with non-empty markdown_text:
        * Computes stable parent_id and fingerprint.
        * Skips records that haven't changed since last run.
        * Skips records whose title matches any excluded_titles (case-insensitive).
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
        excluded_titles: Optional list of document titles to exclude from ingestion.
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

    logger.info(
        "Using header-aware markdown chunking (chunk_size=%d, overlap=%d)",
        child_chunk_size,
        child_chunk_overlap,
    )
    logger.info("LaTeX/citation normalization: %s", "enabled" if normalize_latex else "disabled")
    if only_items:
        logger.info("FILTERING: Only ingesting items: %s", only_items)

    parent_docs: List[Document] = []
    total_lines = 0
    skipped_no_text = 0
    skipped_unchanged = 0
    skipped_excluded = 0
    skipped_not_in_only = 0
    added_or_updated = 0
    excluded_title_list: List[str] = []  # Track actual titles that were excluded

    # Normalize excluded titles to lowercase for case-insensitive matching
    excluded_titles_lower = []
    if excluded_titles:
        excluded_titles_lower = [title.lower() for title in excluded_titles]
        logger.info("Excluding documents with titles: %s", excluded_titles)

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
                skipped_no_text += 1
                continue

            # If only_items is specified, skip any item not in the list
            if only_items:
                item_id = rec.get("item_id", "")
                if item_id not in only_items:
                    skipped_not_in_only += 1
                    continue

            # Check if this document's title should be excluded
            title = rec.get("Title", "")
            if excluded_titles_lower and title and title.lower() in excluded_titles_lower:
                skipped_excluded += 1
                excluded_title_list.append(title)
                logger.debug("Skipping excluded document: %s", title)
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
        "skipped_no_text=%d, skipped_unchanged=%d, skipped_excluded=%d, skipped_not_in_only=%d",
        total_lines,
        added_or_updated,
        skipped_no_text,
        skipped_unchanged,
        skipped_excluded,
        skipped_not_in_only,
    )
    log_dir = os.path.join(store_dir, "logs")
    log_kwargs = dict(
        log_dir=log_dir,
        jsonl_path=jsonl_path,
        store_dir=store_dir,
        chroma_collection=chroma_collection,
        embeddings_model=embeddings_model,
        child_chunk_size=child_chunk_size,
        child_chunk_overlap=child_chunk_overlap,
        excluded_titles=excluded_titles,
        total_lines=total_lines,
        added_or_updated=added_or_updated,
        skipped_no_text=skipped_no_text,
        skipped_unchanged=skipped_unchanged,
        skipped_excluded=skipped_excluded,
        excluded_title_list=excluded_title_list,
    )

    if not parent_docs:
        logger.info("No new or updated parent docs to ingest; nothing to do.")
        _write_ingest_log(**log_kwargs, child_chunks_created=0)
        return

    ids, texts, metadatas = build_child_chunks(
        parent_docs,
        chunk_size=child_chunk_size,
        chunk_overlap=child_chunk_overlap,
        normalize_text=normalize_latex,
    )

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
    _write_ingest_log(**log_kwargs, child_chunks_created=len(texts))
    logger.info("Ingestion complete for %s", jsonl_path)

def main() -> None:
    """
    CLI entry point for ingestion.

    Usage:
        # Basic usage (outputs to ./fink_archive by default)
        python app/ingest.py --jsonl-path data/compiled_dataset.jsonl

        # With custom output directory
        python app/ingest.py --jsonl-path data/compiled_dataset.jsonl --store-dir ./custom_dir
    """
    parser = argparse.ArgumentParser(description="Ingest a JSONL file into Chroma + parents.jsonl")
    parser.add_argument(
        "--jsonl-path",
        required=True,
        help="Path to the input JSONL file (e.g. compiled_dataset.jsonl)",
    )
    parser.add_argument(
        "--store-dir",
        default="./fink_archive",
        help="Directory where Chroma and parents.jsonl will be written (default: ./fink_archive)",
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
        default=1000,
        help="Child chunk size in characters (default: 1000)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Child chunk overlap in characters (default: 100)",
    )
    parser.add_argument(
        "--exclude-titles",
        nargs="*",
        help="List of document titles to exclude from ingestion (case-insensitive)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable LaTeX and citation normalization (default: normalization enabled)",
    )
    parser.add_argument(
        "--only-items",
        nargs="*",
        help="Only ingest specific item IDs (e.g., --only-items item_6761 item_6762)",
    )

    args = parser.parse_args()

    ingest_jsonl(
        jsonl_path=args.jsonl_path,
        store_dir=args.store_dir,
        chroma_collection=args.collection,
        embeddings_model=args.embeddings_model,
        child_chunk_size=args.chunk_size,
        child_chunk_overlap=args.chunk_overlap,
        excluded_titles=args.exclude_titles,
        normalize_latex=not args.no_normalize,
        only_items=args.only_items,
    )

if __name__ == "__main__":
    main()