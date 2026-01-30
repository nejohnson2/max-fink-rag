from __future__ import annotations

"""Retrieval-Augmented Generation (RAG) system for the Max Fink archive.

This module implements a multi-retriever architecture for searching archival
collections with guaranteed biographical context:

Architecture Overview:
    1. Intent Classification: LLM classifies query to determine supplemental needs
    2. Multi-Retriever Execution:
       - Biographical retriever ALWAYS runs (searches "Biographical Files")
       - Supplemental retriever runs conditionally based on intent:
         * 'research' → Published Works, Research Files
         * 'correspondence' → Correspondence
         * 'none' → biographical only
    3. Cross-Encoder Reranking: Combined results reranked by relevance
    4. Biographical Guarantee: At least MIN_BIOGRAPHICAL_CHUNKS in final selection
    5. Answer Generation: LLM generates response from selected context

Components:
    - Chroma vector store (semantic similarity search)
    - Optional BM25 keyword search (lexical retrieval for exact terms)
    - Cross-encoder reranking (BAAI/bge-reranker-base)
    - Parent/child document reconstruction (optional, per intent)

The public entrypoint is `RAGSystem.ask()`, which returns an answer plus source
metadata suitable for displaying citations/links in the UI.
"""

import os
import json
import sys
import time
from typing import List, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ConfigDict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# Retrievers moved to langchain_classic in v1 packaging
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever

# CrossEncoderReranker path can vary by release; keep resilient fallbacks
try:
    from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
except Exception:  # pragma: no cover
    try:
        # Optional fallback if `langchain_classic` isn't present.
        from langchain.retrievers.document_compressors import CrossEncoderReranker  # type: ignore
    except Exception:  # pragma: no cover
        # Last resort: implement rerank manually (see note below)
        CrossEncoderReranker = None  # type: ignore

#from remote_ollama import RemoteOllamaLLM
#from config import OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_URL
from remote_ollama import RemoteOllamaLLM
from config import (
    OLLAMA_API_KEY,
    OLLAMA_MODEL,
    OLLAMA_URL,
    ENABLE_MULTI_QUERY,
    ENABLE_INTENT_CLASSIFICATION,
    ENABLE_PARENT_CHILD,
    DEBUG_RETRIEVAL,
    logger,
    SYSTEM_PROMPT,
    INTENT_CLASSIFICATION_PROMPT,
)

class _ParentFromChildRetriever(BaseRetriever):
    """Retriever adapter that expands child chunks to full parent documents.

    Used when ENABLE_PARENT_CHILD is True. The vector store contains small
    child chunks for precise semantic matching, but the LLM often benefits
    from seeing the full parent document for better context.

    Process:
    1. Invoke the underlying child_retriever to get chunk matches
    2. Extract parent_id from each chunk's metadata
    3. Look up full parent documents in parent_lookup dict
    4. Deduplicate parents (multiple chunks may share a parent)
    5. Apply collection_filter if specified

    Falls back to returning original child chunks if no parents found.

    Note:
        Currently used selectively based on USE_PARENT_DOCUMENTS_BY_INTENT.
        Large documents (e.g., biographical PDFs) may exceed context limits.
    """

    child_retriever: BaseRetriever
    parent_lookup: Dict[str, Document]
    collection_filter: Optional[List[str]] = None
    excluded_parent_ids: Optional[List[str]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        # LangChain retrievers have been moving from `.get_relevant_documents()`
        # to `.invoke()`; we use `.invoke()` for forward compatibility.
        child_hits: List[Document] = self.child_retriever.invoke(query)
        seen = set()
        parents: List[Document] = []
        for d in child_hits:
            pid = (d.metadata or {}).get("parent_id")
            if pid and pid in self.parent_lookup and pid not in seen:
                parent_doc = self.parent_lookup[pid]

                # Apply collection filter if specified
                if self.collection_filter:
                    doc_collection = parent_doc.metadata.get("collection")
                    if doc_collection not in self.collection_filter:
                        continue

                # Apply parent ID exclusion filter if specified
                if self.excluded_parent_ids and pid in self.excluded_parent_ids:
                    continue

                seen.add(pid)
                parents.append(parent_doc)
        return parents if parents else child_hits


class _MetadataFilterRetriever(BaseRetriever):
    """Post-retrieval metadata filter for non-Chroma retrievers.

    Chroma supports native metadata filtering, but BM25 and ensemble retrievers
    do not. This wrapper applies the same filters after retrieval to ensure
    consistent results across retriever types.

    Filters supported:
    - allowed_collections: Only include docs from these collections
    - doc_type: Only include docs with this doc_type value
    - excluded_parent_ids: Exclude docs with these parent IDs

    Note:
        Applied after retrieval, so may return fewer than k results.
        Primarily used for BM25/ensemble alignment with Chroma filters.
    """

    child_retriever: BaseRetriever
    allowed_collections: Optional[List[str]] = None
    doc_type: Optional[str] = None
    excluded_parent_ids: Optional[List[str]] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        docs: List[Document] = self.child_retriever.invoke(query)
        if not self.allowed_collections and not self.doc_type and not self.excluded_parent_ids:
            return docs

        filtered: List[Document] = []
        for d in docs:
            md = d.metadata or {}
            if self.allowed_collections:
                collection = md.get("collection")
                if collection not in self.allowed_collections:
                    continue
            if self.doc_type:
                if md.get("doc_type") != self.doc_type:
                    continue
            if self.excluded_parent_ids:
                parent_id = md.get("parent_id")
                if parent_id in self.excluded_parent_ids:
                    continue
            filtered.append(d)

        return filtered


def _load_parent_lookup(parents_jsonl_path: str) -> Dict[str, Document]:
    """Load parent documents from JSONL into an in-memory lookup dictionary.

    Parent documents are the full original documents before chunking. They're
    stored separately from Chroma (which only holds child chunks) to enable
    chunk→parent expansion when needed.

    Args:
        parents_jsonl_path: Path to parents.jsonl file

    Returns:
        Dict mapping parent_id → Document with full text and metadata

    Expected JSONL record format (one per line):
        {"parent_id": "...", "text": "...", "metadata": {...}}
    """
    lookup: Dict[str, Document] = {}
    if not os.path.exists(parents_jsonl_path):
        return lookup

    with open(parents_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            pid = rec["parent_id"]
            text = rec["text"]
            md = rec.get("metadata", {}) or {}
            md["parent_id"] = pid
            lookup[pid] = Document(page_content=text, metadata=md)

    return lookup


class RAGSystem:
    """End-to-end RAG pipeline with multi-retriever architecture.

    This system implements a two-tier retrieval strategy:
    1. Primary (always runs): Biographical collection for foundational context
    2. Supplemental (intent-based): Research or correspondence collections

    The pipeline ensures biographical information is always represented in the
    final context sent to the LLM, while allowing intent classification to
    dynamically add relevant supplemental materials.

    Pipeline Steps:
        1. Intent Classification → determines supplemental collections
        2. Biographical Retrieval → always executes (k_recall chunks)
        3. Supplemental Retrieval → executes if intent requires it
        4. Cross-Encoder Reranking → orders combined results by relevance
        5. Biographical Guarantee → ensures MIN_BIOGRAPHICAL_CHUNKS in top-k
        6. Answer Generation → LLM produces response from selected context

    Attributes:
        BIOGRAPHICAL_COLLECTION: Collection name that is always searched
        SUPPLEMENTAL_COLLECTIONS: Mapping of intent → collection names
        MIN_BIOGRAPHICAL_CHUNKS: Minimum biographical docs in final selection
        USE_PARENT_DOCUMENTS_BY_INTENT: Whether to expand chunks to parents
    """

    # ---------------------------------------------------------------------------
    # Collection Configuration
    # ---------------------------------------------------------------------------

    # The biographical collection is ALWAYS searched, regardless of intent.
    # This ensures foundational context about Max Fink is always available.
    BIOGRAPHICAL_COLLECTION = "Biographical Files"

    # Supplemental collections are searched based on LLM intent classification.
    # The intent classifier returns one of these keys based on the query.
    SUPPLEMENTAL_COLLECTIONS = {
        "research": ["Published Works", "Research Files and Unpublished Works"],
        "correspondence": ["Correspondence"],
        "none": [],  # No supplemental search needed
    }

    # Minimum biographical chunks guaranteed in final selection after reranking.
    # If cross-encoder ranks all biographical chunks below this threshold,
    # the system will replace the lowest-ranked supplemental chunks to meet
    # this minimum.
    MIN_BIOGRAPHICAL_CHUNKS = 1

    # Controls whether to expand child chunks to full parent documents.
    # Large biographical documents can exceed context limits, so we default
    # to False for most intents. Correspondence documents are typically
    # smaller and benefit from full context.
    USE_PARENT_DOCUMENTS_BY_INTENT = {
        "research": False,       # Research docs can be very large
        "correspondence": True,  # Letters are typically short
        "none": False,           # Biographical only, keep chunks
    }

    def __init__(
        self,
        store_dir: str = "./fink_archive",
        chroma_collection: str = "rag_collection",
        embeddings_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        k_recall: int = 30,
        k_ensemble: int = 20,
        k_after_rerank: int = 6,
        enable_bm25: bool = True,
    ):
        """Initialize the RAG system components.

        Args:
            store_dir: Directory containing Chroma index and parents.jsonl
            chroma_collection: Name of the Chroma collection to query
            embeddings_model: HuggingFace model for semantic embeddings
            reranker_model: Cross-encoder model for reranking
            k_recall: Number of chunks to retrieve from each collection
            k_ensemble: Number of docs for ensemble retrieval (if BM25 enabled)
            k_after_rerank: Final number of chunks after cross-encoder reranking
            enable_bm25: Whether to enable BM25 lexical search (for exact terms)

        Initializes:
            - Sentence-transformer embeddings for Chroma similarity search
            - Chroma vector store connection
            - Parent document lookup (for chunk→parent expansion)
            - BM25 retriever (optional, for keyword matching)
            - Cross-encoder reranker (for relevance scoring)
            - LLM connection (for intent classification and answer generation)
            - Prompt template with system instructions
        """
        self.store_dir = store_dir
        self.k_recall = k_recall
        self.k_ensemble = k_ensemble
        self.k_after_rerank = k_after_rerank
        self.enable_bm25 = enable_bm25
        self._embeddings_model = embeddings_model
        self._reranker_model = reranker_model
        self._chroma_collection = chroma_collection

        # Persisted artifacts live under `store_dir`:
        # - `chroma/` holds the vector index
        # - `parents.jsonl` maps parent_id → full parent text/metadata
        chroma_dir = os.path.join(store_dir, "chroma")
        parents_path = os.path.join(store_dir, "parents.jsonl")

        if DEBUG_RETRIEVAL:
            logger.info("=" * 80)
            logger.info("DEBUG MODE: RAG System Initialization")
            logger.info("=" * 80)
            logger.info("Storage Configuration:")
            logger.info("  Store directory: %s", os.path.abspath(store_dir))
            logger.info("  Chroma directory: %s", os.path.abspath(chroma_dir))
            logger.info("  Parents JSONL: %s", os.path.abspath(parents_path))
            logger.info("-" * 40)

        # Sentence-transformer style embeddings used by Chroma similarity search.
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vs = Chroma(
            collection_name=chroma_collection,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir,
        )

        # Parent documents are stored separately so we can retrieve chunks but
        # present the larger parent to the model/UI.
        self._parent_lookup = _load_parent_lookup(parents_path)

        self.bm25: Optional[BM25Retriever] = None
        if enable_bm25:
            # BM25 needs the raw documents in-memory.
            self.bm25 = self._build_bm25_from_chroma()
            self.bm25.k = self.k_recall

        # Cross-encoder reranker is used by ContextualCompressionRetriever.
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)

        if CrossEncoderReranker is None:
            raise ImportError(
                "CrossEncoderReranker could not be imported. "
                "Either install a compatible langchain_classic build, or switch to manual reranking."
            )

        self.compressor = CrossEncoderReranker(
            model=self.cross_encoder,
            top_n=self.k_after_rerank,
        )

        # LLM endpoint used for:
        # - intent classification
        # - multi-query expansion
        # - final answering
        self.llm = RemoteOllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        )

        # Prompt template: system instructions + question + retrieved context
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", "Question: {question}\n\nArchival Context:\n{context}"),
        ])

        self._answer_chain = self.prompt | self.llm | StrOutputParser()

        # Print comprehensive debug info about RAG system structure
        if DEBUG_RETRIEVAL:
            self._print_debug_initialization()

    def _build_bm25_from_chroma(self) -> BM25Retriever:
        """Build a BM25 retriever from whatever is currently stored in Chroma."""
        col = self.vs._collection
        data = col.get(include=["documents", "metadatas"])
        docs: List[Document] = []
        for text, md in zip(data.get("documents", []), data.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=md or {}))
        return BM25Retriever.from_documents(docs)

    def _create_collection_retriever(
        self,
        collections: List[str],
        k: int,
        excluded_parent_ids: Optional[List[str]] = None,
        doc_type: Optional[str] = None,
    ) -> BaseRetriever:
        """Create a Chroma retriever filtered to specific collections.

        This factory method creates retrievers for the multi-retriever pipeline.
        Each retriever searches a subset of the archive (biographical or
        supplemental collections) using Chroma's metadata filtering.

        Args:
            collections: List of collection names to search (e.g., ["Biographical Files"])
            k: Number of chunks to retrieve
            excluded_parent_ids: Parent IDs to exclude (for follow-up queries)
            doc_type: Optional additional filter on doc_type metadata

        Returns:
            A configured Chroma retriever with collection and exclusion filters
        """
        # Build Chroma filter: collection must be in the allowed list
        search_kwargs: Dict[str, object] = {"k": k}
        search_kwargs["filter"] = {"collection": {"$in": collections}}

        # Exclude specific parent IDs (useful for "tell me more" follow-ups)
        if excluded_parent_ids:
            search_kwargs["filter"]["parent_id"] = {"$nin": excluded_parent_ids}

        # Optional doc_type filter (e.g., filter to only PDF documents)
        if doc_type:
            search_kwargs["filter"]["doc_type"] = doc_type

        return self.vs.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

    def _ensure_biographical_chunk(
        self,
        reranked_docs: List[Document],
        biographical_docs: List[Document],
        k_final: int,
    ) -> List[Document]:
        """Guarantee biographical representation in the final document selection.

        The cross-encoder reranks all documents (biographical + supplemental) by
        relevance to the query. However, for archival questions about Max Fink,
        we always want some biographical context. This method ensures at least
        MIN_BIOGRAPHICAL_CHUNKS biographical documents appear in the final selection.

        Algorithm:
        1. Mark each document with is_biographical flag based on collection
        2. Take top k_final documents from reranked results
        3. Count biographical documents in selection
        4. If below minimum, replace lowest-ranked supplemental docs with
           highest-ranked biographical docs that didn't make the cut

        Args:
            reranked_docs: All documents after cross-encoder reranking (ordered by score)
            biographical_docs: Original biographical chunks (fallback candidates)
            k_final: Target number of documents for final selection

        Returns:
            List of k_final documents with guaranteed MIN_BIOGRAPHICAL_CHUNKS
            biographical representation
        """
        # Mark documents with their source type
        for doc in reranked_docs:
            md = doc.metadata or {}
            md["is_biographical"] = md.get("collection") == self.BIOGRAPHICAL_COLLECTION

        # Take top k_final
        top_docs = reranked_docs[:k_final]

        # Count biographical chunks in selection
        bio_count = sum(1 for d in top_docs if (d.metadata or {}).get("is_biographical", False))

        if DEBUG_RETRIEVAL:
            logger.info("📊 Biographical guarantee check:")
            logger.info("   Top %d docs have %d biographical chunks", k_final, bio_count)
            logger.info("   Minimum required: %d", self.MIN_BIOGRAPHICAL_CHUNKS)

        # If we have enough biographical chunks, return as-is
        if bio_count >= self.MIN_BIOGRAPHICAL_CHUNKS:
            if DEBUG_RETRIEVAL:
                logger.info("   ✓ Requirement met, no adjustment needed")
            return top_docs

        # Need to insert biographical chunks
        needed = self.MIN_BIOGRAPHICAL_CHUNKS - bio_count

        # Find biographical chunks from the reranked results that aren't already in top_docs
        top_doc_ids = {id(d) for d in top_docs}
        bio_candidates = [
            d for d in reranked_docs
            if (d.metadata or {}).get("is_biographical", False) and id(d) not in top_doc_ids
        ]

        # If no bio candidates in reranked, fall back to original biographical_docs
        if not bio_candidates and biographical_docs:
            # Use the first biographical docs (they were already retrieved, just not reranked high)
            bio_candidates = biographical_docs[:needed]

        if DEBUG_RETRIEVAL:
            logger.info("   ⚠️ Need to insert %d biographical chunk(s)", needed)
            logger.info("   Available biographical candidates: %d", len(bio_candidates))

        # Insert biographical chunks, replacing lowest-ranked supplemental docs
        result = list(top_docs)
        inserted = 0

        for bio_doc in bio_candidates[:needed]:
            # Find the lowest-ranked non-biographical doc to replace
            for i in range(len(result) - 1, -1, -1):
                if not (result[i].metadata or {}).get("is_biographical", False):
                    replaced = result[i]
                    result[i] = bio_doc
                    inserted += 1
                    if DEBUG_RETRIEVAL:
                        replaced_title = (replaced.metadata or {}).get("Title", "Unknown")[:30]
                        bio_title = (bio_doc.metadata or {}).get("Title", "Unknown")[:30]
                        logger.info("   Replaced '%s...' with biographical '%s...'", replaced_title, bio_title)
                    break

            if inserted >= needed:
                break

        if DEBUG_RETRIEVAL:
            final_bio_count = sum(1 for d in result if (d.metadata or {}).get("is_biographical", False))
            logger.info("   Final biographical count: %d", final_bio_count)

        return result

    def _print_debug_initialization(self) -> None:
        """Print comprehensive debug information about the RAG system structure.

        Called during initialization when DEBUG_RETRIEVAL is enabled.
        Outputs configuration details including:
        - Model names and endpoints
        - Retrieval parameters (k values)
        - Feature flags (intent classification, parent/child, etc.)
        - Index statistics (chunk counts by collection)
        - Multi-retriever architecture configuration
        """
        # Get Chroma collection stats
        try:
            col = self.vs._collection
            col_count = col.count()
        except Exception:
            col_count = "unknown"

        # Count parent documents
        parent_count = len(self._parent_lookup)

        # Analyze collections in the data
        collections_found = {}
        try:
            data = self.vs._collection.get(include=["metadatas"])
            for md in data.get("metadatas", []):
                if md:
                    coll = md.get("collection", "Unknown")
                    collections_found[coll] = collections_found.get(coll, 0) + 1
        except Exception:
            pass

        logger.info("Models & Embeddings:")
        logger.info("  Embeddings model: %s", self._embeddings_model)
        logger.info("  Reranker model: %s", self._reranker_model)
        logger.info("  LLM model: %s", OLLAMA_MODEL)
        logger.info("  LLM endpoint: %s", OLLAMA_URL)
        logger.info("-" * 40)
        logger.info("Retrieval Parameters:")
        logger.info("  k_recall (initial retrieval): %d", self.k_recall)
        logger.info("  k_ensemble: %d", self.k_ensemble)
        logger.info("  k_after_rerank (final docs): %d", self.k_after_rerank)
        logger.info("  BM25 enabled: %s", self.enable_bm25)
        logger.info("-" * 40)
        logger.info("Feature Flags:")
        logger.info("  Multi-query expansion: %s", ENABLE_MULTI_QUERY)
        logger.info("  Intent classification: %s", ENABLE_INTENT_CLASSIFICATION)
        logger.info("  Parent/child chunking: %s", ENABLE_PARENT_CHILD)
        logger.info("-" * 40)
        logger.info("Index Statistics:")
        logger.info("  Chroma collection: %s", self._chroma_collection)
        logger.info("  Total child chunks: %s", col_count)
        logger.info("  Total parent documents: %d", parent_count)
        if collections_found:
            logger.info("  Documents by collection:")
            for coll_name, count in sorted(collections_found.items()):
                logger.info("    - %s: %d chunks", coll_name, count)
        logger.info("-" * 40)
        logger.info("Multi-Retriever Architecture:")
        logger.info("  Biographical collection (ALWAYS runs): %s", self.BIOGRAPHICAL_COLLECTION)
        logger.info("  Min biographical chunks guaranteed: %d", self.MIN_BIOGRAPHICAL_CHUNKS)
        logger.info("-" * 40)
        logger.info("Supplemental Retrievers (run based on intent):")
        for intent, collections in self.SUPPLEMENTAL_COLLECTIONS.items():
            use_parent = self.USE_PARENT_DOCUMENTS_BY_INTENT.get(intent, True)
            if collections:
                logger.info("  %s → %s (use_parent=%s)", intent, collections, use_parent)
            else:
                logger.info("  %s → (biographical only)", intent)
        logger.info("=" * 80)

    def classify_intent(self, question: str) -> str:
        """Classify which supplemental collections to search alongside biographical docs.

        Uses the LLM to analyze the question and determine if supplemental
        archival materials would help answer it. The biographical collection
        is ALWAYS searched regardless of the classification result.

        Args:
            question: The user's query to classify

        Returns:
            One of:
            - 'research': Search Published Works and Research Files
            - 'correspondence': Search Correspondence collection
            - 'none': Only search biographical files (no supplemental)

        Note:
            Falls back to 'none' if classification fails or returns invalid value.
        """

        # Load classification prompt from config (prompts.py)
        classification_prompt = INTENT_CLASSIFICATION_PROMPT.format(question=question)

        try:
            intent = self.llm.invoke(classification_prompt).strip().lower()

            # Validate and default to 'none' if unclear (just biographical)
            if intent not in ["research", "correspondence", "none"]:
                logger.warning(f"Unclear intent classification: {intent}, defaulting to none (biographical only)")
                intent = "none"

            logger.info(f"Supplemental intent: {intent} (biographical always included)")
            return intent
        except Exception as e:
            logger.error(f"Intent classification failed: {e}, defaulting to none (biographical only)")
            return "none"  # Default fallback - just biographical

    def ask(
        self,
        question: str,
        chat_session_id: str = "default",
        *,
        doc_type: Optional[str] = None,
        excluded_parent_ids: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """Answer a question using the multi-retriever RAG pipeline.

        Executes the full retrieval-augmented generation pipeline:
        1. Intent classification (if enabled) → determines supplemental collections
        2. Biographical retrieval → always runs, searches BIOGRAPHICAL_COLLECTION
        3. Supplemental retrieval → runs if intent requires research/correspondence
        4. Cross-encoder reranking → orders combined results by relevance
        5. Biographical guarantee → ensures MIN_BIOGRAPHICAL_CHUNKS in selection
        6. Answer generation → LLM generates response from selected context

        Args:
            question: The user's query to answer
            chat_session_id: Session identifier (kept for API compatibility, unused)
            doc_type: Optional metadata filter for document type
            excluded_parent_ids: Parent IDs to exclude from retrieval results

        Returns:
            Dict containing:
            - answer: str - The generated response
            - sources: list[dict] - Source documents with metadata:
                - parent_id: Document identifier
                - title: Document title
                - source: URL to original document
                - collection: Archive collection name
                - text: Chunk content used in context
            - _metadata: dict - Internal metrics (intent, timing, etc.)

        Note:
            The biographical collection is ALWAYS searched. Intent classification
            only determines whether to ALSO search supplemental collections.
        """
        _ = chat_session_id  # Suppress unused parameter warning
        t_start = time.time()
        logger.info(f"⏱️  Processing question: {question}")

        if DEBUG_RETRIEVAL:
            logger.info("=" * 80)
            logger.info("DEBUG MODE: Query Processing")
            logger.info("=" * 80)
            logger.info("Input:")
            logger.info("  Question: %s", question)
            logger.info("  Session ID: %s", chat_session_id)
            logger.info("  Doc type filter: %s", doc_type if doc_type else "(none)")
            logger.info("  Excluded parent IDs: %s", excluded_parent_ids if excluded_parent_ids else "(none)")
            logger.info("-" * 40)

        # Step 1: Classify intent to determine which SUPPLEMENTAL collections to search.
        # Biographical retriever ALWAYS runs.
        if ENABLE_INTENT_CLASSIFICATION:
            t1 = time.time()
            intent = self.classify_intent(question)
            supplemental_collections = self.SUPPLEMENTAL_COLLECTIONS.get(intent, [])
            logger.info(f"⏱️  Intent classification: {time.time() - t1:.2f}s")
            logger.info(f"   Result: '{intent}' → supplemental={supplemental_collections}")
        else:
            intent = None
            # Search all supplemental collections when intent classification is disabled
            supplemental_collections = list(set(
                col for cols in self.SUPPLEMENTAL_COLLECTIONS.values() for col in cols
            ))
            logger.info("Intent classification DISABLED - searching all supplemental collections")

        # Determine whether to use parent documents or just child chunks
        if ENABLE_PARENT_CHILD:
            use_parent_documents = self.USE_PARENT_DOCUMENTS_BY_INTENT.get(intent, True)
        else:
            use_parent_documents = False

        if DEBUG_RETRIEVAL:
            logger.info("=" * 80)
            logger.info("🔍 MULTI-RETRIEVER PIPELINE")
            logger.info("=" * 80)
            logger.info("Configuration:")
            logger.info("  Intent: %s", intent if intent else "(classification disabled)")
            logger.info("  Biographical collection (always): %s", self.BIOGRAPHICAL_COLLECTION)
            logger.info("  Supplemental collections: %s", supplemental_collections if supplemental_collections else "(none)")
            logger.info("  Min biographical chunks: %d", self.MIN_BIOGRAPHICAL_CHUNKS)
            logger.info("  Use parent documents: %s", use_parent_documents)
            logger.info("  Excluded parent IDs: %s", excluded_parent_ids if excluded_parent_ids else "(none)")
            logger.info("-" * 40)

        # Step 2: Run BIOGRAPHICAL retriever (ALWAYS runs)
        t2 = time.time()
        logger.info("📚 Step 2a: Biographical retrieval...")

        bio_retriever = self._create_collection_retriever(
            collections=[self.BIOGRAPHICAL_COLLECTION],
            k=self.k_recall,
            excluded_parent_ids=excluded_parent_ids,
            doc_type=doc_type,
        )
        biographical_docs = bio_retriever.invoke(question)
        bio_retrieval_time = time.time() - t2

        logger.info(f"   Retrieved {len(biographical_docs)} biographical chunks ({bio_retrieval_time:.2f}s)")

        # Step 3: Run SUPPLEMENTAL retrievers (based on intent)
        supplemental_docs: List[Document] = []

        if supplemental_collections:
            t3 = time.time()
            logger.info("📚 Step 2b: Supplemental retrieval (%s)...", supplemental_collections)

            supp_retriever = self._create_collection_retriever(
                collections=supplemental_collections,
                k=self.k_recall,
                excluded_parent_ids=excluded_parent_ids,
                doc_type=doc_type,
            )
            supplemental_docs = supp_retriever.invoke(question)
            supp_retrieval_time = time.time() - t3

            logger.info(f"   Retrieved {len(supplemental_docs)} supplemental chunks ({supp_retrieval_time:.2f}s)")
        else:
            logger.info("📚 Step 2b: No supplemental retrieval (intent='none' or disabled)")

        # Step 4: Combine all documents for reranking
        all_docs = biographical_docs + supplemental_docs

        if DEBUG_RETRIEVAL:
            logger.info("-" * 40)
            logger.info("📊 Combined for reranking: %d total chunks", len(all_docs))
            logger.info("   Biographical: %d", len(biographical_docs))
            logger.info("   Supplemental: %d", len(supplemental_docs))

        # Step 5: Cross-encoder reranking on combined results
        t4 = time.time()
        logger.info("🔄 Step 3: Cross-encoder reranking...")

        # Create a simple retriever wrapper that returns our combined docs
        class _StaticRetriever(BaseRetriever):
            """Returns a fixed list of documents (for reranking pre-retrieved docs)."""
            docs: List[Document]
            model_config = ConfigDict(arbitrary_types_allowed=True)

            def _get_relevant_documents(self, query: str) -> List[Document]:
                _ = query  # Not used - we return pre-fetched docs
                return self.docs

        static_retriever = _StaticRetriever(docs=all_docs)

        # Apply cross-encoder reranking
        compression_ret = ContextualCompressionRetriever(
            base_retriever=static_retriever,
            base_compressor=self.compressor,
        )
        reranked_docs = compression_ret.invoke(question)
        rerank_time = time.time() - t4

        logger.info(f"   Reranked to {len(reranked_docs)} chunks ({rerank_time:.2f}s)")

        # Step 6: Ensure at least MIN_BIOGRAPHICAL_CHUNKS in final selection
        top_docs = self._ensure_biographical_chunk(
            reranked_docs=reranked_docs,
            biographical_docs=biographical_docs,
            k_final=self.k_after_rerank,
        )

        retrieval_time = time.time() - t2  # Total retrieval time
        context_text = "\n\n---\n\n".join(d.page_content for d in top_docs)

        # Debug mode: print retrieved chunks to console
        if DEBUG_RETRIEVAL:
            # Count documents by collection for summary
            collection_counts = {}
            for doc in top_docs:
                coll = (doc.metadata or {}).get("collection", "Unknown")
                collection_counts[coll] = collection_counts.get(coll, 0) + 1

            bio_count = sum(1 for d in top_docs if (d.metadata or {}).get("is_biographical", False))
            supp_count = len(top_docs) - bio_count

            logger.info("=" * 80)
            logger.info("📊 FINAL SELECTION: %d chunks for generation", len(top_docs))
            logger.info("   ⭐ Biographical: %d chunks (min required: %d)", bio_count, self.MIN_BIOGRAPHICAL_CHUNKS)
            logger.info("   Supplemental: %d chunks", supp_count)
            logger.info("   By collection:")
            for coll, count in sorted(collection_counts.items()):
                marker = "⭐" if coll == self.BIOGRAPHICAL_COLLECTION else "  "
                logger.info("     %s %s: %d", marker, coll, count)
            logger.info("=" * 80)

            for i, doc in enumerate(top_docs, 1):
                md = doc.metadata or {}
                title = md.get("Title") or md.get("title") or "Unknown"
                parent_id = md.get("parent_id", "Unknown")
                collection = md.get("collection", "Unknown")
                is_biographical = md.get("is_biographical", False)
                relevance_score = md.get("relevance_score", "N/A")
                logger.info("-" * 40)
                if is_biographical:
                    logger.info("CHUNK %d: ⭐ BIOGRAPHICAL", i)
                else:
                    logger.info("CHUNK %d: SUPPLEMENTAL", i)
                logger.info("  Title: %s", title)
                logger.info("  Parent ID: %s", parent_id)
                logger.info("  Collection: %s", collection)
                logger.info("  Relevance score: %s", relevance_score)
                logger.info("  Content (%d chars):", len(doc.page_content))
                # Print content with indentation, truncate if very long
                content_preview = doc.page_content[:2000] + ("..." if len(doc.page_content) > 2000 else "")
                for line in content_preview.split("\n"):
                    logger.info("    %s", line)
            logger.info("=" * 80)

        # Build a lightweight metadata list for UI citations.
        # (The UI expects a URL when possible; we derive it from known metadata.)
        sources = []
        base_url = "https://exhibits.library.stonybrook.edu/mfp/files/original/"

        for d in top_docs:
            md = d.metadata or {}
            # Prefer a stable file URL constructed from `pdf_filename`/`filename`.
            # Fall back to whatever source-like fields exist.
            pdf_filename = md.get("pdf_filename") or md.get("filename")
            source_url = base_url + pdf_filename if pdf_filename else (md.get("item_url") or md.get("Source") or md.get("source"))

            sources.append(
                {
                    "parent_id": md.get("parent_id"),
                    "title": md.get("Title") or md.get("title"),
                    "source": source_url,
                    "collection": md.get("collection"),
                    "text": d.page_content,  # Include the actual source text content
                }
            )

        t4 = time.time()
        answer = self._answer_chain.invoke(
            {"question": question, "context": context_text},
        )
        answer_time = time.time() - t4
        total_time = time.time() - t_start

        logger.info(f"⏱️  Answer generation: {answer_time:.2f}s")
        logger.info(f"⏱️  TOTAL query time: {total_time:.2f}s")

        if DEBUG_RETRIEVAL:
            # Count biographical vs supplemental sources
            bio_count = sum(1 for d in top_docs if (d.metadata or {}).get("is_biographical", False))
            supp_count = len(top_docs) - bio_count

            # Count by collection
            source_collections = {}
            for d in top_docs:
                coll = (d.metadata or {}).get("collection", "Unknown")
                source_collections[coll] = source_collections.get(coll, 0) + 1

            logger.info("=" * 80)
            logger.info("📋 QUERY SUMMARY")
            logger.info("=" * 80)
            logger.info("Timing:")
            logger.info("  Retrieval + reranking: %.2fs", retrieval_time)
            logger.info("  Answer generation: %.2fs", answer_time)
            logger.info("  Total: %.2fs", total_time)
            logger.info("-" * 40)
            logger.info("Sources used for answer:")
            logger.info("  Total: %d chunks", len(sources))
            logger.info("  ⭐ Biographical: %d (guaranteed min: %d)", bio_count, self.MIN_BIOGRAPHICAL_CHUNKS)
            logger.info("  Supplemental: %d", supp_count)
            logger.info("  By collection:")
            for coll, count in sorted(source_collections.items()):
                marker = "⭐" if coll == self.BIOGRAPHICAL_COLLECTION else "  "
                logger.info("    %s %s: %d", marker, coll, count)
            logger.info("  Context length: %d chars", len(context_text))
            logger.info("  Answer length: %d chars", len(answer))
            logger.info("=" * 80)

        return {
            "answer": answer,
            "sources": sources,
            # Add timing and metadata for optional logging
            "_metadata": {
                "intent": intent,
                "retrieval_time": retrieval_time,
                "answer_time": answer_time,
                "total_time": total_time,
                "num_sources": len(sources),
                "excluded_parent_ids": excluded_parent_ids or [],
            }
        }

    def cleanup_session(self, session_id: str) -> None:
        """No-op since chat history is disabled. Kept for API compatibility."""
        _ = session_id  # Suppress unused parameter warning
        pass

    @staticmethod
    def log_interaction(
        log_path: str,
        session_id: str,
        question: str,
        answer: str,
        sources: List[Dict],
        intent: Optional[str] = None,
        retrieval_time: Optional[float] = None,
        answer_time: Optional[float] = None,
        total_time: Optional[float] = None,
        excluded_parent_ids: Optional[List[str]] = None,
        num_sources: Optional[int] = None,
    ) -> None:
        """Log a chat interaction to a JSONL file for data collection.

        Each log entry is a single line JSON object containing:
        - timestamp: ISO format datetime
        - session_id: unique identifier for the chat session
        - question: user's query
        - answer: RAG system response
        - sources: list of source documents with metadata and full text content
        - intent: classified intent (biographical/research/correspondence)
        - retrieval_time: seconds spent on retrieval + reranking
        - answer_time: seconds spent generating answer
        - total_time: total query processing time
        - excluded_parent_ids: list of excluded document IDs (if any)
        - num_sources: number of source documents used

        This format is optimized for:
        - Lightweight append-only writes (no file locking needed)
        - Easy parsing with standard JSON tools
        - Simple analysis with pandas or jq
        """
        # Ensure log directory exists
        log_file = Path(log_path)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Build log entry
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "question": question,
            "answer": answer,
            "sources": sources,
            "intent": intent,
            "retrieval_time_seconds": retrieval_time,
            "answer_time_seconds": answer_time,
            "total_time_seconds": total_time,
            "excluded_parent_ids": excluded_parent_ids or [],
            "num_sources": num_sources or len(sources),
        }

        # Append as single line JSON (JSONL format)
        try:
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Failed to log interaction: {e}")