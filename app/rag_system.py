from __future__ import annotations

"""Retrieval-Augmented Generation (RAG) system for the Max Fink archive.

This module wires together:
- Chroma vector search (semantic retrieval)
- Optional BM25 keyword search (lexical retrieval)
- Parent/child document reconstruction (retrieve chunk → return full parent)
- Multi-query expansion (LLM generates alternative queries)
- Cross-encoder reranking / contextual compression (select best passages)
- A chat-style prompt with conversation history

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
    BIOGRAPHY_CONTEXT,
)

class _ParentFromChildRetriever(BaseRetriever):
    """Retriever adapter that converts child chunk hits into unique parent docs.

    The vector store is typically populated with *child chunks* for recall.
    The UI/LLM, however, usually benefits from seeing the full *parent document*
    (or larger sections). This wrapper:

    1) Invokes an underlying `child_retriever`.
    2) Reads each child's `metadata.parent_id`.
    3) Looks up the corresponding parent `Document` in `parent_lookup`.
    4) Optionally filters parents by `collection_filter`.

    If no parents are found, it falls back to returning the original child hits.
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
    """Filter retriever results by simple metadata constraints.

    This is used when we intentionally keep results as *child chunks* (e.g.
    biographical intent) but still want to enforce the same metadata filters
    that Chroma supports natively.

    Note: This filter is applied *after* the wrapped retriever returns results.
    It is primarily to keep BM25/ensemble results aligned with vector filters.
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
    """Load parent documents from a JSONL file into an in-memory lookup.

    Expected JSONL record shape (per line):
    - parent_id: str
    - text: str
    - metadata: dict (optional)

    This enables fast parent reconstruction during retrieval without having to
    re-query a separate store.
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
    """End-to-end RAG pipeline: retrieval → rerank → prompt → answer.

    Design notes:
    - Vector retrieval provides semantic recall; optional BM25 helps with exact
      terms (names, acronyms, citations).
    - Parent/child reconstruction ensures the LLM sees coherent source text.
    - Intent classification gates retrieval to relevant archive collections.
    """

    # Collection definitions for multi-retriever architecture
    # Biographical is ALWAYS searched; supplemental collections are added based on intent
    BIOGRAPHICAL_COLLECTION = "Biographical Files"

    SUPPLEMENTAL_COLLECTIONS = {
        "research": ["Published Works", "Research Files and Unpublished Works"],
        "correspondence": ["Correspondence"],
        "none": [],  # No supplemental collections needed
    }

    # Minimum number of biographical chunks guaranteed in final selection
    MIN_BIOGRAPHICAL_CHUNKS = 1

    # Parent docs can be extremely large for some collections (e.g. long
    # biographical PDFs). For those intents, default to sending chunk text.
    USE_PARENT_DOCUMENTS_BY_INTENT = {
        "research": False,
        "correspondence": True,
        "none": False,
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
        """Initialize embeddings, vectorstore, retrievers, reranker, and LLM."""
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

        # Prompt: system behavior + question+context (chat history disabled).
        # Load system prompt from config (prompts.py)
        # Biography context (from biography.md) is included as foundational knowledge
        # Escape curly braces in biography to prevent LangChain template parsing errors
        escaped_biography = BIOGRAPHY_CONTEXT.replace("{", "{{").replace("}", "}}") if BIOGRAPHY_CONTEXT else ""
        biography_section = f"Background Information:\n{escaped_biography}\n\n" if escaped_biography else ""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", f"Question: {{question}}\n\n{biography_section}Archival Context:\n{{context}}"),
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
        """Create a retriever for specific collections.

        Args:
            collections: List of collection names to search
            k: Number of documents to retrieve
            excluded_parent_ids: Parent IDs to exclude from results
            doc_type: Optional doc_type filter

        Returns:
            A retriever configured for the specified collections
        """
        search_kwargs: Dict[str, object] = {"k": k}
        search_kwargs["filter"] = {"collection": {"$in": collections}}

        if excluded_parent_ids:
            search_kwargs["filter"]["parent_id"] = {"$nin": excluded_parent_ids}

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
        """Ensure at least MIN_BIOGRAPHICAL_CHUNKS biographical chunks in final selection.

        After cross-encoder reranking, this method checks if the top k_final documents
        include at least MIN_BIOGRAPHICAL_CHUNKS biographical chunks. If not, it inserts
        the highest-ranked biographical chunk(s) to meet the minimum.

        Args:
            reranked_docs: Documents after cross-encoder reranking (ordered by relevance)
            biographical_docs: Original biographical chunks before reranking
            k_final: Number of documents to return

        Returns:
            Final list of k_final documents with guaranteed biographical representation
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
        """Print comprehensive debug information about the RAG system structure."""
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
        logger.info("  Biography context: %s (%d chars)",
                   bool(BIOGRAPHY_CONTEXT), len(BIOGRAPHY_CONTEXT) if BIOGRAPHY_CONTEXT else 0)
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

        Returns one of: 'research', 'correspondence', or 'none'
        Biographical files are ALWAYS searched regardless of intent.
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
        chat_session_id: str = "default",  # Kept for API compatibility, but unused (history disabled)
        *,
        doc_type: Optional[str] = None,
        excluded_parent_ids: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """Answer a question using RAG and return answer + source metadata.

        Returns a dict of:
        - answer: str
        - sources: list[dict] with parent_id/title/collection/source URL/text content

        `chat_session_id` is kept for API compatibility but currently unused
        (chat history is disabled).

        `doc_type` is an additional metadata filter layered on top of the
        intent-based collection filter.

        `excluded_parent_ids` is a list of parent_id values to exclude from
        retrieval. Documents with these IDs will not be returned in results.
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