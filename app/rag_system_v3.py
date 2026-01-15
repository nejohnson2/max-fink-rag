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
from typing import List, Dict, Optional
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ConfigDict

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory

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
from config import OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_URL, ENABLE_MULTI_QUERY, logger

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

    # Collection filter mapping for intent classification
    COLLECTION_FILTERS = {
        "biographical": ["Biographical Files"],
        "research": ["Published Works", "Research Files and Unpublished Works"],
        "correspondence": ["Correspondence"]
    }

    # Parent docs can be extremely large for some collections (e.g. long
    # biographical PDFs). For those intents, default to sending chunk text.
    USE_PARENT_DOCUMENTS_BY_INTENT = {
        "biographical": False,
        "research": False,
        "correspondence": True,
    }

    def __init__(
        self,
        #store_dir: str = "./rag_store",
        store_dir: str = "/fink_archive",
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

        # Persisted artifacts live under `store_dir`:
        # - `chroma/` holds the vector index
        # - `parents.jsonl` maps parent_id → full parent text/metadata
        chroma_dir = os.path.join(store_dir, "chroma")
        parents_path = os.path.join(store_dir, "parents.jsonl")

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

        # Prompt: system behavior + conversation history + question+context.
        # self.prompt = ChatPromptTemplate.from_messages(
        #     [
        #         (
        #             "system",
        #             "You are a helpful assistant. Use the provided context to answer concisely and cite facts from it. "
        #             "If the answer is not in the context, say you don't know.",
        #         ),
        #         MessagesPlaceholder("history"),
        #         ("human", "Question: {question}\n\nContext:\n{context}"),
        #     ]
        # )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are a digital librarian assisting users with a curated archival collection. "

                "Your role is to help users understand, interpret, and navigate the materials,"
                "not just to provide direct answers."

                "Use only the provided context to respond to the user’s question."
                "Base your answer on the documents retrieved from the collection."

                "If the context clearly answers the question:"
                "- Explain the answer in clear, accessible language."
                "- Briefly indicate what type of document(s) the information comes from"
                "(for example, correspondence, bibliographic records, or published work)."

                "If the context does not fully answer the question:"
                "- Say what can and cannot be determined from the available materials."
                "- Suggest what kinds of documents or information might help answer it."

                "Do not speculate or introduce information that is not supported by the context."
                "It is acceptable and encouraged to say “I don’t know” when appropriate."
                "Write in a neutral, professional tone suitable for a library reference interaction."
            # "You are a knowledgeable archivist assistant specializing in Max Fink's life and work. "
            # "Answer questions using the provided archival documents. "
            # "Be precise, scholarly, and cite specific details from the context. "
            # "If information is not in the documents, say so clearly. "
            # "When discussing dates, events, or people, be specific and reference the source material."),
            ),
            MessagesPlaceholder("history"),
            ("human", "Question: {question}\n\nArchival Context:\n{context}"),
        ])

        self._history_store: Dict[str, InMemoryChatMessageHistory] = {}
        self._answer_chain = self.prompt | self.llm | StrOutputParser()

    def _build_bm25_from_chroma(self) -> BM25Retriever:
        """Build a BM25 retriever from whatever is currently stored in Chroma."""
        col = self.vs._collection
        data = col.get(include=["documents", "metadatas"])
        docs: List[Document] = []
        for text, md in zip(data.get("documents", []), data.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=md or {}))
        return BM25Retriever.from_documents(docs)

    def classify_intent(self, question: str) -> str:
        """Classify user question intent into biographical, research, or correspondence."""

        classification_prompt = f"""Classify the following question about Max Fink into ONE of these categories:
- biographical: Questions about Max Fink's life, background, education, career, personal history
- research: Questions about his scientific work, publications, studies, findings, theories
- correspondence: Questions about letters, communications, exchanges with colleagues

Question: {question}

Respond with ONLY ONE WORD: biographical, research, or correspondence"""

        try:
            intent = self.llm.invoke(classification_prompt).strip().lower()

            # Validate and default to research if unclear
            if intent not in ["biographical", "research", "correspondence"]:
                logger.warning(f"Unclear intent classification: {intent}, defaulting to research")
                intent = "research"

            logger.info(f"Classified intent as: {intent}")
            return intent
        except Exception as e:
            logger.error(f"Intent classification failed: {e}, defaulting to research")
            return "research"  # Default fallback

    def ask(
        self,
        question: str,
        chat_session_id: str = "default",
        *,
        doc_type: Optional[str] = None,
        excluded_parent_ids: Optional[List[str]] = None,
    ) -> Dict[str, object]:
        """Answer a question using RAG and return answer + source metadata.

        Returns a dict of:
        - answer: str
        - sources: list[dict] with parent_id/title/collection/source URL

        `doc_type` is an additional metadata filter layered on top of the
        intent-based collection filter.

        `excluded_parent_ids` is a list of parent_id values to exclude from
        retrieval. Documents with these IDs will not be returned in results.
        """
        import time
        t_start = time.time()

        # Step 1: Classify intent to determine which collections are eligible.
        t1 = time.time()
        intent = self.classify_intent(question)
        allowed_collections = self.COLLECTION_FILTERS[intent]
        logger.info(f"⏱️  Intent classification: {time.time() - t1:.2f}s")

        # Step 2: Build vector search kwargs with a metadata filter.
        search_kwargs: Dict[str, object] = {"k": self.k_recall}

        # Apply collection filter based on intent
        search_kwargs["filter"] = {
            "collection": {"$in": allowed_collections}
        }

        # Exclude specific parent IDs if provided
        if excluded_parent_ids:
            search_kwargs["filter"]["parent_id"] = {"$nin": excluded_parent_ids}
            logger.info(f"Excluding {len(excluded_parent_ids)} parent IDs from retrieval")

        # If doc_type is also specified, combine filters
        if doc_type:
            search_kwargs["filter"]["doc_type"] = doc_type

        vec_ret = self.vs.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        use_parent_documents = self.USE_PARENT_DOCUMENTS_BY_INTENT.get(intent, True)

        # Note: BM25 doesn't support metadata filtering.
        # We still include BM25 for recall, but apply the collection filter after
        # parent reconstruction (in `_ParentFromChildRetriever`).
        if self.enable_bm25 and self.bm25 is not None:
            ensemble_child: BaseRetriever = EnsembleRetriever(
                retrievers=[vec_ret, self.bm25],
                weights=[0.5, 0.5],
            )
        else:
            ensemble_child = vec_ret

        # When parent documents are huge (common for biographical PDFs), we keep
        # results as child chunks to avoid sending 70+ pages to the LLM.
        # Otherwise, we reconstruct the parent documents for more coherent context.
        if use_parent_documents:
            base_ret: BaseRetriever = _ParentFromChildRetriever(
                child_retriever=ensemble_child,
                parent_lookup=self._parent_lookup,
                collection_filter=allowed_collections,
                excluded_parent_ids=excluded_parent_ids,
            )
        else:
            # Still enforce collection/doc_type filters even though BM25 doesn't
            # support them natively.
            base_ret = _MetadataFilterRetriever(
                child_retriever=ensemble_child,
                allowed_collections=allowed_collections,
                doc_type=doc_type,
                excluded_parent_ids=excluded_parent_ids,
            )

        # Optional multi-query expansion (can be disabled via config for speed)
        if ENABLE_MULTI_QUERY:
            t2 = time.time()
            expanded_ret: BaseRetriever = MultiQueryRetriever.from_llm(
                retriever=base_ret,
                llm=self.llm,
                include_original=True,
            )
            logger.info(f"⏱️  Multi-query setup: {time.time() - t2:.2f}s")
            logger.info("Multi-query expansion ENABLED")
        else:
            expanded_ret = base_ret
            logger.info("Multi-query expansion DISABLED (using single query)")

        t3 = time.time()
        compression_ret = ContextualCompressionRetriever(
            base_retriever=expanded_ret,
            base_compressor=self.compressor,
        )

        # Retrieval pipeline output is a list of Documents ranked by relevance.
        # NOTE: If multi-query enabled, this calls LLM + retrieval multiple times
        # Then runs cross-encoder reranking
        contexts = compression_ret.invoke(question)
        logger.info(f"⏱️  Retrieval + reranking: {time.time() - t3:.2f}s")
        
        # Build context text for the LLM. Even though the compressor already
        # returns `top_n`, we defensively slice.
        top_docs = contexts[: self.k_after_rerank]
        logger.info(f"Retrieved {len(top_docs)} top documents for context.")
        context_text = "\n\n---\n\n".join(d.page_content for d in top_docs)
        logger.info(f"Context text built with {len(top_docs)} documents.")
        logger.info(f"Context text: {context_text}")

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
                }
            )

        def _get_history(session_id: str) -> BaseChatMessageHistory:
            # Simple in-memory conversation store keyed by session id.
            if session_id not in self._history_store:
                self._history_store[session_id] = InMemoryChatMessageHistory()
            return self._history_store[session_id]

        chain_with_memory = RunnableWithMessageHistory(
            self._answer_chain,
            get_session_history=_get_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        t4 = time.time()
        answer = chain_with_memory.invoke(
            {"question": question, "context": context_text},
            config={"configurable": {"session_id": chat_session_id}},
        )
        answer_time = time.time() - t4
        retrieval_time = time.time() - t3
        total_time = time.time() - t_start

        logger.info(f"⏱️  Answer generation: {answer_time:.2f}s")
        logger.info(f"⏱️  TOTAL query time: {total_time:.2f}s")

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
        """Remove conversation history for a specific session.

        Called when a browser tab closes to free up memory.
        """
        if session_id in self._history_store:
            del self._history_store[session_id]
            logger.info(f"Session history cleared: {session_id}")
        else:
            logger.warning(f"Attempted to clean up non-existent session: {session_id}")

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
        - sources: list of source documents with metadata
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