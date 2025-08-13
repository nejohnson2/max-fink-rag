# RAGSystem.py
from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterable
import os
import uuid

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Vector store + retrievers
from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# Cross-encoder reranker (bge-reranker)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import CrossEncoderReranker

# Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import ConfigDict

from remote_ollama import RemoteOllamaLLM

# LLM (Ollama)
try:
    # Preferred modern import
    from langchain_ollama import ChatOllama
except Exception:
    # Fallback for older installs
    from langchain_community.chat_models import ChatOllama

# LCEL prompt + memory wrapper
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
try:
    from langchain_core.retrievers import BaseRetriever
except Exception:
    from langchain.retrievers import BaseRetriever  # fallback
import PyPDF2

from config import OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_URL

class _ParentFromChildRetriever(BaseRetriever):
    # def __init__(self, child_retriever, parent_lookup: Dict[str, Document]):
    #     self.child_retriever = child_retriever
    #     self.parent_lookup = parent_lookup
    child_retriever: BaseRetriever
    parent_lookup: Dict[str, Document]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        child_hits: List[Document] = self.child_retriever.get_relevant_documents(query)
        seen = set()
        parents: List[Document] = []
        for d in child_hits:
            pid = d.metadata.get("parent_id")
            if pid and pid in self.parent_lookup and pid not in seen:
                seen.add(pid)
                parents.append(self.parent_lookup[pid])
        # Fallback: if a child lacked parent_id, return it as-is
        if not parents and child_hits:
            return child_hits
        return parents

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        # Async variant for compatibility
        child_hits: List[Document] = await self.child_retriever.aget_relevant_documents(query)
        seen = set()
        parents: List[Document] = []
        for d in child_hits:
            pid = d.metadata.get("parent_id")
            if pid and pid in self.parent_lookup and pid not in seen:
                seen.add(pid)
                parents.append(self.parent_lookup[pid])
        if not parents and child_hits:
            return child_hits
        return parents


class RAGSystem:
    def __init__(
        self,
        persist_dir: str = "./chroma_rag",
        ollama_model: str = "llama3.1:latest",
        embeddings_model: str = "BAAI/bge-small-en-v1.5",
        reranker_model: str = "BAAI/bge-reranker-base",
        k_recall: int = 30,           # initial recall per retriever before fusion
        k_ensemble: int = 20,         # top-k after ensemble fusion
        k_after_rerank: int = 6,      # final k passed to the LLM
        child_chunk_size: int = 300,
        child_chunk_overlap: int = 40,
        parent_chunk_size: int = 1000,
        parent_chunk_overlap: int = 100,
        chroma_collection: str = "rag_collection",
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.persist_dir = persist_dir
        self.k_recall = k_recall
        self.k_ensemble = k_ensemble
        self.k_after_rerank = k_after_rerank

        # Embeddings for vector store
        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)

        # Vector store placeholder (populated after indexing)
        self.vs: Optional[Chroma] = None

        # BM25 retriever placeholder (populated after indexing)
        self.bm25: Optional[BM25Retriever] = None

        # Parent doc storage (in-memory)
        self._parent_lookup: Dict[str, Document] = {}

        # Splitters
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size, chunk_overlap=child_chunk_overlap
        )
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size, chunk_overlap=parent_chunk_overlap
        )

        # Reranker
        self.cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
        self.compressor = CrossEncoderReranker(model=self.cross_encoder, top_n=self.k_after_rerank)

        # LLM (Ollama)
        #self.llm = ChatOllama(model=ollama_model, temperature=0.2)
        self.llm = RemoteOllamaLLM(
                model=OLLAMA_MODEL,
                base_url=OLLAMA_URL,
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"}
            )

        # Prompt (with history + retrieved context)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system",
                 "You are a helpful assistant. Use the provided context to answer concisely and cite facts from it."
                 "The context provided is about Max Fink based on an extensive collection of his work."
                 "If the answer is not in the context, say you don't know."),
                MessagesPlaceholder("history"),
                ("human", "Question: {question}\n\nContext:\n{context}")
            ]
        )

        # Message history store (per-session)
        self._history_store: Dict[str, InMemoryChatMessageHistory] = {}

        # Build an LCEL chain; the retriever is attached per-call in ask()
        self._answer_chain = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Collection name if you need it later
        self.chroma_collection = chroma_collection

    # ---------- Public API ----------

    def index_documents(
        self,
        docs: Iterable[Document] | Iterable[Dict[str, Any]],
        *,
        clear_existing: bool = False,
    ) -> None:
        """
        Indexes a set of parent documents using parent/child chunking.
        Each input may be a langchain Document or a dict with keys: page_content, metadata (optional), id (optional).
        """
        # Normalize inputs into parent Documents with guaranteed IDs
        parent_docs: List[Document] = []
        for d in docs:
            if isinstance(d, Document):
                pid = d.metadata.get("id") or d.metadata.get("doc_id") or str(uuid.uuid4())
                md = dict(d.metadata or {})
                md["parent_id"] = pid
                parent_docs.append(Document(page_content=d.page_content, metadata=md))
            else:
                content = d.get("page_content", "")
                md = dict(d.get("metadata", {}) or {})
                pid = d.get("id") or md.get("id") or md.get("doc_id") or str(uuid.uuid4())
                md["parent_id"] = pid
                parent_docs.append(Document(page_content=content, metadata=md))

        # Split parents into children
        child_docs: List[Document] = []
        for pdoc in parent_docs:
            for chunk in self.child_splitter.split_text(pdoc.page_content):
                child_docs.append(
                    Document(
                        page_content=chunk,
                        metadata={**pdoc.metadata}  # carries parent_id + any other metadata
                    )
                )

        # Build / refresh Chroma vector store on children
        if clear_existing and os.path.isdir(self.persist_dir):
            # naive clear: remove directory contents
            for root, dirs, files in os.walk(self.persist_dir, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))

        self.vs = Chroma.from_documents(
            documents=child_docs,
            embedding=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.chroma_collection,
        )

        # BM25 over the same child docs
        self.bm25 = BM25Retriever.from_documents(child_docs)
        self.bm25.k = self.k_recall  # initial recall depth

        # Keep a parent lookup for mapping child->parent at retrieval time
        # If desired, pre-split parents with a larger splitter and store those as parent docs
        self._parent_lookup = {}
        for pdoc in parent_docs:
            # Optionally, you can re-chunk parents into ~1000-token spans; here we store the whole parent as one
            self._parent_lookup[pdoc.metadata["parent_id"]] = pdoc

    def ask(self, question: str, chat_session_id: str = "default") -> str:
        """
        Runs the full pipeline: (Vector + BM25) -> Ensemble -> Parent mapping -> MultiQuery -> Rerank
        Then answers with Ollama, using chat memory for the given session_id.
        """
        if self.vs is None or self.bm25 is None:
            raise RuntimeError("Index is empty. Call index_documents(...) first.")

        # Base child retrievers
        vec_ret = self.vs.as_retriever(search_type="similarity", search_kwargs={"k": self.k_recall})
        bm25_ret = self.bm25

        # Fuse child results
        ensemble_child = EnsembleRetriever(
            retrievers=[vec_ret, bm25_ret],
            weights=[0.5, 0.5],
            search_type="mmr",              # or "similarity_score_threshold" per preference
            search_kwargs={"k": self.k_ensemble},
        )

        # Map to parent documents
        parent_ret = _ParentFromChildRetriever(child_retriever=ensemble_child, parent_lookup=self._parent_lookup)

        # Query expansion
        expanded_ret = MultiQueryRetriever.from_llm(
            retriever=parent_ret,
            llm=self.llm,
            include_original=True,
        )

        # Rerank / compress to most relevant parent spans
        compression_ret = ContextualCompressionRetriever(
            base_retriever=expanded_ret,
            base_compressor=self.compressor,
        )

        # Fetch top context
        contexts = compression_ret.get_relevant_documents(question)
        context_text = "\n\n---\n\n".join(d.page_content for d in contexts[: self.k_after_rerank])

        # Memory-backed chain via RunnableWithMessageHistory
        def _get_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in self._history_store:
                self._history_store[session_id] = InMemoryChatMessageHistory()
            return self._history_store[session_id]

        chain_with_memory = RunnableWithMessageHistory(
            self._answer_chain,
            get_session_history=_get_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        result = chain_with_memory.invoke(
            {"question": question, "context": context_text},
            config={"configurable": {"session_id": chat_session_id}},
        )
        return result

    # Optional: expose a simple add() to append new docs without full rebuild
    def add_documents(self, docs: Iterable[Document] | Iterable[Dict[str, Any]]):
        """
        Adds new documents to the existing index (keeps prior content).
        """
        if self.vs is None or self.bm25 is None:
            return self.index_documents(docs)

        # Normalize & split
        new_parent_docs: List[Document] = []
        for d in docs:
            if isinstance(d, Document):
                pid = d.metadata.get("id") or d.metadata.get("doc_id") or str(uuid.uuid4())
                md = dict(d.metadata or {})
                md["parent_id"] = pid
                new_parent_docs.append(Document(page_content=d.page_content, metadata=md))
            else:
                content = d.get("page_content", "")
                md = dict(d.get("metadata", {}) or {})
                pid = d.get("id") or md.get("id") or md.get("doc_id") or str(uuid.uuid4())
                md["parent_id"] = pid
                new_parent_docs.append(Document(page_content=content, metadata=md))

        new_child_docs: List[Document] = []
        for pdoc in new_parent_docs:
            for chunk in self.child_splitter.split_text(pdoc.page_content):
                new_child_docs.append(
                    Document(page_content=chunk, metadata={**pdoc.metadata})
                )

        # Upsert into Chroma and BM25
        self.vs.add_documents(new_child_docs)
        self.bm25.add_documents(new_child_docs)

        # Update parent lookup
        for pdoc in new_parent_docs:
            self._parent_lookup[pdoc.metadata["parent_id"]] = pdoc

    def add_pdfs(
        self,
        pdf_paths: List[str],
        metadata_list: Optional[List[Dict[str, Any]]] = None,
    ):
        """
        Reads one or more PDF files, extracts text, and adds them as documents to the vector store.
        Args:
            pdf_paths: List of file paths to PDF documents.
            metadata_list: Optional list of metadata dicts (one per PDF).
        """
        docs = []
        for idx, pdf_path in enumerate(pdf_paths):
            print(idx, pdf_path)
            try:
                with open(pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    text = ""
                    for page in reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                # Use provided metadata or default
                metadata = metadata_list[idx] if metadata_list and idx < len(metadata_list) else {}
                metadata = dict(metadata)  # copy to avoid mutation
                metadata["source"] = pdf_path
                docs.append(Document(page_content=text, metadata=metadata))
            except Exception as e:
                print(f"Error reading {pdf_path}: {e}")
        if docs:
            self.add_documents(docs)