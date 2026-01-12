from __future__ import annotations

import os
import json
import sys
from typing import List, Dict, Optional

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
        from langchain.retrievers.document_compressors import CrossEncoderReranker  # if langchain is installed
    except Exception:  # pragma: no cover
        # Last resort: implement rerank manually (see note below)
        CrossEncoderReranker = None  # type: ignore

#from remote_ollama import RemoteOllamaLLM
#from config import OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_URL
from remote_ollama import RemoteOllamaLLM
from config import OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_URL

class _ParentFromChildRetriever(BaseRetriever):
    child_retriever: BaseRetriever
    parent_lookup: Dict[str, Document]
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        #child_hits: List[Document] = self.child_retriever.get_relevant_documents(query)
        child_hits: List[Document] = self.child_retriever.invoke(query)
        seen = set()
        parents: List[Document] = []
        for d in child_hits:
            pid = (d.metadata or {}).get("parent_id")
            if pid and pid in self.parent_lookup and pid not in seen:
                seen.add(pid)
                parents.append(self.parent_lookup[pid])
        return parents if parents else child_hits


def _load_parent_lookup(parents_jsonl_path: str) -> Dict[str, Document]:
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
        self.store_dir = store_dir
        self.k_recall = k_recall
        self.k_ensemble = k_ensemble
        self.k_after_rerank = k_after_rerank
        self.enable_bm25 = enable_bm25

        chroma_dir = os.path.join(store_dir, "chroma")
        parents_path = os.path.join(store_dir, "parents.jsonl")

        self.embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        self.vs = Chroma(
            collection_name=chroma_collection,
            embedding_function=self.embeddings,
            persist_directory=chroma_dir,
        )

        self._parent_lookup = _load_parent_lookup(parents_path)

        self.bm25: Optional[BM25Retriever] = None
        if enable_bm25:
            self.bm25 = self._build_bm25_from_chroma()
            self.bm25.k = self.k_recall

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

        self.llm = RemoteOllamaLLM(
            model=OLLAMA_MODEL,
            base_url=OLLAMA_URL,
            headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Use the provided context to answer concisely and cite facts from it. "
                    "If the answer is not in the context, say you don't know.",
                ),
                MessagesPlaceholder("history"),
                ("human", "Question: {question}\n\nContext:\n{context}"),
            ]
        )

        self._history_store: Dict[str, InMemoryChatMessageHistory] = {}
        self._answer_chain = self.prompt | self.llm | StrOutputParser()

    def _build_bm25_from_chroma(self) -> BM25Retriever:
        col = self.vs._collection
        data = col.get(include=["documents", "metadatas"])
        docs: List[Document] = []
        for text, md in zip(data.get("documents", []), data.get("metadatas", [])):
            docs.append(Document(page_content=text, metadata=md or {}))
        return BM25Retriever.from_documents(docs)

    def ask(
        self,
        question: str,
        chat_session_id: str = "default",
        *,
        doc_type: Optional[str] = None,
    ) -> str:
        search_kwargs: Dict[str, object] = {"k": self.k_recall}
        if doc_type:
            search_kwargs["filter"] = {"doc_type": doc_type}

        vec_ret = self.vs.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs,
        )

        if self.enable_bm25 and self.bm25 is not None and not doc_type:
            ensemble_child: BaseRetriever = EnsembleRetriever(
                retrievers=[vec_ret, self.bm25],
                weights=[0.5, 0.5],
            )
        else:
            ensemble_child = vec_ret

        parent_ret: BaseRetriever = _ParentFromChildRetriever(
            child_retriever=ensemble_child,
            parent_lookup=self._parent_lookup,
        )

        expanded_ret: BaseRetriever = MultiQueryRetriever.from_llm(
            retriever=parent_ret,
            llm=self.llm,
            include_original=True,
        )

        compression_ret = ContextualCompressionRetriever(
            base_retriever=expanded_ret,
            base_compressor=self.compressor,
        )

        #contexts = compression_ret.get_relevant_documents(question)
        contexts = compression_ret.invoke(question)
        
        # Build context text for the LLM
        top_docs = contexts[: self.k_after_rerank]
        context_text = "\n\n---\n\n".join(d.page_content for d in top_docs)

        # Build a lightweight metadata list to return
        sources = []
        for d in top_docs:
            md = d.metadata or {}
            sources.append(
                {
                    "parent_id": md.get("parent_id"),
                    "title": md.get("Title") or md.get("title"),
                    "source": md.get("item_url") or md.get("Source") or md.get("source"),
                    "collection": md.get("collection"),
                }
            )

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

        answer = chain_with_memory.invoke(
            {"question": question, "context": context_text},
            config={"configurable": {"session_id": chat_session_id}},
        )

        return {
            "answer": answer,
            "sources": sources,
        }