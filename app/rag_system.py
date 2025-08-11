import os
from typing import List, Optional, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM
from PyPDF2 import PdfReader
import torch
from config import logger
from remote_ollama import RemoteOllamaLLM
from config import OLLAMA_MODEL, OLLAMA_URL, OLLAMA_API_KEY

class RAGSystem:
    """
    A complete RAG (Retrieval-Augmented Generation) system using:
    - HuggingFace embeddings
    - Recursive character text splitting
    - Chroma vector database
    - Ollama for generation
    """
    
    def __init__(
        self,
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_documents",
        model_name: str = "Ollama",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        temperature: float = 0.7
    ):
        """
        Initialize the RAG system with all components.
        
        Args:
            persist_directory: Directory to persist Chroma database
            collection_name: Name for the Chroma collection
            ollama_model: Ollama model name (e.g., 'llama2', 'mistral', 'codellama')
            ollama_base_url: Base URL for Ollama server
            openai_api_key: OpenAI API key for GPT-4o
            embedding_model_name: HuggingFace embedding model name
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between chunks
            temperature: Temperature for Ollama generation
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.device = None

        if torch.cuda.is_available():
            self.device = "cuda"  # NVIDIA GPU
        elif torch.backends.mps.is_available():
            self.device = "mps"   # Apple Silicon GPU
        else:
            self.device = "cpu"   # Default: CPU

        logger.info(f"Initializing RAG system with model: {model_name}, collection: {collection_name}")
        logger.info(f"Using embedding model: {embedding_model_name}, chunk size: {chunk_size}, chunk overlap: {chunk_overlap}")
        logger.info(f"Persist directory: {self.persist_directory}")
        logger.info(f"Device set to: {self.device}")

        if model_name.lower() == "ollama":
            self.ollama_model = OLLAMA_MODEL
            self.ollama_base_url = OLLAMA_URL
            self.ollama_api_key = OLLAMA_API_KEY
        elif model_name.lower() == "remote_ollama":
            self.ollama_model = OLLAMA_MODEL
            self.ollama_base_url = OLLAMA_URL
            self.ollama_api_key = OLLAMA_API_KEY
        elif model_name.lower() == "openai":
            self.openai_api_key = os.getenv("OPENAI_API_KEY")

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize HuggingFace embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': self.device}  # Change to 'cuda' if GPU available
        )
        
        # Initialize Chroma vector store
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        if model_name.lower() == "ollama":
            # Initialize Ollama
            self.llm = OllamaLLM(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                seed=42,  # Optional seed for reproducibility
                api_key=self.ollama_api_key,
                temperature=temperature
            )
        elif model_name.lower() == "remote_ollama":
            logger.info("Using Remote Ollama model")
            logger.info(f"Connecting to Remote Ollama at {self.ollama_base_url} with model {self.ollama_model}")
            self.llm = RemoteOllamaLLM(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                headers={"Authorization": f"Bearer {self.ollama_api_key}"}
            )
        elif model_name.lower() == "openai":
            # Initialize GPT-4o
            self.llm = ChatOpenAI(
                model_name="gpt-4o",
                temperature=temperature,
                openai_api_key=self.openai_api_key
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}. Use 'Ollama' or 'OpenAI'.")
    
        # Ensure qa_chain is always defined
        self.qa_chain = None
        # Initialize retrieval chain
        self._setup_qa_chain()

        logger.info("RAG system initialized successfully")
        
    def _setup_qa_chain(self):
        """Setup the retrieval QA chain."""
        if len(self.vector_store.get()['documents']) == 0:
            logger.warning("Vector store is empty. Add documents first.")
            return
        
        # Create custom prompt template
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        The context provided is about Max Fink's work and may include references to his personal experiences or opinions.
        The context provided may refer to "Dr. Fink" or "Fink".  This is the same person as Max Fink.
        Do not respond with "I".  Responses should be in the third person.

        Context:
        {context}

        Question: {question}
        
        Answer:"""
        
        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create retrieval QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(search_kwargs={"k": 10}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        logger.info("QA chain initialized")
    
    def add_documents(self, documents: List[str], sources: Optional[List[str]] = None) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document texts
            sources: Optional list of source identifiers for each document
        """
        logger.info(f"Adding {len(documents)} documents to the vector store")
        if sources is None:
            sources = [f"doc_{i}" for i in range(len(documents))]
        
        all_chunks = []
        all_metadatas = []
        
        for doc_idx, (doc_text, source) in enumerate(zip(documents, sources)):
            # Split document into chunks
            chunks = self.text_splitter.split_text(doc_text)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": source,
                    "chunk_id": chunk_idx,
                    "doc_id": doc_idx
                })
        
        # Create Document objects
        doc_objects = [
            Document(page_content=chunk, metadata=metadata)
            for chunk, metadata in zip(all_chunks, all_metadatas)
        ]
        
        # Add documents to Chroma vector store
        self.vector_store.add_documents(doc_objects)
        
        # Persistence is automatic in Chroma >=0.4.x; no manual persist needed

        logger.info(f"Added {len(all_chunks)} chunks from {len(documents)} documents")

        # Setup QA chain after adding documents
        self._setup_qa_chain()
        logger.info(f"Added {len(documents)} documents to the vector store and initialized QA chain")
    
    def add_documents_from_files(self, file_paths: List[str]) -> None:
        """
        Add documents from text and PDF files.

        Args:
            file_paths: List of paths to text or PDF files
        """
        logger.info(f"Adding documents from {len(file_paths)} files")
        documents = []
        sources = []

        for file_path in file_paths:
            try:
                ext = os.path.splitext(file_path)[1].lower()
                if ext == ".pdf":
                    try:
                        reader = PdfReader(file_path)
                        content = ""
                        for page in reader.pages:
                            content += page.extract_text() or ""
                            if len(content) > 1000000:  # Limit to 1 million characters
                                logger.warning(f"File {file_path} is too large, truncating content.")
                        documents.append(content)
                        sources.append(file_path)
                    except Exception as e:
                        logger.error(f"Error reading PDF file {file_path}: {e}")
                else:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append(content)
                    sources.append(file_path)
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {e}")

        if documents:
            self.add_documents(documents, sources)
            logger.info(f"Added documents from {len(file_paths)} files")

    def query(self, question: str, return_sources: bool = True) -> Dict[str, Any]:
        """
        Query the RAG system.
        
        Args:
            question: The question to ask
            return_sources: Whether to return source documents
            
        Returns:
            Dictionary containing answer and optionally source documents
        """
        logger.info(f"Querying RAG system with question: {question}")
        if self.qa_chain is None:
            return {"error": "QA chain not initialized. Add documents first."}
        
        try:
            result = self.qa_chain.invoke({"query": question})
            response = {
                "answer": result["result"],
                "question": question
            }
            if return_sources and "source_documents" in result:
                sources = []
                for doc in result["source_documents"]:
                    sources.append({
                        "content": doc.page_content,
                        "metadata": doc.metadata
                    })
                response["sources"] = sources
            logger.info(f"Query successful, found {len(response.get('sources', []))} sources")
            return response
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {"error": f"Error during query: {str(e)}"}
    
    # def similarity_search(self, query: str, k: int = 4) -> List[Dict[str, Any]]:
    #     """
    #     Perform similarity search without generation.
        
    #     Args:
    #         query: Search query
    #         k: Number of results to return
            
    #     Returns:
    #         List of similar documents
    #     """
    #     if self.vector_store is None:
    #         return []
        
    #     try:
    #         docs = self.vector_store.similarity_search(query, k=k)
    #         results = []
            
    #         for doc in docs:
    #             results.append({
    #                 "content": doc.page_content,
    #                 "metadata": doc.metadata
    #             })
            
    #         return results
            
    #     except Exception as e:
    #         logger.error(f"Error during similarity search: {e}")
    #         return []
    
    def delete_collection(self):
        """Delete the Chroma collection."""
        try:
            self.vector_store.delete_collection()
            # Recreate empty vector store
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            logger.info(f"Deleted Chroma collection: {self.collection_name}")
            self.qa_chain = None
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the RAG system."""
        stats = {
            "collection_name": self.collection_name,
            "persist_directory": self.persist_directory,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "embedding_model": self.embeddings.model_name,
            "qa_chain_initialized": self.qa_chain is not None
        }
        
        try:
            # Get document count from Chroma
            collection_data = self.vector_store.get()
            stats["document_count"] = len(collection_data['documents'])
        except Exception:
            stats["document_count"] = "unknown"
        
        return stats
    
    def delete_document(self, doc_id: int) -> bool:
        """
        Delete a specific document (by doc_id) from the Chroma collection.
        Args:
            doc_id: The document ID to delete (as stored in metadata)
        Returns:
            True if deletion was successful, False otherwise
        """
        try:
            # Find all ids for chunks with the given doc_id
            collection_data = self.vector_store.get()
            ids_to_delete = [doc_id_str for doc_id_str, metadata in zip(collection_data['ids'], collection_data['metadatas']) if metadata.get('doc_id') == doc_id]
            if not ids_to_delete:
                logger.warning(f"No document found with doc_id={doc_id}")
                return False
            self.vector_store.delete(ids_to_delete)
            logger.info(f"Deleted document with doc_id={doc_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting document with doc_id={doc_id}: {e}")
            return False
        
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document (by doc_id) from the Chroma collection.
        
        Args:
            doc_id: The document ID to retrieve (as stored in metadata)
        
        Returns:
            Document content and metadata if found, None otherwise
        """
        try:
            # Find all documents with the given doc_id
            collection_data = self.vector_store.get()
            for doc, metadata in zip(collection_data['documents'], collection_data['metadatas']):
                if metadata.get('doc_id') == doc_id:
                    return {
                        "content": doc,
                        "metadata": metadata
                    }
            logger.warning(f"No document found with doc_id={doc_id}")
            return None
        except Exception as e:
            logger.error(f"Error retrieving document with doc_id={doc_id}: {e}")
            return None
        
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Retrieve all documents from the Chroma collection.
        
        Returns:
            List of all documents with their content and metadata
        """
        try:
            collection_data = self.vector_store.get()
            all_docs = []
            for doc, metadata in zip(collection_data['documents'], collection_data['metadatas']):
                all_docs.append({
                    "content": doc,
                    "metadata": metadata
                })
            return all_docs
        except Exception as e:
            logger.error(f"Error retrieving all documents: {e}")
            return []

    def get_chunk(self, doc_id: int, chunk_id: int) -> Optional[Dict[str, Any]]:
            """
            Retrieve a specific chunk from a document in the Chroma collection.
            Args:
                doc_id: The document ID to retrieve (as stored in metadata)
                chunk_id: The chunk ID to retrieve
            Returns:
                Chunk content and metadata if found, None otherwise
            """
            try:
                collection_data = self.vector_store.get()
                for doc, metadata in zip(collection_data['documents'], collection_data['metadatas']):
                    if metadata.get('doc_id') == doc_id and metadata.get('chunk_id') == chunk_id:
                        return {
                            "content": doc,
                            "metadata": metadata
                        }
                logger.warning(f"No chunk found with doc_id={doc_id} and chunk_id={chunk_id}")
                return None
            except Exception as e:
                logger.error(f"Error retrieving chunk with doc_id={doc_id}, chunk_id={chunk_id}: {e}")
                return None
