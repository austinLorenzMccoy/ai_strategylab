import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGService:
    """Service for Retrieval-Augmented Generation (RAG)."""
    
    def __init__(self):
        """Initialize the RAG service."""
        self.vector_db_path = settings.VECTOR_DB_PATH
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.embedding_model = None
        self.vector_store = None
        
        # Ensure vector DB directory exists
        os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
        
        # Initialize embedding model
        self._init_embedding_model()
        
        # Load or create vector store
        self._init_vector_store()
    
    def _init_embedding_model(self):
        """Initialize the embedding model."""
        try:
            logger.info(f"Initializing embedding model: {self.embedding_model_name}")
            self.embedding_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name, 
                cache_folder="./data/models"
            )
        except Exception as e:
            logger.error(f"Error initializing embedding model: {str(e)}")
            raise
    
    def _init_vector_store(self):
        """Initialize the vector store."""
        try:
            if os.path.exists(f"{self.vector_db_path}/index.faiss"):
                logger.info(f"Loading existing vector store from {self.vector_db_path}")
                self.vector_store = FAISS.load_local(
                    self.vector_db_path, 
                    self.embedding_model
                )
            else:
                logger.info("No existing vector store found. Creating empty vector store.")
                # Create empty vector store with a dummy document to initialize
                dummy_texts = ["Initialization document for vector store"]
                self.vector_store = FAISS.from_texts(
                    dummy_texts, 
                    self.embedding_model
                )
                # Save the initialized vector store
                self.vector_store.save_local(self.vector_db_path)
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise
    
    async def add_documents(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        """Add documents to the vector store."""
        try:
            logger.info(f"Adding {len(texts)} documents to vector store")
            
            # Create text splitter for chunking
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            
            # Split texts into chunks
            documents = []
            for i, text in enumerate(texts):
                chunks = text_splitter.split_text(text)
                metadata = metadatas[i] if metadatas and i < len(metadatas) else {}
                
                for chunk in chunks:
                    documents.append(Document(page_content=chunk, metadata=metadata))
            
            # Add documents to vector store
            self.vector_store.add_documents(documents)
            
            # Save updated vector store
            self.vector_store.save_local(self.vector_db_path)
            
            logger.info(f"Successfully added documents to vector store")
            return {"status": "success", "documents_added": len(documents)}
        
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            raise
    
    async def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant documents in the vector store."""
        try:
            logger.info(f"Searching for documents relevant to: {query}")
            
            # Search vector store
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": float(score)
                })
            
            logger.info(f"Found {len(formatted_results)} relevant documents")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error searching vector store: {str(e)}")
            raise
    
    async def add_strategy_template(self, strategy_type: str, template: str, metadata: Dict[str, Any]):
        """Add a strategy template to the vector store."""
        try:
            logger.info(f"Adding template for strategy type: {strategy_type}")
            
            texts = [template]
            metadatas = [{
                "type": "strategy_template",
                "strategy_type": strategy_type,
                **metadata
            }]
            
            await self.add_documents(texts, metadatas)
            
            logger.info(f"Successfully added template for strategy type: {strategy_type}")
            return {"status": "success"}
        
        except Exception as e:
            logger.error(f"Error adding strategy template: {str(e)}")
            raise
    
    async def get_relevant_strategy_templates(self, strategy_type: str, description: str, k: int = 3):
        """Get relevant strategy templates based on type and description."""
        try:
            logger.info(f"Retrieving templates for strategy type: {strategy_type}")
            
            # Construct search query
            query = f"Strategy type: {strategy_type}. Description: {description}"
            
            # Search for relevant templates
            results = await self.search(query, k=k)
            
            # Filter for strategy templates
            strategy_templates = [
                result for result in results 
                if result.get("metadata", {}).get("type") == "strategy_template"
            ]
            
            logger.info(f"Found {len(strategy_templates)} relevant strategy templates")
            return strategy_templates
        
        except Exception as e:
            logger.error(f"Error retrieving strategy templates: {str(e)}")
            raise


# Create a singleton instance
rag_service = RAGService()