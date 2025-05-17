import pytest
import asyncio
from unittest.mock import patch, MagicMock
from app.services.rag_service import RAGService


@pytest.fixture
def rag_service():
    """Create a RAG service instance for testing."""
    with patch('app.services.rag_service.HuggingFaceEmbeddings') as mock_embeddings:
        with patch('app.services.rag_service.FAISS') as mock_faiss:
            # Mock the embedding model
            mock_embedding_instance = MagicMock()
            mock_embeddings.return_value = mock_embedding_instance
            
            # Mock the vector store
            mock_vector_store = MagicMock()
            mock_faiss.load_local.return_value = mock_vector_store
            mock_faiss.from_texts.return_value = mock_vector_store
            
            # Create service instance
            service = RAGService()
            
            # Set mocked vector store
            service.vector_store = mock_vector_store
            
            yield service


@pytest.mark.asyncio
async def test_add_documents(rag_service):
    """Test adding documents to the RAG service."""
    # Test data
    texts = ["This is a test document", "Another test document"]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    
    # Mock the vector store's add_documents method
    rag_service.vector_store.add_documents = MagicMock()
    
    # Call the method
    result = await rag_service.add_documents(texts, metadatas)
    
    # Check that add_documents was called
    assert rag_service.vector_store.add_documents.called
    
    # Check the result
    assert result["status"] == "success"
    assert "documents_added" in result


@pytest.mark.asyncio
async def test_search(rag_service):
    """Test searching for documents."""
    # Mock search results
    mock_doc1 = MagicMock()
    mock_doc1.page_content = "Test content 1"
    mock_doc1.metadata = {"source": "test1"}
    
    mock_doc2 = MagicMock()
    mock_doc2.page_content = "Test content 2"
    mock_doc2.metadata = {"source": "test2"}
    
    # Set up the mock return value for similarity_search_with_score
    rag_service.vector_store.similarity_search_with_score.return_value = [
        (mock_doc1, 0.8),
        (mock_doc2, 0.6)
    ]
    
    # Call the method
    results = await rag_service.search("test query", k=2)
    
    # Check that similarity_search_with_score was called with the right arguments
    rag_service.vector_store.similarity_search_with_score.assert_called_with("test query", k=2)
    
    # Check the results
    assert len(results) == 2
    assert results[0]["content"] == "Test content 1"
    assert results[0]["metadata"] == {"source": "test1"}
    assert results[0]["score"] == 0.8
    assert results[1]["content"] == "Test content 2"
    assert results[1]["metadata"] == {"source": "test2"}
    assert results[1]["score"] == 0.6


@pytest.mark.asyncio
async def test_add_strategy_template(rag_service):
    """Test adding a strategy template."""
    # Test data
    strategy_type = "mean_reversion"
    template = "Buy when RSI is below 30, sell when RSI is above 70"
    metadata = {"author": "test_user", "created_at": "2025-05-16"}
    
    # Mock the add_documents method
    rag_service.add_documents = MagicMock()
    rag_service.add_documents.return_value = {"status": "success"}
    
    # Call the method
    result = await rag_service.add_strategy_template(strategy_type, template, metadata)
    
    # Check that add_documents was called with the right arguments
    rag_service.add_documents.assert_called_once()
    call_args = rag_service.add_documents.call_args[0]
    assert call_args[0] == [template]
    assert call_args[1][0]["type"] == "strategy_template"
    assert call_args[1][0]["strategy_type"] == strategy_type
    assert call_args[1][0]["author"] == "test_user"
    
    # Check the result
    assert result["status"] == "success"


@pytest.mark.asyncio
async def test_get_relevant_strategy_templates(rag_service):
    """Test retrieving relevant strategy templates."""
    # Mock the search method
    rag_service.search = MagicMock()
    rag_service.search.return_value = [
        {
            "content": "Template 1",
            "metadata": {"type": "strategy_template", "strategy_type": "mean_reversion"},
            "score": 0.9
        },
        {
            "content": "Template 2",
            "metadata": {"type": "strategy_template", "strategy_type": "momentum"},
            "score": 0.7
        },
        {
            "content": "Not a template",
            "metadata": {"type": "other"},
            "score": 0.5
        }
    ]
    
    # Call the method
    results = await rag_service.get_relevant_strategy_templates(
        "mean_reversion", 
        "Strategy for ETH when oversold", 
        k=3
    )
    
    # Check that search was called with the right arguments
    rag_service.search.assert_called_once()
    
    # Check the results
    assert len(results) == 2  # Only the strategy templates should be returned
    assert results[0]["content"] == "Template 1"
    assert results[0]["metadata"]["strategy_type"] == "mean_reversion"
    assert results[1]["content"] == "Template 2"
    assert results[1]["metadata"]["strategy_type"] == "momentum"
