"""Unit tests for OpenAI embedding client.

Tests batching, retry logic, and error handling with mocked OpenAI client.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from openai import OpenAI
from openai.types import CreateEmbeddingResponse, Embedding
from server.embed_client import (
    generate_embeddings_batch,
    generate_embeddings_chunked,
    generate_single_embedding,
    get_embedding_dimension,
    estimate_embedding_cost
)


# Test fixtures

@pytest.fixture
def mock_openai_response():
    """Mock OpenAI embeddings API response."""
    def create_response(num_embeddings=1, dimension=3072):
        """Create a mock response with specified number of embeddings."""
        embeddings = []
        for i in range(num_embeddings):
            # Create mock embedding with specified dimension
            embedding = Embedding(
                embedding=[0.1] * dimension,
                index=i,
                object="embedding"
            )
            embeddings.append(embedding)
        
        response = CreateEmbeddingResponse(
            data=embeddings,
            model="text-embedding-3-large",
            object="list",
            usage={"prompt_tokens": 100, "total_tokens": 100}
        )
        return response
    
    return create_response


@pytest.fixture
def mock_openai_client(mock_openai_response):
    """Mock OpenAI client with embeddings API."""
    client = Mock(spec=OpenAI)
    embeddings_api = Mock()
    
    # Configure create method to return mock response
    embeddings_api.create = Mock(return_value=mock_openai_response(1))
    client.embeddings = embeddings_api
    
    return client


# Tests for single batch embedding generation

@patch('server.embed_client.init_openai_client')
def test_generate_embeddings_batch_single_text(mock_init, mock_openai_client, mock_openai_response):
    """Test generating embeddings for a single text."""
    mock_init.return_value = mock_openai_client
    mock_openai_client.embeddings.create.return_value = mock_openai_response(1)
    
    texts = ["Test text"]
    embeddings = generate_embeddings_batch(texts)
    
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 3072  # text-embedding-3-large dimension
    
    # Verify API was called with correct parameters
    mock_openai_client.embeddings.create.assert_called_once()
    call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
    assert call_kwargs["input"] == texts
    assert "text-embedding-3" in call_kwargs["model"]


@patch('server.embed_client.init_openai_client')
def test_generate_embeddings_batch_multiple_texts(mock_init, mock_openai_client, mock_openai_response):
    """Test generating embeddings for multiple texts."""
    mock_init.return_value = mock_openai_client
    mock_openai_client.embeddings.create.return_value = mock_openai_response(5)
    
    texts = [f"Text {i}" for i in range(5)]
    embeddings = generate_embeddings_batch(texts)
    
    assert len(embeddings) == 5
    assert all(len(emb) == 3072 for emb in embeddings)


@patch('server.embed_client.init_openai_client')
def test_generate_embeddings_batch_empty_list(mock_init, mock_openai_client):
    """Test that empty list returns empty result."""
    mock_init.return_value = mock_openai_client
    
    embeddings = generate_embeddings_batch([])
    
    assert embeddings == []
    mock_openai_client.embeddings.create.assert_not_called()


# Tests for chunked embedding generation

@patch('server.embed_client.generate_embeddings_batch')
def test_generate_embeddings_chunked_single_batch(mock_batch):
    """Test chunked generation with single batch."""
    mock_batch.return_value = [[0.1] * 3072 for _ in range(50)]
    
    texts = [f"Text {i}" for i in range(50)]
    embeddings = generate_embeddings_chunked(texts, batch_size=100)
    
    assert len(embeddings) == 50
    mock_batch.assert_called_once()


@patch('server.embed_client.generate_embeddings_batch')
def test_generate_embeddings_chunked_multiple_batches(mock_batch):
    """Test chunked generation splits into multiple batches."""
    # Mock returns different embeddings for each call
    mock_batch.side_effect = [
        [[0.1] * 3072 for _ in range(100)],
        [[0.2] * 3072 for _ in range(100)],
        [[0.3] * 3072 for _ in range(50)]
    ]
    
    texts = [f"Text {i}" for i in range(250)]
    embeddings = generate_embeddings_chunked(texts, batch_size=100)
    
    assert len(embeddings) == 250
    assert mock_batch.call_count == 3
    
    # Verify batch sizes
    call_args_list = mock_batch.call_args_list
    assert len(call_args_list[0][0][0]) == 100  # First batch: 100 texts
    assert len(call_args_list[1][0][0]) == 100  # Second batch: 100 texts
    assert len(call_args_list[2][0][0]) == 50   # Third batch: 50 texts


@patch('server.embed_client.generate_embeddings_batch')
def test_generate_embeddings_chunked_empty_list(mock_batch):
    """Test chunked generation with empty list."""
    embeddings = generate_embeddings_chunked([])
    
    assert embeddings == []
    mock_batch.assert_not_called()


# Tests for retry logic

@patch('server.embed_client.init_openai_client')
def test_retry_on_api_error(mock_init, mock_openai_client, mock_openai_response):
    """Test that API errors trigger retry with exponential backoff."""
    mock_init.return_value = mock_openai_client
    
    # First two calls fail, third succeeds
    mock_openai_client.embeddings.create.side_effect = [
        Exception("API Error"),
        Exception("Rate limit"),
        mock_openai_response(1)
    ]
    
    texts = ["Test text"]
    embeddings = generate_embeddings_batch(texts)
    
    # Should succeed after retries
    assert len(embeddings) == 1
    assert mock_openai_client.embeddings.create.call_count == 3


@patch('server.embed_client.init_openai_client')
def test_retry_exhausted(mock_init, mock_openai_client):
    """Test that retry exhausts after max attempts."""
    mock_init.return_value = mock_openai_client
    
    # All calls fail
    mock_openai_client.embeddings.create.side_effect = Exception("Persistent error")
    
    texts = ["Test text"]
    
    with pytest.raises(Exception) as exc_info:
        generate_embeddings_batch(texts)
    
    assert "Persistent error" in str(exc_info.value)
    # Should retry 3 times (initial + 2 retries based on tenacity config)
    assert mock_openai_client.embeddings.create.call_count == 3


# Tests for single embedding generation

@patch('server.embed_client.generate_embeddings_batch')
def test_generate_single_embedding(mock_batch):
    """Test convenience function for single embedding."""
    mock_batch.return_value = [[0.1] * 3072]
    
    embedding = generate_single_embedding("Test query")
    
    assert len(embedding) == 3072
    mock_batch.assert_called_once_with(["Test query"], model=None)


# Tests for embedding dimension

def test_get_embedding_dimension_known_models():
    """Test dimension retrieval for known models."""
    assert get_embedding_dimension("text-embedding-3-large") == 3072
    assert get_embedding_dimension("text-embedding-3-small") == 1536
    assert get_embedding_dimension("text-embedding-ada-002") == 1536


def test_get_embedding_dimension_unknown_model():
    """Test dimension defaults for unknown models."""
    assert get_embedding_dimension("unknown-model") == 1536


def test_get_embedding_dimension_uses_config():
    """Test that None uses config default."""
    with patch('server.embed_client.get_config') as mock_config:
        mock_config.return_value.embedding_model = "text-embedding-3-large"
        
        dimension = get_embedding_dimension(None)
        assert dimension == 3072


# Tests for cost estimation

def test_estimate_embedding_cost_basic():
    """Test basic cost estimation."""
    cost = estimate_embedding_cost(
        num_texts=1000,
        avg_tokens_per_text=500,
        model="text-embedding-3-large"
    )
    
    assert cost["model"] == "text-embedding-3-large"
    assert cost["num_texts"] == 1000
    assert cost["avg_tokens_per_text"] == 500
    assert cost["total_tokens"] == 500000
    assert cost["price_per_million_tokens"] == 0.13
    assert cost["estimated_cost_usd"] > 0
    assert cost["estimated_cost_usd"] == round((500000 / 1_000_000) * 0.13, 4)


def test_estimate_embedding_cost_different_models():
    """Test cost estimation for different models."""
    large_cost = estimate_embedding_cost(100, 500, "text-embedding-3-large")
    small_cost = estimate_embedding_cost(100, 500, "text-embedding-3-small")
    
    # text-embedding-3-small should be cheaper
    assert small_cost["estimated_cost_usd"] < large_cost["estimated_cost_usd"]


def test_estimate_embedding_cost_scaling():
    """Test that cost scales linearly with input."""
    cost_100 = estimate_embedding_cost(100, 500)
    cost_1000 = estimate_embedding_cost(1000, 500)
    
    # Cost should scale 10x
    ratio = cost_1000["estimated_cost_usd"] / cost_100["estimated_cost_usd"]
    assert abs(ratio - 10.0) < 0.01  # Allow small floating point error


# Tests for error handling

@patch('server.embed_client.init_openai_client')
def test_handle_invalid_api_key(mock_init, mock_openai_client):
    """Test handling of invalid API key error."""
    mock_init.return_value = mock_openai_client
    mock_openai_client.embeddings.create.side_effect = Exception("Invalid API key")
    
    with pytest.raises(Exception) as exc_info:
        generate_embeddings_batch(["Test"])
    
    assert "Invalid API key" in str(exc_info.value)


@patch('server.embed_client.init_openai_client')
def test_chunked_embedding_partial_failure(mock_init, mock_openai_client, mock_openai_response):
    """Test that chunked generation fails fast on batch error."""
    mock_init.return_value = mock_openai_client
    
    # First batch succeeds, second fails
    mock_openai_client.embeddings.create.side_effect = [
        mock_openai_response(100),
        Exception("Batch failed")
    ]
    
    texts = [f"Text {i}" for i in range(200)]
    
    with pytest.raises(Exception):
        generate_embeddings_chunked(texts, batch_size=100)


# Integration-style tests

@pytest.mark.skip(reason="Requires actual OpenAI API key")
def test_real_embedding_generation():
    """Integration test with real OpenAI API (requires valid API key)."""
    # This would test against the real API
    # Skipped by default to avoid API costs
    texts = ["This is a test."]
    embeddings = generate_embeddings_batch(texts)
    
    assert len(embeddings) == 1
    assert len(embeddings[0]) == 3072


# Tests for model parameter passing

@patch('server.embed_client.init_openai_client')
def test_custom_model_parameter(mock_init, mock_openai_client, mock_openai_response):
    """Test that custom model parameter is passed correctly."""
    mock_init.return_value = mock_openai_client
    mock_openai_client.embeddings.create.return_value = mock_openai_response(1, 1536)
    
    generate_embeddings_batch(["Test"], model="text-embedding-3-small")
    
    call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-3-small"


@patch('server.embed_client.init_openai_client')
@patch('server.embed_client.get_config')
def test_default_model_from_config(mock_config, mock_init, mock_openai_client, mock_openai_response):
    """Test that default model comes from config."""
    mock_config.return_value.embedding_model = "text-embedding-3-large"
    mock_init.return_value = mock_openai_client
    mock_openai_client.embeddings.create.return_value = mock_openai_response(1)
    
    generate_embeddings_batch(["Test"])
    
    call_kwargs = mock_openai_client.embeddings.create.call_args.kwargs
    assert call_kwargs["model"] == "text-embedding-3-large"
