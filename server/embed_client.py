"""OpenAI embedding client for PineScript RAG Server.

Provides batched embedding generation with retries and concurrency controls.
"""
from typing import List, Dict, Any, Optional
import logging
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from server.config import get_config

logger = logging.getLogger(__name__)


# Singleton OpenAI client
_openai_client: Optional[OpenAI] = None


def init_openai_client() -> OpenAI:
    """Initialize and return OpenAI client singleton.
    
    Returns:
        Configured OpenAI client
    """
    global _openai_client
    
    if _openai_client is None:
        config = get_config()
        _openai_client = OpenAI(api_key=config.openai_api_key)
        logger.info("OpenAI client initialized")
    
    return _openai_client


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)
def generate_embeddings_batch(
    texts: List[str],
    model: Optional[str] = None
) -> List[List[float]]:
    """Generate embeddings for a batch of texts with retry logic.
    
    Uses tenacity for exponential backoff retry on API failures.
    Handles rate limits, timeouts, and transient errors.
    
    Args:
        texts: List of text strings to embed (max 2048 per batch for OpenAI)
        model: Embedding model name (defaults to config.embedding_model)
    
    Returns:
        List of embedding vectors in same order as input texts
        
    Raises:
        Exception: If all retry attempts fail
    """
    if not texts:
        return []
    
    config = get_config()
    model = model or config.embedding_model
    client = init_openai_client()
    
    try:
        logger.info(f"Generating embeddings for {len(texts)} texts using {model}")
        
        # Call OpenAI embeddings API
        response = client.embeddings.create(
            input=texts,
            model=model
        )
        
        # Extract embeddings in order
        embeddings = [item.embedding for item in response.data]
        
        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings
    
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise


def generate_embeddings_chunked(
    texts: List[str],
    batch_size: int = 100,
    model: Optional[str] = None
) -> List[List[float]]:
    """Generate embeddings for texts in batches with rate limiting.
    
    Splits large lists into smaller batches to respect API limits and
    improve reliability. Each batch is retried independently.
    
    Args:
        texts: List of text strings to embed
        batch_size: Number of texts per API call (default 100, max 2048 for OpenAI)
        model: Embedding model name (defaults to config.embedding_model)
    
    Returns:
        List of embedding vectors in same order as input texts
        
    Raises:
        Exception: If any batch fails after retries
    """
    if not texts:
        return []
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
        
        try:
            embeddings = generate_embeddings_batch(batch, model=model)
            all_embeddings.extend(embeddings)
        except Exception as e:
            logger.error(f"Batch {batch_num} failed after retries: {e}")
            raise
    
    logger.info(f"Completed embedding generation for {len(all_embeddings)} texts")
    return all_embeddings


def generate_single_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """Generate embedding for a single text.
    
    Convenience wrapper for single text embedding (e.g., query embedding).
    
    Args:
        text: Text string to embed
        model: Embedding model name (defaults to config.embedding_model)
    
    Returns:
        Embedding vector
        
    Raises:
        Exception: If embedding generation fails
    """
    embeddings = generate_embeddings_batch([text], model=model)
    return embeddings[0]


def get_embedding_dimension(model: Optional[str] = None) -> int:
    """Get the dimension of embeddings for a given model.
    
    Args:
        model: Embedding model name (defaults to config.embedding_model)
    
    Returns:
        Embedding dimension (e.g., 3072 for text-embedding-3-large)
    """
    config = get_config()
    model = model or config.embedding_model
    
    # Known dimensions for OpenAI models
    dimensions = {
        "text-embedding-3-large": 3072,
        "text-embedding-3-small": 1536,
        "text-embedding-ada-002": 1536
    }
    
    return dimensions.get(model, 1536)  # Default to 1536


def estimate_embedding_cost(
    num_texts: int,
    avg_tokens_per_text: int = 500,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Estimate cost for embedding generation.
    
    Args:
        num_texts: Number of texts to embed
        avg_tokens_per_text: Average tokens per text
        model: Embedding model name (defaults to config.embedding_model)
    
    Returns:
        Dict with cost estimate details
    """
    config = get_config()
    model = model or config.embedding_model
    
    # OpenAI pricing (as of 2024)
    # text-embedding-3-large: $0.13 per 1M tokens
    # text-embedding-3-small: $0.02 per 1M tokens
    # text-embedding-ada-002: $0.10 per 1M tokens
    
    pricing = {
        "text-embedding-3-large": 0.13,
        "text-embedding-3-small": 0.02,
        "text-embedding-ada-002": 0.10
    }
    
    price_per_million = pricing.get(model, 0.10)
    total_tokens = num_texts * avg_tokens_per_text
    estimated_cost = (total_tokens / 1_000_000) * price_per_million
    
    return {
        "model": model,
        "num_texts": num_texts,
        "avg_tokens_per_text": avg_tokens_per_text,
        "total_tokens": total_tokens,
        "price_per_million_tokens": price_per_million,
        "estimated_cost_usd": round(estimated_cost, 4)
    }
