"""Data models for PineScript RAG Server.

Defines Pydantic models for API requests/responses and internal data structures.
"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field, ConfigDict


class Document(BaseModel):
    """Document chunk with embedding and metadata."""
    
    id: str = Field(..., description="Deterministic ID: sha256(source_filename + chunk_index)")
    content: str = Field(..., description="Chunk content (full file or section)")
    source_filename: str = Field(..., description="Original source file name")
    chunk_index: int = Field(..., ge=0, description="0-based chunk number")
    chunk_count: int = Field(..., ge=1, description="Total chunks from this file")
    section_heading: Optional[str] = Field(None, description="H1/H2 heading if chunked")
    token_count: int = Field(..., ge=0, description="Token estimate for this chunk")
    code_snippet: bool = Field(..., description="True if contains code (any format)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "id": "abc123...",
                "content": "## Welcome to PineScript...",
                "source_filename": "processed_1_welcome_20251031_113440.md",
                "chunk_index": 0,
                "chunk_count": 1,
                "section_heading": None,
                "token_count": 1234,
                "code_snippet": True,
                "metadata": {"processed_timestamp": "2025-11-01T00:00:00Z"},
                "embedding": None
            }
        }
    )


class FileManifest(BaseModel):
    """File manifest for tracking indexed documents."""
    
    filename: str = Field(..., description="Source filename")
    content_hash: str = Field(..., description="SHA256 hash of file content")
    last_indexed: datetime = Field(..., description="Last indexing timestamp")
    doc_id: str = Field(..., description="References documents.id")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "processed_1_welcome_20251031_113440.md",
                "content_hash": "def456...",
                "last_indexed": "2025-11-01T00:00:00Z",
                "doc_id": "abc123..."
            }
        }
    )


class Source(BaseModel):
    """Source document reference with provenance."""
    
    filename: str = Field(..., description="Source filename")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    excerpt: str = Field(..., description="First 200 chars of content")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "processed_41_strategies_20251031_113440.md",
                "similarity_score": 0.92,
                "excerpt": "Strategies in PineScript allow you to backtest..."
            }
        }
    )


class ChatRequest(BaseModel):
    """Request model for /chat endpoint."""
    
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    conversation_history: Optional[List[Dict[str, str]]] = Field(
        None,
        description="Previous conversation turns with 'role' and 'content'"
    )
    max_context_docs: int = Field(
        default=8,
        ge=1,
        le=20,
        description="Maximum context documents to retrieve"
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=1.0,
        description="LLM temperature (0=deterministic, 1=creative)"
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "How do I create a simple moving average indicator?",
                "conversation_history": None,
                "max_context_docs": 8,
                "temperature": 0.1
            }
        }
    )


class ChatResponse(BaseModel):
    """Response model for /chat endpoint."""
    
    response: str = Field(..., description="Generated answer with code")
    sources: List[Source] = Field(..., description="Source documents with provenance")
    tokens_used: Dict[str, int] = Field(
        ...,
        description="Token usage breakdown (prompt, completion, total)"
    )
    model: str = Field(..., description="LLM model used (gpt-4o or gpt-4o-mini)")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "response": "To create a simple moving average...\n```pine\n//@version=6\n...",
                "sources": [
                    {
                        "filename": "processed_3_first-indicator_20251031_113440.md",
                        "similarity_score": 0.94,
                        "excerpt": "Your first indicator will use..."
                    }
                ],
                "tokens_used": {"prompt": 1500, "completion": 400, "total": 1900},
                "model": "gpt-4o"
            }
        }
    )
