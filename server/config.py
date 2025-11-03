"""Configuration management for PineScript RAG Server.

Loads and validates environment variables using Pydantic Settings.
"""
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Config(BaseSettings):
    """Application configuration loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Supabase configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_service_role_key: str = Field(..., description="Supabase service role key")
    
    # OpenAI configuration
    openai_api_key: str = Field(..., description="OpenAI API key")
    embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model"
    )
    llm_model: str = Field(
        default="gpt-4o",
        description="OpenAI LLM model for generation"
    )
    
    # Database configuration
    rag_vector_table: str = Field(
        default="documents",
        description="Supabase table name for vector storage"
    )
    
    # Server configuration
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    admin_api_key: Optional[str] = Field(
        default=None,
        description="Admin API key for /internal/* endpoints"
    )
    
    # RAG configuration
    max_context_docs: int = Field(
        default=12,
        description="Maximum documents to retrieve for context"
    )
    chunk_token_threshold: int = Field(
        default=1500,
        description="Token count threshold for chunking files"
    )
    chunk_overlap_tokens: int = Field(
        default=150,
        description="Token overlap between chunks"
    )
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is valid."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v


# Singleton instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the singleton configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config
