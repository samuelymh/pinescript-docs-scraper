"""Utility functions for PineScript RAG Server.

Provides token counting, hashing, code detection, and logging setup.
"""
import hashlib
import logging
import re
from typing import Optional
import tiktoken


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "module": "%(name)s", "message": "%(message)s"}',
        datefmt="%Y-%m-%dT%H:%M:%S"
    )


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens in text using tiktoken.
    
    Args:
        text: Text to count tokens for
        model: Model name to use for encoding (default: gpt-4o uses cl100k_base)
    
    Returns:
        Number of tokens
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        # Fallback to cl100k_base for unknown models
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))


def hash_string(text: str) -> str:
    """Generate SHA256 hash of string.
    
    Args:
        text: Text to hash
    
    Returns:
        Hex digest of SHA256 hash
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def detect_code_snippets(content: str) -> bool:
    """Detect if content contains PineScript code snippets.
    
    Handles three patterns:
    1. Triple backtick blocks: ```pine ... ```
    2. Single backticks: `code`
    3. Pine Script® marker: "Pine Script®\\nCopied\\n`code`"
    
    Args:
        content: Document content to check
    
    Returns:
        True if code snippets are detected
    """
    # Pattern 1: Triple backtick blocks with optional pine language marker
    if re.search(r'```(?:pine)?[\s\S]*?```', content):
        return True
    
    # Pattern 2: Single backticks (exclude very short ones to reduce false positives)
    # Look for backticks with at least 3 characters inside
    if re.search(r'`[^`]{3,}`', content):
        return True
    
    # Pattern 3: Pine Script® marker followed by code
    if re.search(r'Pine Script®\s*\n\s*Copied\s*\n\s*`', content):
        return True
    
    return False


def extract_code_blocks(content: str) -> list[str]:
    """Extract all code blocks from content.
    
    Args:
        content: Document content
    
    Returns:
        List of extracted code snippets
    """
    code_blocks = []
    
    # Extract triple backtick blocks
    triple_blocks = re.findall(r'```(?:pine)?([\s\S]*?)```', content)
    code_blocks.extend([block.strip() for block in triple_blocks if block.strip()])
    
    # Extract Pine Script® marked blocks
    pine_marked = re.findall(
        r'Pine Script®\s*\n\s*Copied\s*\n\s*`([^`]+)`',
        content
    )
    code_blocks.extend([block.strip() for block in pine_marked if block.strip()])
    
    return code_blocks


def generate_doc_id(source_filename: str, chunk_index: int = 0) -> str:
    """Generate deterministic document ID.
    
    Args:
        source_filename: Source file name
        chunk_index: Chunk index (0-based)
    
    Returns:
        SHA256 hash of filename + chunk_index
    """
    composite = f"{source_filename}:{chunk_index}"
    return hash_string(composite)


# Logger for this module
logger = logging.getLogger(__name__)
