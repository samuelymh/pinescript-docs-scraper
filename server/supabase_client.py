"""Supabase client for PineScript RAG Server.

Provides functions for document and manifest operations with Supabase database.
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
import logging
from supabase import create_client, Client
from server.config import get_config
from server.models import Document, FileManifest

logger = logging.getLogger(__name__)


# Singleton Supabase client
_supabase_client: Optional[Client] = None


def init_supabase_client() -> Client:
    """Initialize and return Supabase client singleton.
    
    Returns:
        Configured Supabase client
    """
    global _supabase_client
    
    if _supabase_client is None:
        config = get_config()
        _supabase_client = create_client(
            config.supabase_url,
            config.supabase_service_role_key
        )
        logger.info("Supabase client initialized")
    
    return _supabase_client


def upsert_documents(documents: List[Document]) -> Dict[str, Any]:
    """Upsert document chunks to Supabase.
    
    Uses upsert to handle both inserts and updates based on document ID.
    
    Args:
        documents: List of Document objects to upsert
    
    Returns:
        Dict with success status and count of upserted documents
        
    Raises:
        Exception: If upsert operation fails
    """
    if not documents:
        logger.warning("No documents to upsert")
        return {"success": True, "count": 0}
    
    client = init_supabase_client()
    config = get_config()
    
    # Convert documents to dict format for Supabase
    records = []
    for doc in documents:
        record = {
            "id": doc.id,
            "content": doc.content,
            "source_filename": doc.source_filename,
            "chunk_index": doc.chunk_index,
            "chunk_count": doc.chunk_count,
            "section_heading": doc.section_heading,
            "token_count": doc.token_count,
            "code_snippet": doc.code_snippet,
            "metadata": doc.metadata,
            "embedding": doc.embedding
        }
        records.append(record)
    
    try:
        # Batch upsert
        result = client.table(config.rag_vector_table).upsert(
            records,
            on_conflict="id"
        ).execute()
        
        count = len(result.data) if result.data else 0
        logger.info(f"Upserted {count} documents to Supabase")
        
        return {"success": True, "count": count}
    
    except Exception as e:
        logger.error(f"Failed to upsert documents: {e}")
        raise


def update_manifest(manifest_entries: List[FileManifest]) -> Dict[str, Any]:
    """Update file manifest in Supabase.
    
    Args:
        manifest_entries: List of FileManifest objects to upsert
    
    Returns:
        Dict with success status and count of updated entries
        
    Raises:
        Exception: If update operation fails
    """
    if not manifest_entries:
        logger.warning("No manifest entries to update")
        return {"success": True, "count": 0}
    
    client = init_supabase_client()
    
    # Convert to dict format
    records = []
    for entry in manifest_entries:
        record = {
            "filename": entry.filename,
            "content_hash": entry.content_hash,
            "last_indexed": entry.last_indexed.isoformat(),
            "doc_id": entry.doc_id,
            "chunk_count": 1  # Will be updated based on actual chunks
        }
        records.append(record)
    
    try:
        # Batch upsert to file_manifest
        result = client.table("file_manifest").upsert(
            records,
            on_conflict="filename"
        ).execute()
        
        count = len(result.data) if result.data else 0
        logger.info(f"Updated {count} manifest entries in Supabase")
        
        return {"success": True, "count": count}
    
    except Exception as e:
        logger.error(f"Failed to update manifest: {e}")
        raise


def fetch_manifest() -> Dict[str, FileManifest]:
    """Fetch existing file manifest from Supabase.
    
    Returns:
        Dict mapping filename to FileManifest object
        
    Raises:
        Exception: If fetch operation fails
    """
    client = init_supabase_client()
    
    try:
        result = client.table("file_manifest").select("*").execute()
        
        manifest_dict = {}
        if result.data:
            for row in result.data:
                manifest_dict[row["filename"]] = FileManifest(
                    filename=row["filename"],
                    content_hash=row["content_hash"],
                    last_indexed=datetime.fromisoformat(row["last_indexed"]),
                    doc_id=row["doc_id"]
                )
        
        logger.info(f"Fetched {len(manifest_dict)} manifest entries from Supabase")
        return manifest_dict
    
    except Exception as e:
        logger.error(f"Failed to fetch manifest: {e}")
        raise


def delete_documents_by_filename(filename: str) -> Dict[str, Any]:
    """Delete all document chunks for a given source filename.
    
    Used when a file is modified and needs to be re-indexed.
    
    Args:
        filename: Source filename to delete
    
    Returns:
        Dict with success status and count of deleted documents
        
    Raises:
        Exception: If delete operation fails
    """
    client = init_supabase_client()
    config = get_config()
    
    try:
        result = client.table(config.rag_vector_table).delete().eq(
            "source_filename", filename
        ).execute()
        
        count = len(result.data) if result.data else 0
        logger.info(f"Deleted {count} documents for {filename}")
        
        return {"success": True, "count": count}
    
    except Exception as e:
        logger.error(f"Failed to delete documents for {filename}: {e}")
        raise


def get_document_stats() -> Dict[str, Any]:
    """Get statistics about indexed documents.
    
    Returns:
        Dict with document count and last index time
        
    Raises:
        Exception: If query fails
    """
    client = init_supabase_client()
    config = get_config()
    
    try:
        # Get total document count
        count_result = client.table(config.rag_vector_table).select(
            "id", count="exact"
        ).execute()
        doc_count = count_result.count if count_result.count is not None else 0
        
        # Get last indexed time from manifest
        manifest_result = client.table("file_manifest").select(
            "last_indexed"
        ).order("last_indexed", desc=True).limit(1).execute()
        
        last_indexed = None
        if manifest_result.data and len(manifest_result.data) > 0:
            last_indexed = manifest_result.data[0]["last_indexed"]
        
        return {
            "documents_count": doc_count,
            "last_index_time": last_indexed
        }
    
    except Exception as e:
        logger.error(f"Failed to get document stats: {e}")
        return {
            "documents_count": None,
            "last_index_time": None
        }


def search_similar_documents(
    query_embedding: List[float],
    limit: int = 12,
    similarity_threshold: float = 0.5
) -> List[Dict[str, Any]]:
    """Search for similar documents using vector similarity.
    
    Args:
        query_embedding: Query embedding vector
        limit: Maximum number of results to return
        similarity_threshold: Minimum similarity score (0-1)
    
    Returns:
        List of documents with similarity scores
        
    Raises:
        Exception: If search fails
    """
    client = init_supabase_client()
    config = get_config()
    
    try:
        # Use Supabase's vector similarity function
        # Note: This uses cosine distance where lower is more similar
        result = client.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": 1 - similarity_threshold,  # Convert to distance
                "match_count": limit
            }
        ).execute()
        
        documents = result.data if result.data else []
        logger.info(f"Found {len(documents)} similar documents")
        
        return documents
    
    except Exception as e:
        logger.error(f"Failed to search similar documents: {e}")
        raise


def clear_all_data() -> Dict[str, Any]:
    """Clear all data from documents and manifest tables.
    
    Used for full re-indexing. Deletes in correct order to respect foreign keys.
    
    Returns:
        Dict with success status and counts
        
    Raises:
        Exception: If delete operations fail
    """
    client = init_supabase_client()
    config = get_config()
    
    try:
        # Delete manifest first (has foreign key to documents)
        manifest_result = client.table("file_manifest").delete().neq(
            "filename", ""  # Delete all rows
        ).execute()
        manifest_count = len(manifest_result.data) if manifest_result.data else 0
        
        # Delete documents
        docs_result = client.table(config.rag_vector_table).delete().neq(
            "id", ""  # Delete all rows
        ).execute()
        docs_count = len(docs_result.data) if docs_result.data else 0
        
        logger.info(f"Cleared {docs_count} documents and {manifest_count} manifest entries")
        
        return {
            "success": True,
            "documents_deleted": docs_count,
            "manifest_deleted": manifest_count
        }
    
    except Exception as e:
        logger.error(f"Failed to clear data: {e}")
        raise
