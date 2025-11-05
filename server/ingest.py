"""Document ingestion pipeline for PineScript RAG Server.

Scans, parses, chunks, embeds, and indexes processed markdown files.
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import logging

from server.config import get_config
from server.models import Document, FileManifest
from server.utils import (
    count_tokens,
    hash_string,
    detect_code_snippets,
    generate_doc_id
)
from server.embed_client import generate_embeddings_chunked, estimate_embedding_cost
from server.supabase_client import (
    fetch_manifest,
    update_manifest,
    upsert_documents,
    delete_documents_by_filename,
    clear_all_data
)

logger = logging.getLogger(__name__)


def scan_documents(docs_dir: Optional[str] = None) -> List[Path]:
    """Scan processed documents directory for markdown files.
    
    Args:
        docs_dir: Path to processed documents directory (defaults to pinescript_docs/processed)
    
    Returns:
        List of Path objects for markdown files
    """
    if docs_dir is None:
        # Default to pinescript_docs/processed relative to project root
        project_root = Path(__file__).parent.parent
        docs_dir = project_root / "pinescript_docs" / "processed"
    else:
        docs_dir = Path(docs_dir)
    
    if not docs_dir.exists():
        logger.error(f"Documents directory not found: {docs_dir}")
        return []
    
    # Find all .md files
    md_files = list(docs_dir.glob("*.md"))
    logger.info(f"Found {len(md_files)} markdown files in {docs_dir}")
    
    return sorted(md_files)


def chunk_document(
    content: str,
    filename: str,
    chunk_token_threshold: int,
    overlap_tokens: int
) -> List[Tuple[str, Optional[str], int]]:
    """Chunk document by H1/H2 headings if it exceeds token threshold.
    
    Splits document into sections based on markdown headings. Adds overlap
    by including a portion of the previous chunk's ending content.
    
    Args:
        content: Full document content
        filename: Source filename (for logging)
        chunk_token_threshold: Token count threshold for chunking
        overlap_tokens: Number of tokens to overlap between chunks
    
    Returns:
        List of tuples: (chunk_content, section_heading, chunk_index)
    """
    total_tokens = count_tokens(content)
    
    # If under threshold, return as single chunk
    if total_tokens <= chunk_token_threshold:
        logger.debug(f"{filename}: {total_tokens} tokens, no chunking needed")
        return [(content, None, 0)]
    
    logger.info(f"{filename}: {total_tokens} tokens, chunking by headings")
    
    # Split by H1 and H2 headings (## or #)
    # Pattern matches: start of line, one or two #, space, heading text
    heading_pattern = r'^(#{1,2})\s+(.+)$'
    
    chunks = []
    current_chunk = []
    current_heading = None
    previous_chunk_content = ""
    
    lines = content.split('\n')
    
    for line in lines:
        heading_match = re.match(heading_pattern, line)
        
        if heading_match:
            # Save previous chunk if it exists
            if current_chunk:
                chunk_content = '\n'.join(current_chunk)
                chunks.append((chunk_content, current_heading, len(chunks)))
                
                # Calculate overlap from previous chunk
                chunk_tokens = count_tokens(chunk_content)
                if chunk_tokens > overlap_tokens:
                    # Extract last overlap_tokens worth of content
                    overlap_content = extract_token_overlap(
                        chunk_content,
                        overlap_tokens
                    )
                    previous_chunk_content = overlap_content
                else:
                    previous_chunk_content = chunk_content
            
            # Start new chunk with heading
            current_heading = heading_match.group(2).strip()
            current_chunk = []
            
            # Add overlap from previous chunk if available
            if previous_chunk_content:
                current_chunk.append("... " + previous_chunk_content)
            
            current_chunk.append(line)
        else:
            current_chunk.append(line)
    
    # Add final chunk
    if current_chunk:
        chunk_content = '\n'.join(current_chunk)
        chunks.append((chunk_content, current_heading, len(chunks)))
    
    # If chunking resulted in only one chunk, return original
    if len(chunks) <= 1:
        logger.debug(f"{filename}: Chunking produced only 1 chunk, using original")
        return [(content, None, 0)]
    
    logger.info(f"{filename}: Split into {len(chunks)} chunks")
    return chunks


def extract_token_overlap(text: str, target_tokens: int) -> str:
    """Extract approximately target_tokens worth of text from the end.
    
    Args:
        text: Text to extract from
        target_tokens: Approximate number of tokens to extract
    
    Returns:
        Extracted text from end
    """
    # Estimate: ~4 characters per token on average
    char_estimate = target_tokens * 4
    
    if len(text) <= char_estimate:
        return text
    
    # Extract from end, try to break at sentence boundary
    excerpt = text[-char_estimate:]
    
    # Try to find a sentence boundary
    sentence_breaks = ['. ', '.\n', '? ', '!\n']
    best_pos = 0
    
    for sep in sentence_breaks:
        pos = excerpt.find(sep)
        if pos > best_pos:
            best_pos = pos
    
    if best_pos > 0:
        excerpt = excerpt[best_pos + 2:].strip()
    
    return excerpt


def parse_document(filepath: Path) -> List[Document]:
    """Parse a markdown document into Document objects.
    
    Reads file, computes metadata, optionally chunks by headings,
    and creates Document objects ready for embedding.
    
    Args:
        filepath: Path to markdown file
    
    Returns:
        List of Document objects (one if not chunked, multiple if chunked)
    """
    config = get_config()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        logger.error(f"Failed to read {filepath}: {e}")
        return []
    
    if not content.strip():
        logger.warning(f"Empty file: {filepath}")
        return []
    
    filename = filepath.name
    
    # Chunk if needed
    chunks = chunk_document(
        content,
        filename,
        config.chunk_token_threshold,
        config.chunk_overlap_tokens
    )
    
    documents = []
    chunk_count = len(chunks)
    
    for chunk_content, section_heading, chunk_index in chunks:
        # Calculate metadata (count tokens for embedding model)
        token_count = count_tokens(chunk_content, model=config.embedding_model)
        has_code = detect_code_snippets(chunk_content)
        doc_id = generate_doc_id(filename, chunk_index)
        
        # Create Document object (without embedding yet)
        doc = Document(
            id=doc_id,
            content=chunk_content,
            source_filename=filename,
            chunk_index=chunk_index,
            chunk_count=chunk_count,
            section_heading=section_heading,
            token_count=token_count,
            code_snippet=has_code,
            metadata={
                "file_path": str(filepath),
                "processed_timestamp": datetime.now().isoformat()
            },
            embedding=None  # Will be populated later
        )
        
        documents.append(doc)
        
        logger.debug(
            f"Parsed {filename} chunk {chunk_index + 1}/{chunk_count}: "
            f"{token_count} tokens, code={has_code}, heading={section_heading}"
        )
    
    return documents


def split_text_by_token_limit(text: str, max_tokens: int, overlap_tokens: int, model: str) -> List[str]:
    """Split text into parts each under max_tokens (approximate using token counts).

    Uses a conservative character-per-token estimate for slicing and then adjusts
    boundaries to prefer sentence breaks via extract_token_overlap.
    """
    parts = []
    # Fast path
    if count_tokens(text, model=model) <= max_tokens:
        return [text]

    # Estimate characters per token (conservative)
    # Use a smaller chars-per-token to produce smaller chunks (safer for code/docs)
    chars_per_token = 3
    max_chars = max_tokens * chars_per_token
    start = 0
    length = len(text)

    while start < length:
        end = min(length, start + max_chars)
        chunk = text[start:end]

        # If we're not at end, try to back up to last sentence boundary for nicer splits
        if end < length:
            # Look for a sentence break near the end of chunk
            sentence_breaks = ['. ', '.\n', '? ', '!\n']
            best_pos = -1
            for sep in sentence_breaks:
                pos = chunk.rfind(sep)
                if pos > best_pos:
                    best_pos = pos
            if best_pos > 0:
                # keep up to sentence end
                chunk = chunk[: best_pos + 1]
                end = start + len(chunk)

        # Ensure chunk truly fits the token limit; if not, shrink progressively.
        actual_tokens = count_tokens(chunk, model=model)
        if actual_tokens > max_tokens:
            # progressively shrink chunk until it fits
            attempt = 0
            while actual_tokens > max_tokens and attempt < 10:
                # shrink to 80% of current size (conservative)
                new_len = max(200, int(len(chunk) * 0.8))
                chunk = chunk[:new_len]
                actual_tokens = count_tokens(chunk, model=model)
                attempt += 1
            if actual_tokens > max_tokens:
                # As a last resort, force cut to max_chars/2
                chunk = chunk[: max_chars // 2]
                actual_tokens = count_tokens(chunk, model=model)

        parts.append(chunk.strip())

        # Prepare next start with overlap
        if end >= length:
            break

        # compute overlap in chars
        overlap_chars = max(50, overlap_tokens * chars_per_token)
        start = max(0, end - overlap_chars)

    return parts


def check_manifest(
    files: List[Path],
    existing_manifest: Dict[str, FileManifest]
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Compare files against manifest to detect changes.
    
    Args:
        files: List of file paths to check
        existing_manifest: Dict mapping filename to FileManifest
    
    Returns:
        Tuple of (new_files, modified_files, unchanged_files)
    """
    new_files = []
    modified_files = []
    unchanged_files = []
    
    for filepath in files:
        filename = filepath.name
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            content_hash = hash_string(content)
        except Exception as e:
            logger.error(f"Failed to read {filepath} for hash: {e}")
            continue
        
        if filename not in existing_manifest:
            new_files.append(filepath)
            logger.debug(f"New file: {filename}")
        elif existing_manifest[filename].content_hash != content_hash:
            modified_files.append(filepath)
            logger.debug(f"Modified file: {filename}")
        else:
            unchanged_files.append(filepath)
            logger.debug(f"Unchanged file: {filename}")
    
    logger.info(
        f"Manifest check: {len(new_files)} new, "
        f"{len(modified_files)} modified, {len(unchanged_files)} unchanged"
    )
    
    return new_files, modified_files, unchanged_files


async def index_documents(full_reindex: bool = False) -> Dict[str, any]:
    """Main indexing pipeline orchestrator.
    
    Scans documents, checks for changes, parses, generates embeddings,
    and upserts to Supabase with manifest updates.
    
    Args:
        full_reindex: If True, clear all data and reindex everything
    
    Returns:
        Dict with indexing results and statistics
    """
    config = get_config()
    start_time = datetime.now()
    
    logger.info(f"Starting document indexing (full_reindex={full_reindex})")
    
    # Step 1: Scan documents
    files = scan_documents()
    if not files:
        return {
            "success": False,
            "error": "No documents found to index",
            "files_scanned": 0
        }
    
    # Step 2: Handle full reindex
    if full_reindex:
        logger.info("Full reindex requested, clearing existing data")
        try:
            clear_result = clear_all_data()
            logger.info(f"Cleared {clear_result['documents_deleted']} documents")
        except Exception as e:
            logger.error(f"Failed to clear data: {e}")
            return {
                "success": False,
                "error": f"Failed to clear existing data: {str(e)}"
            }
        
        files_to_process = files
    else:
        # Step 3: Check manifest for incremental update
        try:
            existing_manifest = fetch_manifest()
        except Exception as e:
            logger.error(f"Failed to fetch manifest: {e}")
            return {
                "success": False,
                "error": f"Failed to fetch manifest: {str(e)}"
            }
        
        new_files, modified_files, unchanged_files = check_manifest(
            files, existing_manifest
        )
        
        # Delete old chunks for modified files
        for filepath in modified_files:
            try:
                delete_documents_by_filename(filepath.name)
            except Exception as e:
                logger.error(f"Failed to delete old chunks for {filepath.name}: {e}")
        
        files_to_process = new_files + modified_files
        
        if not files_to_process:
            logger.info("No new or modified files to process")
            return {
                "success": True,
                "files_scanned": len(files),
                "files_processed": 0,
                "documents_indexed": 0,
                "unchanged_files": len(unchanged_files)
            }
    
    logger.info(f"Processing {len(files_to_process)} files")
    
    # Step 4: Parse documents
    all_documents = []
    for filepath in files_to_process:
        try:
            docs = parse_document(filepath)
            all_documents.extend(docs)
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
    
    if not all_documents:
        return {
            "success": False,
            "error": "No documents parsed successfully",
            "files_processed": len(files_to_process)
        }
    
    logger.info(f"Parsed {len(all_documents)} document chunks")
    
    # Step 4.5: Ensure no document exceeds embedding model context length by
    # splitting overly-large chunks. Rebuild documents per-file so chunk_count
    # and chunk_index remain consistent for each source file.
    embedding_model = config.embedding_model
    # conservative per-model max token limits (fallback to 8192)
    MODEL_MAX_TOKENS = {
        "text-embedding-3-small": 8192,
        "text-embedding-3-large": 8192,
        "text-embedding-ada-002": 8192
    }
    max_tokens_allowed = MODEL_MAX_TOKENS.get(embedding_model, 8192)

    expanded_documents: List[Document] = []
    # Group by filename
    files_map: Dict[str, List[Document]] = {}
    for doc in all_documents:
        files_map.setdefault(doc.source_filename, []).append(doc)

    for filename, docs in files_map.items():
        new_parts = []  # tuples (content, section_heading, code_snippet, metadata)

        for doc in docs:
            # If doc is small enough (by embedding model), keep as-is
            if doc.token_count <= max_tokens_allowed:
                new_parts.append((doc.content, doc.section_heading, doc.code_snippet, doc.metadata))
                continue

            # Otherwise split into smaller pieces
            subtexts = split_text_by_token_limit(doc.content, max_tokens_allowed, config.chunk_overlap_tokens, model=embedding_model)
            for i, sub in enumerate(subtexts):
                # keep the section heading only for the first subpart of this doc
                heading = doc.section_heading if i == 0 else None
                has_code = detect_code_snippets(sub)
                new_parts.append((sub, heading, has_code, doc.metadata))

        # Create Document objects with new chunk_count and chunk_index
        total = len(new_parts)
        for idx, (content, heading, has_code, metadata) in enumerate(new_parts):
            token_count = count_tokens(content, model=embedding_model)
            doc_id = generate_doc_id(filename, idx)
            new_doc = Document(
                id=doc_id,
                content=content,
                source_filename=filename,
                chunk_index=idx,
                chunk_count=total,
                section_heading=heading,
                token_count=token_count,
                code_snippet=has_code,
                metadata=metadata,
                embedding=None
            )
            expanded_documents.append(new_doc)

    all_documents = expanded_documents

    # Step 5: Estimate embedding cost
    avg_tokens = sum(doc.token_count for doc in all_documents) / len(all_documents)
    cost_estimate = estimate_embedding_cost(
        len(all_documents),
        int(avg_tokens)
    )
    logger.info(f"Embedding cost estimate: ${cost_estimate['estimated_cost_usd']:.4f}")
    
    # Step 6: Generate embeddings
    try:
        texts = [doc.content for doc in all_documents]
        embeddings = generate_embeddings_chunked(texts, batch_size=100)
        
        # Attach embeddings to documents
        for doc, embedding in zip(all_documents, embeddings):
            doc.embedding = embedding
        
        logger.info(f"Generated {len(embeddings)} embeddings")
    
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        return {
            "success": False,
            "error": f"Failed to generate embeddings: {str(e)}",
            "documents_parsed": len(all_documents)
        }
    
    # Step 7: Upsert documents to Supabase
    try:
        upsert_result = upsert_documents(all_documents)
        logger.info(f"Upserted {upsert_result['count']} documents")
    except Exception as e:
        logger.error(f"Failed to upsert documents: {e}")
        return {
            "success": False,
            "error": f"Failed to upsert documents: {str(e)}",
            "documents_with_embeddings": len(all_documents)
        }
    
    # Step 8: Update manifest
    manifest_entries = []
    file_doc_map = {}  # Map filename to first doc for manifest
    
    for doc in all_documents:
        if doc.source_filename not in file_doc_map:
            file_doc_map[doc.source_filename] = doc
    
    for filepath in files_to_process:
        filename = filepath.name
        
        if filename not in file_doc_map:
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            content_hash = hash_string(content)
            
            manifest_entry = FileManifest(
                filename=filename,
                content_hash=content_hash,
                last_indexed=datetime.now(),
                doc_id=file_doc_map[filename].id
            )
            manifest_entries.append(manifest_entry)
        
        except Exception as e:
            logger.error(f"Failed to create manifest entry for {filename}: {e}")
    
    try:
        manifest_result = update_manifest(manifest_entries)
        logger.info(f"Updated {manifest_result['count']} manifest entries")
    except Exception as e:
        logger.error(f"Failed to update manifest: {e}")
        # Non-fatal - documents are still indexed
    
    # Calculate results
    elapsed = (datetime.now() - start_time).total_seconds()
    
    results = {
        "success": True,
        "files_scanned": len(files),
        "files_processed": len(files_to_process),
        "documents_indexed": len(all_documents),
        "embeddings_generated": len(embeddings),
        "elapsed_seconds": round(elapsed, 2),
        "cost_estimate_usd": cost_estimate["estimated_cost_usd"]
    }
    
    if not full_reindex:
        results["unchanged_files"] = len(files) - len(files_to_process)
    
    logger.info(f"Indexing complete: {results}")
    
    return results
