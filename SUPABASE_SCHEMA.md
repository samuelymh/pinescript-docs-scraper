# Supabase Schema Documentation

This document describes the database schema for the PineScript RAG Server.

## Prerequisites

Enable the `vector` extension in your Supabase project:

```sql
-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;
```

## Tables

### `documents` Table

Stores document chunks with embeddings for vector similarity search.

```sql
-- Create documents table
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source_filename TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    section_heading TEXT,
    token_count INTEGER NOT NULL,
    code_snippet BOOLEAN NOT NULL DEFAULT false,
    metadata JSONB DEFAULT '{}',
    embedding vector(1536),  -- text-embedding-3-small dimension
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create index on source_filename for efficient lookups
CREATE INDEX IF NOT EXISTS idx_documents_source_filename 
ON documents(source_filename);

-- Create index on code_snippet for filtering
CREATE INDEX IF NOT EXISTS idx_documents_code_snippet 
ON documents(code_snippet);

-- Create vector index for similarity search (using HNSW)
-- Cosine distance is used for similarity
CREATE INDEX IF NOT EXISTS idx_documents_embedding 
ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- Create trigger for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

### `file_manifest` Table

Tracks indexed files with content hashes for change detection and incremental indexing.

```sql
-- Create file_manifest table
CREATE TABLE IF NOT EXISTS file_manifest (
    filename TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    last_indexed TIMESTAMP WITH TIME ZONE NOT NULL,
    doc_id TEXT NOT NULL,
    chunk_count INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
);

-- Create index on content_hash for efficient comparison
CREATE INDEX IF NOT EXISTS idx_file_manifest_content_hash 
ON file_manifest(content_hash);

-- Create index on last_indexed for sorting
CREATE INDEX IF NOT EXISTS idx_file_manifest_last_indexed 
ON file_manifest(last_indexed DESC);
```

## Foreign Key & Testing Notes

- **Foreign key constraint:** The `file_manifest.doc_id` column is a foreign key referencing `documents(id)` with `ON DELETE CASCADE`. This means any insert or upsert into `file_manifest` must reference an existing `documents.id` value, or the database will raise a foreign-key violation (this is the cause of the failure seen in live tests).

- **Testing guidance:** Integration or live tests that call `update_manifest()` (or otherwise insert into `file_manifest`) should ensure a corresponding `documents` row exists first. In tests it's acceptable to upsert a minimal placeholder `Document` with the same `doc_id` used in the manifest entry, then clean it up after the test. Example (Python):

```py
# Example: create a minimal document before upserting a manifest entry
from server.models import Document
from server.supabase_client import upsert_documents
from server.embed_client import EMBEDDING_DIM

doc = Document(
    id="doc_live_test",
    content="placeholder",
    source_filename="manifest_test.md",
    chunk_index=0,
    chunk_count=1,
    section_heading=None,
    token_count=1,
    code_snippet=False,
    metadata={"test": True},
    embedding=[0.0] * EMBEDDING_DIM,
)

upsert_documents([doc])
```

- **Cleanup:** After the manifest test completes you can delete the placeholder document (or rely on `ON DELETE CASCADE` behavior if you remove the `file_manifest` row first).

## Schema Details

### Field Descriptions

#### `documents` Table

- **id**: Deterministic document ID generated as `sha256(source_filename + ":" + chunk_index)`
- **content**: Full text content of the document chunk
- **source_filename**: Original filename from `pinescript_docs/processed/`
- **chunk_index**: 0-based chunk number (0 if not chunked)
- **chunk_count**: Total number of chunks from this file (1 if not chunked)
- **section_heading**: H1/H2 heading text if document was chunked by section
- **token_count**: Estimated token count using tiktoken (cl100k_base encoding)
- **code_snippet**: Boolean flag indicating presence of code (triple backticks, single backticks, or "Pine Script®\nCopied\n" pattern)
- **metadata**: Flexible JSONB field for additional metadata (e.g., processed_timestamp)
- **embedding**: Vector embedding from OpenAI `text-embedding-3-small` (1536 dimensions)
- **created_at**: Timestamp when document was first indexed
- **updated_at**: Timestamp when document was last updated

#### `file_manifest` Table

- **filename**: Source filename (unique identifier)
- **content_hash**: SHA256 hash of file content for change detection
- **last_indexed**: Timestamp of last successful indexing
- **doc_id**: Reference to first chunk's document ID (for tracking)
- **chunk_count**: Number of chunks created from this file

## Indexing Strategy

### Full Reindex
1. Drop all existing records from both tables
2. Scan all files in `pinescript_docs/processed/`
3. Parse, chunk, embed, and insert all documents

### Incremental Reindex (Default)
1. Fetch existing manifest from `file_manifest`
2. Scan files and compute content hashes
3. Compare hashes to detect changes:
   - **New files**: Parse, embed, and insert
   - **Modified files**: Delete old chunks, parse, embed, and insert new chunks
   - **Unchanged files**: Skip
4. Update manifest with new hashes and timestamps

## Vector Search

### Similarity Query Example

```sql
-- Find top 12 most similar documents to a query embedding
SELECT 
    id,
    source_filename,
    section_heading,
    content,
    token_count,
    code_snippet,
    1 - (embedding <=> $1::vector) AS similarity_score
FROM documents
WHERE embedding IS NOT NULL
ORDER BY embedding <=> $1::vector
LIMIT $2;
```

Where `$1` is the query embedding vector and `$2` is the limit (typically 8-12).

## Maintenance

### Recompute Embeddings

If switching embedding models, drop and recreate the `documents` table, then run a full reindex.

### Vacuum and Analyze

For optimal performance after bulk inserts/updates:

```sql
VACUUM ANALYZE documents;
VACUUM ANALYZE file_manifest;
```

## Row-Level Security (RLS)

For production deployments, enable RLS and create policies:

```sql
-- Enable RLS
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE file_manifest ENABLE ROW LEVEL SECURITY;

-- Allow service role full access
CREATE POLICY service_role_all ON documents
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

CREATE POLICY service_role_all ON file_manifest
    FOR ALL
    TO service_role
    USING (true)
    WITH CHECK (true);

-- Allow authenticated users read-only access
CREATE POLICY authenticated_read ON documents
    FOR SELECT
    TO authenticated
    USING (true);
```
