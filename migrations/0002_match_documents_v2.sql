-- Migration: 0002_match_documents_v2
--
-- Adds `public.match_documents_v2`, a filtered replacement for `match_documents`.
--
-- Why a new function instead of changing `match_documents`:
--   * `match_documents` is the live contract for POST /chat via
--     server.supabase_client.search_similar_documents. Changing its parameters or
--     return shape would change that behaviour, which this change must not do.
--   * The v1 function is therefore left byte-for-byte untouched, including its
--     unused `match_threshold` argument.
--
-- What v2 fixes, relative to v1:
--   1. `match_threshold` is now applied. It means *similarity* (higher is better,
--      range [-1, 1]), not distance. v1 accepted the argument and ignored it, so
--      every query returned exactly `match_count` rows regardless of relevance.
--   2. Filtering happens here, not in the caller. v1 had no filter parameters, so
--      callers over-fetched (4x) and discarded rows in Python.
--   3. The `embedding` column is no longer returned. It is a 1536-dimension vector
--      that no caller reads; see scripts/benchmark_filtering.py for the measured
--      transfer cost. That column is the bulk of v1's response payload.
--
-- Deliberately NOT included here (tracked separately, see SECURITY-REVIEW.md):
--   * Row Level Security. SEC-01 still applies to `documents` and the new function
--     is protected only by the EXECUTE grants at the bottom of this file.
--   * A `tsvector` / GIN index for BM25. Hybrid search still needs the work
--     described in verificatie_mcp.md before it can be enabled.
--
-- Rollback:
--   DROP FUNCTION IF EXISTS public.match_documents_v2(vector, double precision, integer, text, boolean);
--   (Dropping this function does not affect match_documents or any indexed data.)

-- -----------------------------------------------------------------------------
-- RPC: match_documents_v2
-- -----------------------------------------------------------------------------

CREATE OR REPLACE FUNCTION public.match_documents_v2(
    query_embedding vector(1536),
    match_threshold DOUBLE PRECISION DEFAULT -1.0,
    match_count INTEGER DEFAULT 12,
    filter_source TEXT DEFAULT NULL,
    filter_code_only BOOLEAN DEFAULT false
)
RETURNS TABLE (
    id TEXT,
    content TEXT,
    source_filename TEXT,
    chunk_index INTEGER,
    chunk_count INTEGER,
    section_heading TEXT,
    token_count INTEGER,
    code_snippet BOOLEAN,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE,
    similarity DOUBLE PRECISION
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        d.id,
        d.content,
        d.source_filename,
        d.chunk_index,
        d.chunk_count,
        d.section_heading,
        d.token_count,
        d.code_snippet,
        d.metadata,
        d.created_at,
        d.updated_at,
        1 - (d.embedding <=> query_embedding) AS similarity
    FROM documents d
    WHERE d.embedding IS NOT NULL
      -- Relevance floor. Similarity, not distance: higher is better.
      -- The default of -1.0 is the bottom of the cosine range and means "no floor",
      -- so a caller that passes nothing gets v1-like behaviour.
      AND 1 - (d.embedding <=> query_embedding) >= match_threshold
      -- Literal, case-insensitive substring match. `strpos` is used instead of
      -- ILIKE so that a source filter containing '%' or '_' cannot turn into a
      -- wildcard expression; this matches the previous Python semantics exactly
      -- (`needle in filename.lower()`) rather than approximating them.
      AND (
            filter_source IS NULL
            OR strpos(lower(d.source_filename), lower(filter_source)) > 0
          )
      AND (filter_code_only IS FALSE OR d.code_snippet IS TRUE)
    ORDER BY d.embedding <=> query_embedding
    LIMIT match_count;
$$;

COMMENT ON FUNCTION public.match_documents_v2(
    vector, double precision, integer, text, boolean
) IS
    'Filtered vector search over documents. match_threshold is a SIMILARITY floor '
    '(>= is better), not a distance. Returns no embedding column.';

-- -----------------------------------------------------------------------------
-- Execution rights
-- -----------------------------------------------------------------------------
--
-- PostgreSQL grants EXECUTE on new functions to PUBLIC by default, which in
-- Supabase means the `anon` and `authenticated` roles -- the roles a browser
-- client uses with the public anon key. Creating the function without the
-- statements below would introduce a second publicly callable vector-search
-- endpoint, exactly the exposure SEC-01 already reports for `match_documents`.
--
-- The application authenticates with SUPABASE_SERVICE_ROLE_KEY
-- (see server/supabase_client.py), i.e. it runs as `service_role`, so granting
-- to that role alone is sufficient and no less functional.
--
-- If your deployment calls this RPC with a different role, add that role here.
-- To revert to PostgreSQL's default behaviour instead, drop these two statements.

REVOKE EXECUTE ON FUNCTION public.match_documents_v2(
    vector, double precision, integer, text, boolean
) FROM PUBLIC;

GRANT EXECUTE ON FUNCTION public.match_documents_v2(
    vector, double precision, integer, text, boolean
) TO service_role;

-- -----------------------------------------------------------------------------
-- Notes
-- -----------------------------------------------------------------------------
--
-- 1. Filters are applied after the HNSW index traversal, because pgvector ordered
--    scans cannot push a predicate into the graph walk. A highly selective
--    `filter_source` can therefore make the planner examine many more rows than
--    `match_count` before it has enough matches, and in the worst case return
--    fewer rows than requested. On the current corpus (73 pages) this is not
--    measurable. If the index grows past ~100k chunks and filtered queries become
--    slow, enable pgvector's iterative scan:
--
--        SET LOCAL hnsw.iterative_scan = 'relaxed_order';
--
--    That requires pgvector >= 0.8 and is deliberately not set inside the function
--    body, because `SET` on an unknown parameter would break older deployments.
--
-- 2. `idx_documents_source_filename` is a B-tree and cannot serve `strpos()`. The
--    vector index does the heavy lifting and the source filter is a cheap
--    post-filter on a small candidate set. Switch to a prefix match if the B-tree
--    needs to be usable.
--
-- 3. `code_snippet` is true for 531 of 594 chunks (89%), so `filter_code_only`
--    only removes a small tail. It is not a substitute for extracting actual code
--    blocks, which still has to happen in the caller.
--
-- 4. When an embedding model with a different dimension is introduced, this
--    function's signature must change with it: `vector(1536)` is fixed here just
--    as it is in `documents.embedding`.
