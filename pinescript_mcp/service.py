"""Read-only retrieval service backing the MCP tools.

This module is the only place that talks to the existing RAG stack. It reuses
`server/`:

    server.embed_client.generate_single_embedding   query -> vector
    server.supabase_client.search_documents_v2      vector + filters -> RPC rows
    server.retriever.assemble_context               rows -> token-budgeted selection
    server.utils.extract_code_blocks                chunk -> code blocks
    server.config.get_config                        settings

Token counting happens inside `assemble_context`, which uses
`server.utils.count_tokens`; this module never counts tokens itself.

Why the raw RPC rows and not `server.retriever.vector_search`?
`match_documents_v2` already returns `section_heading`, `chunk_count` and
`code_snippet`, but `vector_search` maps them into `server.models.RetrievedDocument`,
which only carries seven fields and silently drops those three. Reading the RPC
rows directly buys real provenance for the MCP output without modifying a single
existing file. If `get_chunk` is added later, extending `RetrievedDocument`
becomes worthwhile; until then this keeps the blast radius nil.

Filtering is done by the database.
The relevance floor, the source substring and the code-only flag are passed to
`match_documents_v2` (see `migrations/0002_match_documents_v2.sql`), so the RPC
returns only rows that already satisfy every criterion and this module discards
nothing. Before migration 0002 the RPC ignored its threshold and had no filter
parameters, which forced this module to over-fetch four times the requested
number of rows and filter them in Python; that over-fetch is gone.

The one remaining over-fetch is `EXAMPLE_SCAN_FACTOR`, and it exists for a
different reason: see its definition.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

from server.config import get_config
from server.embed_client import generate_single_embedding
from server.models import RetrievedDocument
from server.retriever import assemble_context
from server.supabase_client import MigrationRequiredError, search_documents_v2
from server.utils import extract_code_blocks

from pinescript_mcp.schemas import ContextPack, DocHit, ExampleHit, Passage

logger = logging.getLogger(__name__)

MAX_TOP_K = 50
"""Upper bound on rows requested from the RPC, so a bad argument cannot pull the whole table."""

EXAMPLE_SCAN_FACTOR = 2
"""Extra rows fetched per requested example in `find_examples`.

Not a filtering over-fetch. The `code_snippet` flag is set for any chunk containing
backticks, including chunks whose only code is inline (`plot(close)` in a sentence),
and those yield no fenced block for `extract_code_blocks` to return. Since choosing
the best block cannot be expressed as a SQL predicate, a small surplus is scanned so
a page of prose-heavy chunks does not silently reduce the example count.
"""

DEFAULT_TOPK = 8
DEFAULT_CONTEXT_TOPK = 12
DEFAULT_TOKEN_BUDGET = 8000

DEFAULT_MIN_SCORE = -1.0
"""Default score floor: the bottom of the cosine range, i.e. no filtering.

Cosine similarity over normalised OpenAI embeddings lies in [-1, 1], so negative
values are legitimate. Defaulting the floor to 0.0 would silently discard them,
which is why the floor only drops below this when a caller asks for it.
"""

MIN_TOKEN_BUDGET = 256
"""Smallest context budget `build_context` will honour; also the tool's declared minimum."""

EXCERPT_CHARS = 500
"""Excerpt length for `DocHit`. Full text is available through `get_context`."""

MAX_CODE_CHARS = 2000
"""Cap on a code example, to protect the client's context window."""

MIN_CODE_CHARS = 30
"""Shorter blocks are noise (inline snippets, fence artefacts), not examples."""


def _clamp_top_k(top_k: Any, default: int = DEFAULT_TOPK) -> int:
    """Coerce a caller-supplied top_k into 1..MAX_TOP_K."""
    try:
        value = int(top_k)
    except (TypeError, ValueError):
        value = default
    return max(1, min(value, MAX_TOP_K))


def _similarity(row: Mapping[str, Any]) -> float:
    """Read the similarity score from an RPC row, tolerating either field name."""
    for key in ("similarity", "similarity_score"):
        if key in row and row[key] is not None:
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def _text(row: Mapping[str, Any], key: str, default: str = "") -> str:
    value = row.get(key)
    return value if isinstance(value, str) else default


def _fetch_rows(
    query: str,
    limit: int,
    *,
    min_score: float = DEFAULT_MIN_SCORE,
    source: Optional[str] = None,
    code_only: bool = False,
) -> List[Dict[str, Any]]:
    """Embed `query` and fetch rows the database has already filtered.

    The similarity floor, the source substring and the code-only flag are all
    applied by `match_documents_v2`, so every returned row satisfies them and
    `limit` is exactly how many rows come back. Nothing is discarded here.

    Blocking: the OpenAI embeddings call and the Supabase RPC are both
    synchronous. The MCP SDK runs synchronous tool functions on a worker thread,
    so this does not block the server's event loop.
    """
    cleaned = (query or "").strip()
    if not cleaned:
        return []

    embedding = generate_single_embedding(cleaned)
    rows = search_documents_v2(
        embedding,
        limit=_clamp_top_k(limit),
        similarity_threshold=min_score,
        source_filename=source,
        code_only=code_only,
    )
    return [row for row in (rows or []) if isinstance(row, Mapping)]


def _excerpt(content: str, limit: int = EXCERPT_CHARS) -> Tuple[str, bool]:
    """Return a context-friendly excerpt and whether it was shortened."""
    text = (content or "").strip()
    if len(text) <= limit:
        return text, False

    cut = text[:limit]
    newline = cut.rfind("\n")
    if newline > limit // 2:
        cut = cut[:newline]
    return cut.rstrip() + " [...]", True


def _to_retrieved(row: Mapping[str, Any]) -> RetrievedDocument:
    """Adapt an RPC row to the shape `assemble_context` expects."""
    content = _text(row, "content")
    return RetrievedDocument(
        id=str(row.get("id") or ""),
        content=content,
        source_filename=_text(row, "source_filename"),
        chunk_index=int(row.get("chunk_index") or 0),
        token_count=int(row.get("token_count") or 0),
        similarity_score=_similarity(row),
        excerpt=content[:250],
    )


def search_docs(
    query: str,
    top_k: int = DEFAULT_TOPK,
    min_score: float = DEFAULT_MIN_SCORE,
    source: Optional[str] = None,
    include_code: bool = False,
) -> List[DocHit]:
    """Return ranked documentation chunks for `query`.

    The relevance floor, the source substring and the code-only flag are handed to
    the database, so no row is discarded here.
    """
    wanted = _clamp_top_k(top_k)
    floor = float(min_score) if min_score is not None else DEFAULT_MIN_SCORE
    rows = _fetch_rows(
        query,
        wanted,
        min_score=floor,
        source=source,
        code_only=include_code,
    )

    hits: List[DocHit] = []
    for row in rows:
        filename = _text(row, "source_filename")
        content = _text(row, "content")
        excerpt, truncated = _excerpt(content)

        hits.append(
            DocHit(
                doc_id=str(row.get("id") or ""),
                source=filename,
                section=_text(row, "section_heading") or None,
                chunk_index=int(row.get("chunk_index") or 0),
                chunk_count=int(row.get("chunk_count") or 1),
                score=_similarity(row),
                has_code=bool(row.get("code_snippet")),
                tokens=int(row.get("token_count") or 0),
                excerpt=excerpt,
                truncated=truncated,
            )
        )

    logger.info("search_docs query=%r hits=%d considered=%d", query, len(hits), len(rows))
    return hits


def build_context(
    query: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    top_k: int = DEFAULT_CONTEXT_TOPK,
) -> ContextPack:
    """Retrieve passages and fit them into a token budget."""
    config = get_config()
    max_budget = int(getattr(config, "prompt_token_budget", DEFAULT_TOKEN_BUDGET))
    try:
        budget = int(token_budget)
    except (TypeError, ValueError):
        budget = DEFAULT_TOKEN_BUDGET
    budget = max(MIN_TOKEN_BUDGET, min(budget, max_budget))

    wanted = _clamp_top_k(top_k, default=DEFAULT_CONTEXT_TOPK)
    rows = _fetch_rows(query, wanted)

    by_id = {str(row.get("id") or ""): row for row in rows}
    candidates = [_to_retrieved(row) for row in rows]

    selected, tokens_used = assemble_context(candidates, token_budget=budget)

    passages: List[Passage] = []
    for document in selected:
        row = by_id.get(document.id, {})
        passages.append(
            Passage(
                doc_id=document.id,
                source=document.source_filename,
                section=_text(row, "section_heading") or None,
                score=document.similarity_score,
                tokens=int(document.token_count or 0),
                content=document.content,
            )
        )

    logger.info(
        "build_context query=%r passages=%d tokens=%d budget=%d considered=%d",
        query,
        len(passages),
        tokens_used,
        budget,
        len(rows),
    )
    return ContextPack(
        query=query,
        passages=passages,
        total_tokens=int(tokens_used),
        budget=budget,
        documents_considered=len(rows),
    )


def _iter_code_blocks(row: Mapping[str, Any]) -> Iterator[str]:
    """Yield usable code blocks from a chunk, longest last."""
    blocks = extract_code_blocks(_text(row, "content"))
    usable = [block.strip() for block in blocks if len(block.strip()) >= MIN_CODE_CHARS]
    for block in sorted(usable, key=len):
        yield block


def find_examples(
    topic: str,
    limit: int = 5,
    min_score: float = DEFAULT_MIN_SCORE,
) -> List[ExampleHit]:
    """Return concrete Pine Script code examples for `topic`.

    `code_only=True` restricts the scan to chunks the database flagged as
    containing code, and the relevance floor is applied there too. The only work
    left here is unescaping the code blocks, which SQL cannot do.
    """
    wanted = max(1, min(int(limit or 5), 20))
    floor = float(min_score) if min_score is not None else DEFAULT_MIN_SCORE
    rows = _fetch_rows(
        topic,
        wanted * EXAMPLE_SCAN_FACTOR,
        min_score=floor,
        code_only=True,
    )

    examples: List[ExampleHit] = []
    for row in rows:
        blocks = list(_iter_code_blocks(row))
        if not blocks:
            continue

        code = blocks[-1]
        truncated = len(code) > MAX_CODE_CHARS
        if truncated:
            code = code[:MAX_CODE_CHARS].rstrip() + "\n// [...]"

        examples.append(
            ExampleHit(
                doc_id=str(row.get("id") or ""),
                source=_text(row, "source_filename"),
                section=_text(row, "section_heading") or None,
                score=_similarity(row),
                language="pine",
                line_count=code.count("\n") + 1,
                truncated=truncated,
                code=code,
            )
        )
        if len(examples) >= wanted:
            break

    logger.info("find_examples topic=%r examples=%d considered=%d", topic, len(examples), len(rows))
    return examples
