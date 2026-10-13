"""MCP tool definitions.

Each function here is a thin, synchronous wrapper around `pinescript_mcp.service`.
They are plain functions registered by `pinescript_mcp.app`; nothing in this
module imports the server instance, which keeps it importable from tests and
avoids a circular import.

Synchronous on purpose: the retrieval path is blocking (OpenAI embeddings plus a
Supabase RPC), and the MCP SDK runs synchronous tool functions on a worker
thread, so the event loop is never blocked.

Error strategy: a failure the model can act on is raised as `ToolError`, which
reaches the model with its message intact. Anything unexpected is logged with a
traceback and re-raised as a short `ToolError` that deliberately omits internal
detail.
"""

from __future__ import annotations

import logging
from typing import Annotated, List, Optional

from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field
from server.supabase_client import MigrationRequiredError

from pinescript_mcp import service
from pinescript_mcp.schemas import ContextPack, DocHit, ExampleHit

logger = logging.getLogger(__name__)

_CONFIG_HINT = (
    "The server has no usable configuration. Set SUPABASE_URL, "
    "SUPABASE_SERVICE_ROLE_KEY and OPENAI_API_KEY in the environment of the MCP "
    "client process (see docs/MCP.md)."
)


def _fail(action: str, exc: Exception) -> ToolError:
    """Log `exc` in full and build a concise, non-leaking message for the model."""
    logger.exception("%s failed", action)

    if isinstance(exc, MigrationRequiredError):
        return ToolError(
            "The documentation index does not have migration 0002 installed. Apply "
            "migrations/0002_match_documents_v2.sql to the Supabase project, then retry."
        )

    message = str(exc)
    if "supabase_url" in message or "openai_api_key" in message or "Field required" in message:
        return ToolError(_CONFIG_HINT)

    return ToolError(
        f"{action} failed ({type(exc).__name__}). The documentation index or "
        "embedding service may be unreachable; retrying may help."
    )


def search_docs(
    query: Annotated[
        str,
        Field(description="Natural-language or code search query, e.g. 'ta.sma moving average'."),
    ],
    top_k: Annotated[
        int,
        Field(description="Maximum number of chunks to return (1-50).", ge=1, le=50),
    ] = 8,
    min_score: Annotated[
        float,
        Field(
            description=(
                "Minimum cosine similarity, in [-1, 1]. The default of -1.0 applies "
                "no floor. Raise it (e.g. 0.3) to drop weak matches; values below 0 "
                "are legitimate and are only excluded if you ask for it. Applied by "
                "the database, so filtered-out chunks are never transferred."
            )
        ),
    ] = -1.0,
    source: Annotated[
        Optional[str],
        Field(
            description=(
                "Case-insensitive substring of the source page filename to restrict "
                "to, e.g. 'strategies'. Applied by the database."
            )
        ),
    ] = None,
    include_code: Annotated[
        bool,
        Field(
            description=(
                "When true, only return chunks flagged as containing code. Applied "
                "by the database."
            )
        ),
    ] = False,
) -> List[DocHit]:
    """Find Pine Script v6 documentation chunks relevant to a query.

    Use this first for "how do I" and "what does the docs say about" questions.
    Returns ranked chunks with provenance (source page, section heading, score)
    and a trimmed excerpt. The excerpt is often enough to answer; when it is not,
    call get_context for the same query to get full passages.
    """
    try:
        return service.search_docs(
            query,
            top_k=top_k,
            min_score=min_score,
            source=source,
            include_code=include_code,
        )
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - translated for the model
        raise _fail("Documentation search", exc) from exc


def get_context(
    query: Annotated[
        str,
        Field(description="The question or topic to assemble documentation context for."),
    ],
    token_budget: Annotated[
        int,
        Field(
            description="Maximum tokens of documentation to return (256-20000).",
            ge=256,
            le=20000,
        ),
    ] = 8000,
    top_k: Annotated[
        int,
        Field(description="Number of chunks to consider before trimming to the budget.", ge=1, le=50),
    ] = 12,
) -> ContextPack:
    """Assemble full documentation passages for a query, within a token budget.

    Use this when search_docs excerpts are too short and you need the complete
    text to reason over or quote. Passages come back in relevance order with
    provenance. Keep token_budget modest unless you genuinely need the material:
    every returned token is one the caller cannot use.
    """
    try:
        return service.build_context(query, token_budget=token_budget, top_k=top_k)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - translated for the model
        raise _fail("Context assembly", exc) from exc


def find_examples(
    topic: Annotated[
        str,
        Field(description="What the example should demonstrate, e.g. 'crossover entry orders'."),
    ],
    limit: Annotated[
        int,
        Field(description="Maximum number of examples to return (1-20).", ge=1, le=20),
    ] = 5,
    min_score: Annotated[
        float,
        Field(
            description=(
                "Minimum cosine similarity of the containing chunk, in [-1, 1]. "
                "Defaults to -1.0, meaning no floor. Applied by the database."
            )
        ),
    ] = -1.0,
) -> List[ExampleHit]:
    """Return concrete, copy-ready Pine Script code examples from the documentation.

    Prefer this over search_docs when the goal is working code rather than an
    explanation. Each hit is the longest usable code block from a matching chunk,
    with the page and section it came from. Examples are documentation samples,
    not production scripts: always adapt inputs and risk management.
    """
    try:
        return service.find_examples(topic, limit=limit, min_score=min_score)
    except ToolError:
        raise
    except Exception as exc:  # noqa: BLE001 - translated for the model
        raise _fail("Example extraction", exc) from exc
