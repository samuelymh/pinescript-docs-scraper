"""Pydantic models returned by the MCP tools.

The return annotation of a tool function becomes its MCP output schema, so
these models are the public contract. Field descriptions end up in the schema
and are read by the model, so they are written as instructions.

Note on score ranges: similarity is the raw cosine similarity returned by the
`match_documents` RPC (`1 - distance`). OpenAI embeddings are normalised, so the
value lies in [-1, 1] and can legitimately be negative. Do not add ge/le
constraints here -- `server.models.Source` does (ge=0.0, le=1.0) and would raise
on such values, which is why these tools do not reuse it.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class DocHit(BaseModel):
    """One documentation chunk matching a search query."""

    doc_id: str = Field(
        description="Stable chunk identifier: sha256(source_filename:chunk_index).",
    )
    source: str = Field(
        description="Source page filename inside pinescript_docs/processed.",
    )
    section: Optional[str] = Field(
        default=None,
        description="Heading of the section this chunk came from, when known.",
    )
    chunk_index: int = Field(description="0-based chunk number within the source page.")
    chunk_count: int = Field(description="Number of chunks the source page was split into.")
    score: float = Field(
        description="Cosine similarity in [-1, 1]. Higher is more relevant.",
    )
    has_code: bool = Field(description="Whether the chunk contains a Pine Script code block.")
    tokens: int = Field(description="Token count of the full chunk.")
    excerpt: str = Field(description="Beginning of the chunk, trimmed for context economy.")
    truncated: bool = Field(
        description="True when `excerpt` is shorter than the full chunk.",
    )


class ExampleHit(BaseModel):
    """A concrete Pine Script code example extracted from the documentation."""

    doc_id: str = Field(description="Chunk identifier the example was extracted from.")
    source: str = Field(description="Source page filename the example was found on.")
    section: Optional[str] = Field(
        default=None,
        description="Heading of the section the example belongs to.",
    )
    score: float = Field(description="Cosine similarity of the containing chunk.")
    language: str = Field(description="Language of the extracted block.")
    line_count: int = Field(description="Number of lines in the returned code block.")
    truncated: bool = Field(description="True when the code block was cut to fit the limit.")
    code: str = Field(description="The extracted Pine Script code.")


class Passage(BaseModel):
    """One retrieved passage included in an assembled context pack."""

    doc_id: str = Field(description="Chunk identifier, usable as provenance.")
    source: str = Field(description="Source page filename.")
    section: Optional[str] = Field(default=None, description="Section heading, when known.")
    score: float = Field(description="Cosine similarity in [-1, 1].")
    tokens: int = Field(description="Token count of `content`.")
    content: str = Field(description="Full passage text, ready to quote.")


class ContextPack(BaseModel):
    """An assembled, token-budgeted set of documentation passages."""

    query: str = Field(description="The query the passages were retrieved for.")
    passages: List[Passage] = Field(
        description="Passages ordered by descending relevance, within the budget.",
    )
    total_tokens: int = Field(description="Sum of the passage token counts.")
    budget: int = Field(description="Token budget that was applied.")
    documents_considered: int = Field(
        description="Number of chunks the retrieval step inspected before assembling.",
    )
