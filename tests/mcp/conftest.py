"""Fixtures for the MCP tool tests.

The tests exercise the real `pinescript_mcp` code but stub the two outbound
calls (`generate_single_embedding` and `search_documents_v2`) so that no network
access and no Supabase project are required.

`FakeIndex.search_v2` deliberately reimplements the filtering that
`migrations/0002_match_documents_v2.sql` performs - threshold, literal
case-insensitive source substring, code-only flag, then LIMIT - so the behaviour
tests stay meaningful now that the filtering lives in the database.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

# Configuration is read lazily by server.config.get_config() on the first tool
# call, so the credentials only need to look plausible, never to work.
os.environ.setdefault("SUPABASE_URL", "https://example.supabase.co")
os.environ.setdefault("SUPABASE_SERVICE_ROLE_KEY", "test-service-role-key")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")
os.environ.setdefault("JWT_SECRET", "test-secret")
os.environ.setdefault("ADMIN_API_KEY", "test-admin-key")


class FakeIndex:
    """Emulates the `match_documents_v2` RPC and records how it was called."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self.rows = rows
        self.embed_calls: List[str] = []
        self.search_calls: List[Dict[str, Any]] = []

    def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        self.embed_calls.append(text)
        return [0.0] * 1536

    def search_v2(
        self,
        query_embedding: List[float],
        limit: int = 12,
        similarity_threshold: float = -1.0,
        source_filename: Optional[str] = None,
        code_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """Apply the SQL predicates in Python, mirroring the migration."""
        self.search_calls.append(
            {
                "limit": limit,
                "similarity_threshold": similarity_threshold,
                "source_filename": source_filename,
                "code_only": code_only,
            }
        )

        needle = (source_filename or "").strip().lower()
        matched: List[Dict[str, Any]] = []
        for row in self.rows:
            if float(row.get("similarity", 0.0)) < similarity_threshold:
                continue
            if needle and needle not in str(row.get("source_filename", "")).lower():
                continue
            if code_only and not row.get("code_snippet"):
                continue
            matched.append(row)

        return matched[: max(0, int(limit))]


@pytest.fixture(autouse=True)
def block_outbound_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail loudly when a test forgets to stub the two outbound calls.

    Without this guard a missing stub reaches the real OpenAI or Supabase endpoint
    using the placeholder credentials above. That makes the suite slow and
    network-dependent, and it leaks a request. Tests that need data override these
    with the `fake_index` or `failing_index` fixture, which run afterwards.
    """

    def _blocked(*args: Any, **kwargs: Any):
        raise AssertionError(
            "a test attempted a real outbound call: stub generate_single_embedding / "
            "search_documents_v2 with the fake_index or failing_index fixture"
        )

    monkeypatch.setattr("pinescript_mcp.service.generate_single_embedding", _blocked)
    monkeypatch.setattr("pinescript_mcp.service.search_documents_v2", _blocked)


@pytest.fixture
def fake_index(monkeypatch: pytest.MonkeyPatch):
    """Install a fake index and return a helper to configure it."""

    def _install(rows: List[Dict[str, Any]]) -> FakeIndex:
        index = FakeIndex(rows)
        monkeypatch.setattr("pinescript_mcp.service.generate_single_embedding", index.embed)
        monkeypatch.setattr("pinescript_mcp.service.search_documents_v2", index.search_v2)
        return index

    return _install


@pytest.fixture
def failing_index(monkeypatch: pytest.MonkeyPatch):
    """Make the retrieval step raise, to exercise the ToolError translation."""

    def _boom(*args: Any, **kwargs: Any):
        raise ConnectionError("simulated upstream outage")

    monkeypatch.setattr("pinescript_mcp.service.generate_single_embedding", _boom)
    monkeypatch.setattr("pinescript_mcp.service.search_documents_v2", _boom)
    return _boom
