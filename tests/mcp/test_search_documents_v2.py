"""Unit tests for `server.supabase_client.search_documents_v2`.

These cover the client half of migration 0002: parameter semantics, the retry
policy, and the distinction between "the database is unreachable" and "migration
0002 was never applied" -- the two failures need very different messages.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from server import supabase_client
from server.supabase_client import MigrationRequiredError

EMBEDDING = [0.0] * 1536


def install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows: Optional[List[Dict[str, Any]]] = None,
    error: Optional[Exception] = None,
    error_every_time: bool = True,
) -> Dict[str, Any]:
    """Point the module at a fake RPC endpoint and record what it is sent."""
    state: Dict[str, Any] = {"calls": [], "payloads": []}

    class _FakeResult:
        def __init__(self, data: List[Dict[str, Any]]) -> None:
            self.data = data

    class _FakeRpc:
        def execute(self):
            state["calls"].append(1)
            if error is not None and (error_every_time or len(state["calls"]) == 1):
                raise error
            return _FakeResult(rows or [])

    class _FakeClient:
        def rpc(self, name: str, payload: Dict[str, Any]):
            state["name"] = name
            state["payloads"].append(payload)
            return _FakeRpc()

    monkeypatch.setattr(supabase_client, "init_supabase_client", lambda: _FakeClient())
    monkeypatch.setattr(supabase_client.time, "sleep", lambda *_: None)
    return state


class TestSuccess:
    def test_returns_the_rows_the_database_sent(self, monkeypatch):
        state = install_fake_client(monkeypatch, rows=[{"id": "a"}, {"id": "b"}])

        rows = supabase_client.search_documents_v2(EMBEDDING, limit=5)

        assert rows == [{"id": "a"}, {"id": "b"}]
        assert state["name"] == "match_documents_v2"

    def test_empty_result_is_an_empty_list(self, monkeypatch):
        install_fake_client(monkeypatch, rows=None)

        assert supabase_client.search_documents_v2(EMBEDDING, limit=5) == []


class TestParameterSemantics:
    def test_threshold_is_sent_as_a_similarity_not_a_distance(self, monkeypatch):
        """v1's argument of the same name meant distance and was ignored anyway."""
        state = install_fake_client(monkeypatch, rows=[])

        supabase_client.search_documents_v2(EMBEDDING, limit=5, similarity_threshold=0.42)

        assert state["payloads"][0]["match_threshold"] == pytest.approx(0.42)

    def test_filters_are_passed_through_untouched(self, monkeypatch):
        state = install_fake_client(monkeypatch, rows=[])

        supabase_client.search_documents_v2(
            EMBEDDING,
            limit=7,
            similarity_threshold=0.25,
            source_filename="strategies",
            code_only=True,
        )

        payload = state["payloads"][0]
        assert payload["match_count"] == 7
        assert payload["filter_source"] == "strategies"
        assert payload["filter_code_only"] is True
        assert payload["query_embedding"] is EMBEDDING

    def test_defaults_mean_no_floor_and_no_filters(self, monkeypatch):
        state = install_fake_client(monkeypatch, rows=[])

        supabase_client.search_documents_v2(EMBEDDING)

        payload = state["payloads"][0]
        assert payload["match_threshold"] == pytest.approx(-1.0)
        assert payload["filter_source"] is None
        assert payload["filter_code_only"] is False
        assert payload["match_count"] == 12


class TestFailureHandling:
    def test_missing_function_raises_migration_error_without_retrying(self, monkeypatch):
        error = Exception(
            "{'message': 'Could not find the function public.match_documents_v2(...) "
            "in the schema cache', 'code': 'PGRST202'}"
        )
        state = install_fake_client(monkeypatch, error=error)

        with pytest.raises(MigrationRequiredError, match="0002_match_documents_v2.sql"):
            supabase_client.search_documents_v2(EMBEDDING, limit=3)

        assert len(state["calls"]) == 1, "a missing function is not worth retrying"

    def test_transient_error_is_retried_and_then_succeeds(self, monkeypatch):
        error = ConnectionError("HTTP/2 stream reset")
        state = install_fake_client(
            monkeypatch,
            rows=[{"id": "a"}],
            error=error,
            error_every_time=False,  # fails on attempt 1, succeeds on attempt 2
        )

        rows = supabase_client.search_documents_v2(EMBEDDING, limit=3)

        assert rows == [{"id": "a"}]
        assert len(state["calls"]) == 2

    def test_persistent_transient_error_is_raised_after_three_attempts(self, monkeypatch):
        error = ConnectionError("HTTP/2 stream reset")
        state = install_fake_client(monkeypatch, error=error)

        with pytest.raises(ConnectionError):
            supabase_client.search_documents_v2(EMBEDDING, limit=3)

        assert len(state["calls"]) == 3

    def test_unreachable_database_is_not_reported_as_a_migration_problem(self, monkeypatch):
        install_fake_client(monkeypatch, error=ConnectionError("getaddrinfo failed"))

        with pytest.raises(ConnectionError):
            supabase_client.search_documents_v2(EMBEDDING, limit=3)


class TestMissingFunctionDetection:
    @pytest.mark.parametrize(
        "message",
        [
            "PGRST202",
            "Could not find the function public.match_documents_v2 in the schema cache",
            "function public.match_documents_v2 does not exist",
        ],
    )
    def test_recognised_messages(self, message):
        assert supabase_client._is_missing_function_error(Exception(message)) is True

    def test_ordinary_errors_are_not_misclassified(self):
        assert supabase_client._is_missing_function_error(ConnectionError("timed out")) is False
        assert supabase_client._is_missing_function_error(Exception("JWT expired")) is False
