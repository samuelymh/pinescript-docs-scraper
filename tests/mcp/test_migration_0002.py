"""Contract tests for migration 0002 and the client that calls it.

There is no PostgreSQL available in CI or in this environment (no docker, no
psql, no server), so these tests cannot execute the migration. What they do
instead is pin down the parts that silently drift when SQL and Python are edited
separately:

  * the RPC parameter names in the SQL must equal the payload keys the Python
    client sends -- PostgREST matches them by name, so a rename on either side
    breaks the call at runtime and nothing else would catch it;
  * the SQL body must actually *use* the parameters it declares. v1's bug was
    exactly this: it accepted `match_threshold` and never referenced it;
  * filtering must not be achievable by accident (no `ILIKE '%' || param || '%'`,
    where a caller-supplied '%' would become a wildcard);
  * the v1 function must stay untouched, because POST /chat depends on it.

If `pglast` (the real PostgreSQL parser) happens to be installed, one additional
test parses the migration and its function body. pglast is deliberately NOT a
project dependency; without it that test skips.
"""

from __future__ import annotations

import pathlib
import re
from typing import Any, Dict, List

import pytest

MIGRATIONS = pathlib.Path(__file__).resolve().parents[2] / "migrations"
V2_FILE = "0002_match_documents_v2.sql"
V1_FILE = "0001_create_documents_and_file_manifest.sql"


def read_migration(name: str) -> str:
    return (MIGRATIONS / name).read_text(encoding="utf-8")


def strip_sql_comments(sql: str) -> str:
    """Remove `--` line comments and `/* */` block comments.

    Necessary, not cosmetic: the rollback instructions at the top of the
    migration mention `match_documents_v2(vector, double precision, ...)` in a
    comment, so a naive signature search matches the comment instead of the
    definition.
    """
    without_block = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    return re.sub(r"--[^\n]*", "", without_block)


def read_code(name: str) -> str:
    """Migration source with comments removed."""
    return strip_sql_comments(read_migration(name))


def signature_parameters(sql: str, function_name: str) -> List[str]:
    """Return the parameter names of a CREATE FUNCTION signature, in order.

    Anchored on `FUNCTION <name>(` so that a `DROP FUNCTION IF EXISTS <name>(...)`
    mentioned in a comment cannot be mistaken for the definition.
    """
    match = re.search(
        rf"FUNCTION\s+(?:public\.)?{function_name}\((.*?)\)\s*RETURNS",
        sql,
        re.DOTALL,
    )
    assert match, f"could not find the signature of {function_name}"
    return [part.strip().split()[0] for part in match.group(1).split(",") if part.strip()]


def function_body(sql: str, function_name: str) -> str:
    """Return the dollar-quoted body of the named function."""
    match = re.search(
        rf"FUNCTION\s+(?:public\.)?{function_name}\(.*?AS \$\$(.*?)\$\$;",
        sql,
        re.DOTALL,
    )
    assert match, f"could not find the body of {function_name}"
    return match.group(1)


class TestV2Signature:
    def test_parameters_are_named_as_expected(self):
        assert signature_parameters(read_code(V2_FILE), "match_documents_v2") == [
            "query_embedding",
            "match_threshold",
            "match_count",
            "filter_source",
            "filter_code_only",
        ]

    def test_client_payload_keys_match_the_sql_signature(self, monkeypatch):
        """The one check that catches SQL/client drift."""
        from server import supabase_client

        captured: Dict[str, Any] = {}

        class _FakeResult:
            data: List[Dict[str, Any]] = []

        class _FakeRpc:
            def execute(self):
                return _FakeResult()

        class _FakeClient:
            def rpc(self, name: str, payload: Dict[str, Any]):
                captured["name"] = name
                captured["payload"] = payload
                return _FakeRpc()

        monkeypatch.setattr(supabase_client, "init_supabase_client", lambda: _FakeClient())

        supabase_client.search_documents_v2(
            [0.0] * 1536,
            limit=5,
            similarity_threshold=0.3,
            source_filename="strategies",
            code_only=True,
        )

        assert captured["name"] == "match_documents_v2"
        assert set(captured["payload"]) == set(
            signature_parameters(read_code(V2_FILE), "match_documents_v2")
        )

    def test_threshold_default_is_the_bottom_of_the_cosine_range(self):
        assert re.search(
            r"match_threshold DOUBLE PRECISION DEFAULT -1\.0", read_code(V2_FILE)
        )


class TestV2Body:
    def test_threshold_is_actually_applied(self):
        """v1's bug: the argument existed but the body never used it."""
        body = function_body(read_code(V2_FILE), "match_documents_v2")
        assert "1 - (d.embedding <=> query_embedding) >= match_threshold" in body

    def test_source_filter_is_a_literal_substring_not_a_wildcard_pattern(self):
        """A caller-supplied '%' must not turn the filter into a wildcard."""
        body = function_body(read_code(V2_FILE), "match_documents_v2")
        assert "strpos(lower(d.source_filename), lower(filter_source)) > 0" in body
        assert "ILIKE" not in body.upper()
        assert "||" not in body, "string concatenation would allow wildcard injection"

    def test_source_filter_is_nullable(self):
        body = function_body(read_code(V2_FILE), "match_documents_v2")
        assert "filter_source IS NULL" in body

    def test_code_only_filter_is_applied(self):
        body = function_body(read_code(V2_FILE), "match_documents_v2")
        assert "d.code_snippet IS TRUE" in body
        assert "filter_code_only IS FALSE" in body

    def test_limit_follows_the_similarity_ordering(self):
        body = function_body(read_code(V2_FILE), "match_documents_v2")
        assert "ORDER BY d.embedding <=> query_embedding" in body
        assert "LIMIT match_count" in body

    def test_embedding_column_is_not_returned(self):
        """v1 shipped the 1536-dimension vector; nothing reads it."""
        sql = read_code(V2_FILE)
        returns = re.search(r"RETURNS TABLE \((.*?)\)\s*LANGUAGE", sql, re.DOTALL).group(1)
        columns = [line.strip().split()[0] for line in returns.splitlines() if line.strip()]
        assert "embedding" not in columns
        assert "similarity" in columns
        assert "content" in columns


class TestV2ExecutionRights:
    def test_execute_is_revoked_from_public(self):
        sql = read_code(V2_FILE)
        assert "REVOKE EXECUTE" in sql and "FROM PUBLIC" in sql

    def test_execute_is_granted_to_the_role_the_application_uses(self):
        sql = read_code(V2_FILE)
        assert "GRANT EXECUTE" in sql and "TO service_role" in sql


class TestV1IsUntouched:
    def test_v1_still_exists_with_its_original_signature(self):
        """POST /chat calls this through search_similar_documents."""
        assert signature_parameters(read_code(V1_FILE), "match_documents") == [
            "query_embedding",
            "match_threshold",
            "match_count",
        ]

    def test_v1_still_ignores_its_threshold(self):
        """Preserved on purpose: fixing it would change /chat behaviour."""
        body = function_body(read_code(V1_FILE), "match_documents")
        assert "match_threshold" not in body

    def test_v1_still_returns_the_embedding_column(self):
        body = function_body(read_code(V1_FILE), "match_documents")
        assert "d.embedding," in body

    def test_v2_does_not_redefine_v1(self):
        assert "FUNCTION public.match_documents(" not in read_code(V2_FILE)


class TestOptionalParserValidation:
    def test_migration_parses_with_the_real_postgresql_parser(self):
        pglast = pytest.importorskip(
            "pglast", reason="pglast is an optional SQL syntax validator"
        )
        sql = read_migration(V2_FILE)
        statements = pglast.parse_sql(sql)
        assert len(statements) >= 4  # CREATE FUNCTION, COMMENT, REVOKE, GRANT

        pglast.parse_sql(function_body(sql, "match_documents_v2"))
