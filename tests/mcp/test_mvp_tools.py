"""End-to-end tests for the MCP tools, using the SDK's in-memory client.

The transport is skipped entirely: `Client(mcp)` connects straight to the server
object, so these tests cover tool registration, schema generation and the
service wiring without a subprocess or a port.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import pytest
from mcp import Client

from pinescript_mcp.app import mcp
from pinescript_mcp.service import EXAMPLE_SCAN_FACTOR


def make_row(
    doc_id: str = "doc-1",
    content: str = "Placeholder documentation content.",
    score: float = 0.9,
    **overrides: Any,
) -> Dict[str, Any]:
    """Build a dict shaped like one row of the `match_documents` RPC result."""
    row: Dict[str, Any] = {
        "id": doc_id,
        "content": content,
        "source_filename": "processed_3_first-indicator_20261001_094744.md",
        "chunk_index": 0,
        "chunk_count": 3,
        "section_heading": "Simple Moving Average",
        "token_count": 120,
        "code_snippet": True,
        "similarity": score,
    }
    row.update(overrides)
    return row


def call_tool(name: str, arguments: Dict[str, Any]):
    """Invoke a tool through the in-memory client and return the result."""

    async def _inner():
        async with Client(mcp) as client:
            return await client.call_tool(name, arguments)

    return asyncio.run(_inner())


def structured(result) -> Any:
    """Return the structured payload of a successful tool result."""
    assert result.is_error is False, getattr(result, "content", None)
    return result.structured_content


def error_text(result) -> str:
    """Flatten the error content of a failed tool result into one string."""
    return " ".join(block.text for block in result.content)


def registered_tools() -> List[Any]:
    """List the tools the server advertises."""

    async def _list():
        async with Client(mcp) as client:
            listing = await client.list_tools()
            return getattr(listing, "tools", listing)

    return asyncio.run(_list())


class TestRegistration:
    # Locked contract: migration 0002 changes how filtering is executed, never
    # what the tools accept or return. Any change here is an API break.
    EXPECTED_CONTRACT = {
        "search_docs": {
            "properties": {"query", "top_k", "min_score", "source", "include_code"},
            "required": {"query"},
            "defaults": {"top_k": 8, "min_score": -1.0, "source": None, "include_code": False},
        },
        "get_context": {
            "properties": {"query", "token_budget", "top_k"},
            "required": {"query"},
            "defaults": {"token_budget": 8000, "top_k": 12},
        },
        "find_examples": {
            "properties": {"topic", "limit", "min_score"},
            "required": {"topic"},
            "defaults": {"limit": 5, "min_score": -1.0},
        },
    }

    def test_exactly_the_three_mvp_tools_are_exposed(self):
        names = {tool.name for tool in registered_tools()}
        assert names == {"search_docs", "get_context", "find_examples"}

    def test_tool_input_contract_is_unchanged(self):
        for tool in registered_tools():
            expected = self.EXPECTED_CONTRACT[tool.name]
            schema = tool.input_schema
            assert set(schema["properties"]) == expected["properties"], tool.name
            assert set(schema["required"]) == expected["required"], tool.name
            for name, default in expected["defaults"].items():
                assert schema["properties"][name]["default"] == default, (tool.name, name)

    def test_tool_output_contract_is_unchanged(self):
        expected = {
            "search_docs": "DocHit",
            "get_context": "ContextPack",
            "find_examples": "ExampleHit",
        }
        for tool in registered_tools():
            assert expected[tool.name] in json.dumps(tool.output_schema), tool.name

    def test_search_docs_schema_documents_its_arguments(self):
        tool = next(t for t in registered_tools() if t.name == "search_docs")
        schema = tool.input_schema
        assert schema["required"] == ["query"]
        assert schema["properties"]["query"]["description"]
        assert schema["properties"]["top_k"]["default"] == 8
        assert schema["properties"]["top_k"]["minimum"] == 1
        assert schema["properties"]["top_k"]["maximum"] == 50

    def test_every_tool_declares_an_output_schema(self):
        for tool in registered_tools():
            assert getattr(tool, "output_schema", None), f"{tool.name} has no output schema"

    def test_tools_carry_descriptions_for_the_model(self):
        for tool in registered_tools():
            assert tool.description and tool.description.strip(), f"{tool.name} has no description"


class TestSearchDocs:
    def test_maps_provenance_and_score(self, fake_index):
        fake_index([make_row(doc_id="a", score=0.82, section_heading="Moving Averages")])

        hits = structured(call_tool("search_docs", {"query": "sma"}))["result"]

        assert len(hits) == 1
        hit = hits[0]
        assert hit["doc_id"] == "a"
        assert hit["source"] == "processed_3_first-indicator_20261001_094744.md"
        assert hit["section"] == "Moving Averages"
        assert hit["score"] == pytest.approx(0.82)
        assert hit["chunk_count"] == 3
        assert hit["has_code"] is True

    def test_missing_section_heading_becomes_null(self, fake_index):
        fake_index([make_row(section_heading=None)])

        hit = structured(call_tool("search_docs", {"query": "x"}))["result"][0]

        assert hit["section"] is None

    def test_respects_top_k(self, fake_index):
        fake_index([make_row(doc_id=f"d{i}", score=0.9 - i * 0.01) for i in range(10)])

        hits = structured(call_tool("search_docs", {"query": "x", "top_k": 3}))["result"]

        assert len(hits) == 3
        assert [h["doc_id"] for h in hits] == ["d0", "d1", "d2"]

    def test_applies_min_score_filter(self, fake_index):
        fake_index([make_row(doc_id="keep", score=0.6), make_row(doc_id="drop", score=0.05)])

        hits = structured(call_tool("search_docs", {"query": "x", "min_score": 0.3}))["result"]

        assert [h["doc_id"] for h in hits] == ["keep"]

    def test_keeps_negative_cosine_scores(self, fake_index):
        # Raw cosine can be negative; the schema must not reject it.
        fake_index([make_row(doc_id="neg", score=-0.12)])

        hits = structured(call_tool("search_docs", {"query": "x"}))["result"]

        assert hits[0]["score"] == pytest.approx(-0.12)

    def test_filters_by_source_substring(self, fake_index):
        fake_index(
            [
                make_row(doc_id="hit", source_filename="processed_42_strategies_20261001.md"),
                make_row(doc_id="miss", source_filename="processed_19_arrays_20261001.md"),
            ]
        )

        hits = structured(call_tool("search_docs", {"query": "x", "source": "strateg"}))["result"]

        assert [h["doc_id"] for h in hits] == ["hit"]

    def test_filters_on_code_flag(self, fake_index):
        fake_index([make_row(doc_id="code", code_snippet=True), make_row(doc_id="prose", code_snippet=False)])

        hits = structured(call_tool("search_docs", {"query": "x", "include_code": True}))["result"]

        assert [h["doc_id"] for h in hits] == ["code"]

    def test_requests_exactly_top_k_rows_without_over_fetching(self, fake_index):
        """Migration 0002 filters in SQL, so the 4x surplus is gone."""
        index = fake_index([make_row(doc_id=f"d{i}") for i in range(8)])

        call_tool("search_docs", {"query": "x", "top_k": 2})

        assert index.search_calls[0]["limit"] == 2

    def test_filtering_is_delegated_to_the_database(self, fake_index):
        """Every criterion must reach the RPC; the tool re-filters nothing."""
        index = fake_index([make_row()])

        call_tool(
            "search_docs",
            {"query": "x", "top_k": 5, "min_score": 0.42, "source": "strateg", "include_code": True},
        )

        call = index.search_calls[0]
        assert call["similarity_threshold"] == pytest.approx(0.42)
        assert call["source_filename"] == "strateg"
        assert call["code_only"] is True
        assert call["limit"] == 5

    def test_code_only_flag_is_forwarded_when_not_requested(self, fake_index):
        index = fake_index([make_row()])

        call_tool("search_docs", {"query": "x"})

        call = index.search_calls[0]
        assert call["code_only"] is False
        assert call["source_filename"] is None
        assert call["similarity_threshold"] == pytest.approx(-1.0)

    def test_truncates_long_excerpts_and_flags_it(self, fake_index):
        fake_index([make_row(content="line\n" * 400)])

        hit = structured(call_tool("search_docs", {"query": "x"}))["result"][0]

        assert hit["truncated"] is True
        assert hit["excerpt"].endswith("[...]")
        assert len(hit["excerpt"]) < 600

    def test_short_content_is_not_flagged_as_truncated(self, fake_index):
        fake_index([make_row(content="short body")])

        hit = structured(call_tool("search_docs", {"query": "x"}))["result"][0]

        assert hit["truncated"] is False
        assert hit["excerpt"] == "short body"

    def test_empty_query_short_circuits_without_touching_the_index(self, fake_index):
        index = fake_index([make_row()])

        hits = structured(call_tool("search_docs", {"query": "   "}))["result"]

        assert hits == []
        assert index.embed_calls == []


class TestGetContext:
    def test_returns_passages_with_provenance(self, fake_index):
        fake_index([make_row(doc_id="a", token_count=50, section_heading="Arrays")])

        pack = structured(call_tool("get_context", {"query": "arrays", "token_budget": 1000}))

        assert pack["query"] == "arrays"
        assert pack["budget"] == 1000
        assert pack["documents_considered"] == 1
        assert pack["total_tokens"] == 50
        assert len(pack["passages"]) == 1
        passage = pack["passages"][0]
        assert passage["doc_id"] == "a"
        assert passage["section"] == "Arrays"
        assert passage["content"] == "Placeholder documentation content."

    def test_stops_adding_passages_at_the_budget(self, fake_index):
        fake_index([make_row(doc_id=f"d{i}", token_count=100) for i in range(5)])

        pack = structured(call_tool("get_context", {"query": "x", "token_budget": 350}))

        assert pack["total_tokens"] <= 350
        assert 1 <= len(pack["passages"]) < 5

    def test_clamps_budget_to_the_configured_prompt_budget(self, fake_index):
        fake_index([make_row(token_count=10)])

        pack = structured(call_tool("get_context", {"query": "x", "token_budget": 20000}))

        # The config default of 16000 acts as the ceiling.
        assert pack["budget"] == 16000

    def test_returns_empty_pack_when_nothing_matches(self, fake_index):
        fake_index([])

        pack = structured(call_tool("get_context", {"query": "nothing"}))

        assert pack["passages"] == []
        assert pack["total_tokens"] == 0
        assert pack["documents_considered"] == 0

    def test_requests_exactly_top_k_and_applies_no_filters(self, fake_index):
        index = fake_index([make_row(token_count=10)])

        call_tool("get_context", {"query": "x", "top_k": 3})

        call = index.search_calls[0]
        assert call["limit"] == 3
        assert call["source_filename"] is None
        assert call["code_only"] is False


class TestFindExamples:
    CODE = '//@version=6\nindicator("SMA")\nplot(ta.sma(close, 14))\n'

    def test_extracts_the_longest_code_block_and_counts_lines(self, fake_index):
        content = f"Intro.\n\n```pine\n{self.CODE}```\n\n```pine\nplot(close)\n```\n"
        fake_index([make_row(content=content)])

        examples = structured(call_tool("find_examples", {"topic": "sma"}))["result"]

        assert len(examples) == 1
        assert "ta.sma(close, 14)" in examples[0]["code"]
        assert examples[0]["language"] == "pine"
        assert examples[0]["line_count"] == 3

    def test_skips_chunks_without_usable_code(self, fake_index):
        fake_index([make_row(doc_id="prose", content="No fenced code here, just prose.")])

        examples = structured(call_tool("find_examples", {"topic": "x"}))["result"]

        assert examples == []

    def test_ignores_blocks_below_the_minimum_length(self, fake_index):
        fake_index([make_row(content="```pine\nplot(close)\n```\n")])

        examples = structured(call_tool("find_examples", {"topic": "x"}))["result"]

        assert examples == []

    def test_respects_limit(self, fake_index):
        fake_index(
            [make_row(doc_id=f"d{i}", content=f"```pine\n{self.CODE}// {i}\n```\n") for i in range(6)]
        )

        examples = structured(call_tool("find_examples", {"topic": "x", "limit": 2}))["result"]

        assert len(examples) == 2

    def test_caps_very_large_code_blocks(self, fake_index):
        big = "\n".join(f"plot(close + {i})" for i in range(500))
        fake_index([make_row(content=f"```pine\n{big}\n```\n")])

        example = structured(call_tool("find_examples", {"topic": "x"}))["result"][0]

        assert example["truncated"] is True
        assert example["code"].endswith("// [...]")
        assert len(example["code"]) < 2100

    def test_scan_is_restricted_to_code_chunks_in_sql(self, fake_index):
        index = fake_index([make_row(content=f"```pine\n{self.CODE}\n```\n")])

        call_tool("find_examples", {"topic": "x", "limit": 3})

        call = index.search_calls[0]
        assert call["code_only"] is True
        assert call["limit"] == 3 * EXAMPLE_SCAN_FACTOR

    def test_tolerates_code_flagged_chunks_without_fenced_blocks(self, fake_index):
        """Justifies the remaining scan factor: the flag also counts inline code."""
        rows = [make_row(doc_id="inline", content="Call `plot(close)` in prose.")]
        rows += [
            make_row(doc_id=f"d{i}", content=f"```pine\n{self.CODE}// {i}\n```\n") for i in range(3)
        ]
        fake_index(rows)

        examples = structured(call_tool("find_examples", {"topic": "x", "limit": 2}))["result"]

        assert len(examples) == 2
        assert all(example["doc_id"] != "inline" for example in examples)


class TestErrorHandling:
    def test_upstream_failure_becomes_a_tool_error(self, failing_index):
        result = call_tool("search_docs", {"query": "sma"})

        assert result.is_error is True
        text = error_text(result)
        assert "Documentation search failed" in text
        assert "ConnectionError" in text
        # The raw upstream message must not leak to the model.
        assert "simulated upstream outage" not in text

    def test_missing_configuration_produces_an_actionable_message(self, monkeypatch):
        class FakeConfigError(Exception):
            pass

        def _unconfigured(*args: Any, **kwargs: Any):
            raise FakeConfigError("1 validation error for Config\nsupabase_url\n  Field required")

        monkeypatch.setattr("pinescript_mcp.service.generate_single_embedding", _unconfigured)

        result = call_tool("search_docs", {"query": "sma"})

        assert result.is_error is True
        assert "SUPABASE_URL" in error_text(result)

    def test_invalid_argument_never_reaches_the_tool_body(self, fake_index):
        index = fake_index([make_row()])

        try:
            result = call_tool("search_docs", {"query": "x", "top_k": 999})
        except Exception:
            # The SDK may surface invalid params as a protocol-level error.
            pass
        else:
            assert result.is_error is True

        assert index.embed_calls == [], "the tool body must not run on invalid input"

    def test_get_context_and_find_examples_also_translate_failures(self, failing_index):
        for name, args in (("get_context", {"query": "x"}), ("find_examples", {"topic": "x"})):
            result = call_tool(name, args)
            assert result.is_error is True, name
            assert "failed" in error_text(result), name

    def test_missing_migration_is_reported_actionably(self, monkeypatch, fake_index):
        from server.supabase_client import MigrationRequiredError

        # Patch the embedding call too, otherwise this would reach OpenAI.
        fake_index([make_row()])

        def _missing(*args: Any, **kwargs: Any):
            raise MigrationRequiredError("match_documents_v2 is not installed.")

        monkeypatch.setattr("pinescript_mcp.service.search_documents_v2", _missing)

        result = call_tool("search_docs", {"query": "sma"})

        assert result.is_error is True
        assert "0002_match_documents_v2.sql" in error_text(result)


class TestStdioEntrypoint:
    """Verify `python -m pinescript_mcp` really serves MCP over stdio."""

    def test_subprocess_starts_and_advertises_the_tools(self):
        import os
        import pathlib
        import sys

        from mcp import StdioServerParameters

        repo_root = pathlib.Path(__file__).resolve().parents[2]

        async def _probe():
            params = StdioServerParameters(
                command=sys.executable,
                args=["-m", "pinescript_mcp"],
                env=dict(os.environ),
                cwd=str(repo_root),
            )
            async with Client(params) as client:
                listing = await client.list_tools()
                return getattr(client, "server_info", None), getattr(listing, "tools", listing)

        server_info, tools = asyncio.run(_probe())

        assert server_info is not None and server_info.name == "pinescript-docs"
        assert {tool.name for tool in tools} == {"search_docs", "get_context", "find_examples"}
