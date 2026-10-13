#!/usr/bin/env python3
"""Benchmark: Python-side filtering (pre-0002) versus SQL filtering (0002).

Run from the repository root:

    python scripts/benchmark_filtering.py

WHAT IS MEASURED vs MODELLED
----------------------------
There is no PostgreSQL in this repository's environment (no docker, no psql, no
server), so nothing here executes `match_documents_v2`. Be precise about which
number means what:

  MEASURED   Payload bytes. Built from real corpus text (pinescript_docs/processed)
             and a real 1536-value pgvector text encoding, then JSON-encoded the
             way PostgREST does. Only the embedding *values* are synthetic; the
             per-value encoding cost is what is being measured.
  MEASURED   Rows requested, taken from the live code path through a stubbed RPC.
  MODELLED   Rows returned and rows discarded under the old design. The old
             behaviour was "fetch top_k * 4 unfiltered rows, filter in Python",
             reproduced here from the constant the change removed
             (SEARCH_OVERFETCH = 4). How many of those rows survive filtering
             depends on the filter's selectivity against the real embedding
             distribution, which cannot be known without the live index, so it is
             swept over a range instead of asserted.
  NOT MEASURED   Server-side latency. Reported only as the Python-side work
             (comparisons performed) with no timing claim.

For real end-to-end numbers, run against a live index:

    python scripts/benchmark_filtering.py --live
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys
import time
from typing import Any, Callable, Dict, List, Sequence, Tuple

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

EMBEDDING_DIM = 1536

# The over-fetch the removed design applied before every search. Kept here only to
# reproduce the old behaviour for comparison.
LEGACY_SEARCH_OVERFETCH = 4

# Filter selectivities to sweep. "share" is the fraction of rows that satisfy the
# filter; the old design could only ever return share * (top_k * 4) of them.
SELECTIVITIES: Sequence[Tuple[str, float]] = (
    ("no filter", 1.0),
    ("code_snippet only (measured on the corpus: 89%)", 0.89),
    ("source = one of 73 pages", 1 / 73),
    ("similarity >= 0.5 (typical)", 0.15),
)

CORPUS_DIR = PROJECT_ROOT / "pinescript_docs" / "processed"


# -----------------------------------------------------------------------------
# Measured: payload encoding
# -----------------------------------------------------------------------------


def sample_embedding(dimension: int = EMBEDDING_DIM, seed: int = 7) -> str:
    """Return a pgvector-text representation of one embedding.

    pgvector renders a vector column as the string `[v1,v2,...]` and PostgREST
    serialises that as a JSON string. Values here are drawn from a range typical
    of normalised OpenAI embeddings; the encoding cost per value is what matters
    and does not depend meaningfully on the values themselves.
    """
    rng = random.Random(seed)
    values = ", ".join(f"{rng.uniform(-0.3, 0.3):.6f}" for _ in range(dimension))
    return f"[{values}]"


def sample_chunks(limit: int) -> List[str]:
    """Read real chunk-sized text from the corpus, longest pages first."""
    if not CORPUS_DIR.is_dir():
        return ["Placeholder documentation content." * 40] * limit

    files = sorted(CORPUS_DIR.glob("*.md"), key=lambda p: -p.stat().st_size)
    texts: List[str] = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        # ~1500-token chunks is what the ingest pipeline produces.
        for start in range(0, min(len(text), 12_000), 6_000):
            texts.append(text[start : start + 6_000])
            if len(texts) >= limit:
                return texts
    return texts or ["Placeholder documentation content." * 40] * limit


def build_row_v1(chunk: str, embedding: str, index: int) -> Dict[str, Any]:
    """A row shaped like `match_documents` (v1) output, embedding included."""
    return {
        "id": f"chunk-{index}",
        "content": chunk,
        "source_filename": f"processed_{index}_page_{index:03d}.md",
        "chunk_index": 0,
        "chunk_count": 1,
        "section_heading": "Section",
        "token_count": 1200,
        "code_snippet": True,
        "metadata": {},
        "embedding": embedding,
        "created_at": "2026-10-01T09:47:44+00:00",
        "updated_at": "2026-10-01T09:47:44+00:00",
        "similarity": 0.61,
    }


def build_row_v2(chunk: str, index: int) -> Dict[str, Any]:
    """A row shaped like `match_documents_v2` output: no embedding column."""
    row = build_row_v1(chunk, "", index)
    row.pop("embedding")
    return row


def payload_bytes(rows: Sequence[Dict[str, Any]]) -> int:
    return len(json.dumps(rows, ensure_ascii=False))


def measure_payloads(top_k: int) -> Dict[str, Any]:
    """Compare bytes transferred for the same query under both designs."""
    legacy_rows = top_k * LEGACY_SEARCH_OVERFETCH
    chunks = sample_chunks(max(legacy_rows, top_k))
    embedding = sample_embedding()

    def chunk_at(i: int) -> str:
        return chunks[i % len(chunks)]

    v1_rows = [build_row_v1(chunk_at(i), embedding, i) for i in range(legacy_rows)]
    v2_rows = [build_row_v2(chunk_at(i), i) for i in range(top_k)]

    v1_bytes = payload_bytes(v1_rows)
    v2_bytes = payload_bytes(v2_rows)

    # Per-row cost, to separate the two effects: dropping the embedding column
    # changes the size of a row, while fetching 4x fewer rows changes how many
    # rows there are. Reporting only their product would conflate them.
    one_v1 = payload_bytes([build_row_v1(chunk_at(0), embedding, 0)])
    one_v2 = payload_bytes([build_row_v2(chunk_at(0), 0)])

    return {
        "top_k": top_k,
        "legacy_rows": len(v1_rows),
        "legacy_bytes": v1_bytes,
        "new_rows": len(v2_rows),
        "new_bytes": v2_bytes,
        "bytes_ratio": (v1_bytes / v2_bytes) if v2_bytes else float("inf"),
        "bytes_eliminated": (1 - (v2_bytes / v1_bytes)) if v1_bytes else 0.0,
        "row_bytes_v1": one_v1,
        "row_bytes_v2": one_v2,
        "embedding_share_of_row": (1 - (one_v2 / one_v1)) if one_v1 else 0.0,
    }


# -----------------------------------------------------------------------------
# Measured: rows requested by the live code path
# -----------------------------------------------------------------------------


def rows_requested_by_current_code(top_k: int, code_only: bool) -> int:
    """What the current service asks the RPC for. Measured, not assumed.

    Both outbound calls are stubbed: the embedding call is here only so that
    measuring the row count never reaches the network.
    """
    from pinescript_mcp import service

    captured: Dict[str, Any] = {}

    class _StubRpc:
        def __call__(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            captured["limit"] = kwargs.get("limit", args[1] if len(args) > 1 else None)
            return []

    original_search = service.search_documents_v2
    original_embed = service.generate_single_embedding
    try:
        service.search_documents_v2 = _StubRpc()  # type: ignore[assignment]
        service.generate_single_embedding = (  # type: ignore[assignment]
            lambda *a, **k: [0.0] * EMBEDDING_DIM
        )
        service.search_docs("benchmark query", top_k=top_k, include_code=code_only)
    finally:
        service.search_documents_v2 = original_search  # type: ignore[assignment]
        service.generate_single_embedding = original_embed  # type: ignore[assignment]

    return int(captured.get("limit") or 0)


# -----------------------------------------------------------------------------
# Modelled: starvation under the old design
# -----------------------------------------------------------------------------


def starve_table(top_k: int) -> List[Tuple[str, float, float, float]]:
    """Return (label, selectivity, legacy hits, new hits) for each filter.

    Legacy fetched `top_k * 4` rows with no predicate, so it returns at most
    `top_k * 4 * selectivity` matching rows and falls short of `top_k` whenever
    the filter is even mildly selective.

    The new design applies the predicate before LIMIT, so it returns up to `top_k`
    matching rows. "Up to" is load-bearing: pgvector applies filters after the
    HNSW walk, so a highly selective filter can still surprise the planner and
    return fewer rows (see note 1 in the migration, and `hnsw.iterative_scan`).
    What the change removes is the *systematic* shortfall, not the tail case.
    """
    legacy_window = top_k * LEGACY_SEARCH_OVERFETCH
    rows = []
    for label, share in SELECTIVITIES:
        legacy_hits = min(top_k, legacy_window * share)
        rows.append((label, share, legacy_hits, float(top_k)))
    return rows


# -----------------------------------------------------------------------------
# Optional: live comparison
# -----------------------------------------------------------------------------


def live_benchmark(top_k: int, repetitions: int) -> int:
    """Compare both RPCs against a real Supabase project. Requires credentials."""
    try:
        from server.embed_client import generate_single_embedding
        from server.supabase_client import init_supabase_client
    except Exception as exc:  # pragma: no cover - import guard
        print(f"Could not import the server modules: {exc}")
        return 2

    query = "ta.sma moving average crossover strategy entry"
    print(f"Embedding query with the configured model: {query!r}")
    embedding = generate_single_embedding(query)

    client = init_supabase_client()

    def timed(fn: Callable[[], Any]) -> Tuple[Any, float]:
        started = time.perf_counter()
        result = fn()
        return result, (time.perf_counter() - started) * 1000

    legacy_payload = {
        "query_embedding": embedding,
        # v1 semantics: a distance, and ignored by the function body anyway.
        "match_threshold": 0.5,
        "match_count": top_k * LEGACY_SEARCH_OVERFETCH,
    }
    new_payload = {
        "query_embedding": embedding,
        "match_threshold": -1.0,
        "match_count": top_k,
        "filter_source": None,
        "filter_code_only": False,
    }

    for label, rpc, payload in (
        ("v1 match_documents", "match_documents", legacy_payload),
        ("v2 match_documents_v2", "match_documents_v2", new_payload),
    ):
        try:
            result, elapsed = timed(lambda rpc=rpc, payload=payload: client.rpc(rpc, payload).execute())
        except Exception as exc:
            print(f"{label}: FAILED - {type(exc).__name__}: {exc}")
            return 1

        rows = result.data or []
        sizes = []
        for _ in range(max(1, repetitions - 1)):
            _, ms = timed(lambda rpc=rpc, payload=payload: client.rpc(rpc, payload).execute())
            sizes.append(ms)
        spread = f", median {sorted(sizes)[len(sizes) // 2]:.0f} ms over {len(sizes)} more runs" if sizes else ""

        print(
            f"{label}: {len(rows)} rows, {payload_bytes(rows):,} bytes, first call {elapsed:.0f} ms{spread}"
        )

    return 0


# -----------------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------------


def print_design_section(top_k: int) -> None:
    print("=" * 78)
    print("PAYLOAD AND ROW COUNT  (bytes measured; row counts measured/modelled)")
    print("=" * 78)
    print()
    print(f"{'top_k':>6} {'legacy rows':>12} {'legacy bytes':>14} {'new rows':>9} {'new bytes':>11} {'ratio':>7}")
    print("-" * 78)

    for k in (4, 8, 12, 20):
        m = measure_payloads(k)
        print(
            f"{m['top_k']:>6} {m['legacy_rows']:>12} {m['legacy_bytes']:>14,} "
            f"{m['new_rows']:>9} {m['new_bytes']:>11,} {m['bytes_ratio']:>6.1f}x"
        )

    example = measure_payloads(top_k)
    print()
    print(f"At top_k={example['top_k']}:")
    print(f"  one legacy row    {example['row_bytes_v1']:>10,} bytes")
    print(f"  one new row       {example['row_bytes_v2']:>10,} bytes")
    print(
        f"  the embedding column is {example['embedding_share_of_row'] * 100:.1f}% "
        "of a legacy row"
    )
    print(f"  rows per query    {example['legacy_rows']:>10} -> {example['new_rows']}")
    print(
        f"  bytes eliminated  {example['bytes_eliminated'] * 100:.1f}% "
        f"(row reduction x per-row reduction)"
    )
    print()

    print("=" * 78)
    print("ROWS REQUESTED BY THE LIVE CODE PATH  (measured)")
    print("=" * 78)
    print()
    print(f"{'tool call':<44} {'legacy limit':>12} {'new limit':>10}")
    print("-" * 78)
    for label, top_k_value, code_only in (
        ("search_docs(top_k=8)", 8, False),
        ("search_docs(top_k=8, include_code=True)", 8, True),
        ("search_docs(top_k=3)", 3, False),
    ):
        new_limit = rows_requested_by_current_code(top_k_value, code_only)
        legacy_limit = min(top_k_value * LEGACY_SEARCH_OVERFETCH, 50)
        print(f"{label:<44} {legacy_limit:>12} {new_limit:>10}")
    print()
    print("The new limit equals top_k because the database applies the predicate before")
    print("LIMIT. The legacy limit is the removed constant, reproduced for comparison.")
    print()

    print("=" * 78)
    print("STARVATION UNDER THE OLD DESIGN  (modelled, swept over selectivities)")
    print("=" * 78)
    print()
    print(f"{'filter':<46} {'hits legacy':>12} {'hits new':>9}")
    print("-" * 78)
    for label, share, legacy_hits, new_hits in starve_table(top_k):
        print(f"{label:<46} {legacy_hits:>12.1f} {new_hits:>9.0f}")
    print()
    print(f"At top_k={top_k} the legacy window is {top_k * LEGACY_SEARCH_OVERFETCH} unfiltered rows.")
    print("Legacy applied the predicate after LIMIT, so a selective filter returns fewer")
    print("than top_k hits. That systematic shortfall is what 0002 removes.")
    print("The new column is the ceiling, not a guarantee: pgvector applies filters after")
    print("the HNSW walk, so a very selective filter can still return fewer rows.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="Also benchmark against the real Supabase project")
    parser.add_argument("--top-k", type=int, default=8, help="top_k to use for the report (default 8)")
    parser.add_argument("--repetitions", type=int, default=3, help="Live repetitions (default 3)")
    args = parser.parse_args()

    print()
    print(f"corpus: {CORPUS_DIR}")
    print(f"documents: {len(list(CORPUS_DIR.glob('*.md')))}" if CORPUS_DIR.is_dir() else "corpus not found")
    print()

    print_design_section(args.top_k)

    if args.live:
        print()
        print("=" * 78)
        print("LIVE COMPARISON  (measured against Supabase)")
        print("=" * 78)
        print()
        return live_benchmark(args.top_k, args.repetitions)

    print()
    print("Run with --live to measure both RPCs against a real Supabase project.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
