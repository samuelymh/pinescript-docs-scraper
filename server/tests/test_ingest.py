"""Unit tests for document ingestion pipeline.

Tests parsing, chunking, code detection, and manifest diffing.
"""
import pytest
from pathlib import Path
from datetime import datetime
from server.ingest import (
    parse_document,
    chunk_document,
    check_manifest,
    extract_token_overlap
)
from server.models import FileManifest
from server.utils import count_tokens, hash_string


# Test fixtures

@pytest.fixture
def small_markdown_content():
    """Markdown content under token threshold (< 1500 tokens)."""
    return """## Welcome to PineScript

Pine Script® is TradingView's programming language.

### Simple Example

Here's a simple indicator:

```pine
//@version=6
indicator("My First Indicator")
plot(close)
```

This script plots the closing price.
"""


@pytest.fixture
def large_markdown_content():
    """Markdown content over token threshold (> 1500 tokens)."""
    # Create content with multiple sections that exceeds 1500 tokens
    sections = []
    
    sections.append("""## Introduction to Strategies

Strategies in PineScript allow you to backtest trading systems.
They simulate trades across historical data and provide performance metrics.

### Key Features

Strategies have access to the strategy namespace with functions for:
- Order placement and management
- Position tracking
- Performance analysis
- Risk management

The broker emulator simulates realistic trade execution using chart data.
When creating strategies, you need to understand order types, position sizing,
risk management, and performance metrics. The strategy function provides access
to a comprehensive set of tools for building and testing trading systems.
""")
    
    # Add multiple detailed sections with more content to exceed token threshold
    for i in range(10):
        sections.append(f"""
## Section {i + 1}: Advanced Strategy Concepts

This section covers advanced strategy concepts that are important for
building robust trading systems. Lorem ipsum dolor sit amet, consectetur
adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna
aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris
nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in 
reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur.

### Subsection {i + 1}.1: Implementation Details

When implementing strategies, consider the following important aspects:
Market conditions vary significantly across different timeframes and instruments.
Your strategy should be robust enough to handle trending markets, ranging markets,
and highly volatile periods. Risk management is crucial for long-term success.
Always implement proper position sizing based on your account size and risk tolerance.

Consider using stop losses to protect your capital from unexpected market moves.
Take profit levels help lock in gains and prevent giving back profits during reversals.
The strategy tester provides comprehensive performance metrics including win rate,
profit factor, maximum drawdown, Sharpe ratio, and many other important statistics.

```pine
//@version=6
strategy("Advanced Example {i + 1}", overlay=true, margin_long=100, margin_short=100)

// Input parameters
length = input.int(14, "Length", minval=1)
multiplier = input.float(2.0, "Multiplier", minval=0.1, step=0.1)

// Calculate indicators
fastMA = ta.sma(close, length)
slowMA = ta.sma(close, length * 2)
atr = ta.atr(14)

// Entry conditions
longCondition = ta.crossover(fastMA, slowMA) and close > ta.ema(close, 200)
shortCondition = ta.crossunder(fastMA, slowMA) and close < ta.ema(close, 200)

// Execute trades
if longCondition
    strategy.entry("Long", strategy.long)
    strategy.exit("Exit Long", "Long", stop=close - atr * multiplier, limit=close + atr * multiplier * 2)

if shortCondition
    strategy.entry("Short", strategy.short)
    strategy.exit("Exit Short", "Short", stop=close + atr * multiplier, limit=close - atr * multiplier * 2)

// Plot indicators
plot(fastMA, "Fast MA", color.blue)
plot(slowMA, "Slow MA", color.orange)
```

### Subsection {i + 1}.2: Performance Optimization

More detailed explanations about trading logic, risk management, and
performance optimization. The strategy should handle various market conditions
including trending, ranging, and volatile markets. Consider implementing
proper position sizing, stop losses, and take profit levels for optimal results.

Backtesting is an essential step in strategy development but it's important to
avoid overfitting. Use out-of-sample testing and walk-forward analysis to validate
your strategy's performance. Monitor metrics like maximum drawdown, win rate, and
profit factor to ensure your strategy maintains consistent performance across
different market conditions and time periods.
""")
    
    return "\n".join(sections)


@pytest.fixture
def code_pattern_examples():
    """Examples of different code patterns to detect."""
    return {
        "triple_backticks": """
Here's an example:
```pine
indicator("Test")
plot(close)
```
""",
        "single_backticks": """
Use the `ta.sma()` function to calculate moving averages.
The `close` variable contains the closing price.
""",
        "pine_marker": """
Pine Script®
Copied
`indicator("Test")
plot(close)`
""",
        "no_code": """
This is plain text without any code snippets.
Just documentation and explanations.
"""
    }


@pytest.fixture
def temp_markdown_file(tmp_path, small_markdown_content):
    """Create a temporary markdown file for testing."""
    filepath = tmp_path / "test_doc.md"
    filepath.write_text(small_markdown_content)
    return filepath


@pytest.fixture
def sample_manifest():
    """Sample file manifest for testing."""
    return {
        "file1.md": FileManifest(
            filename="file1.md",
            content_hash="abc123",
            last_indexed=datetime(2025, 11, 1),
            doc_id="doc1"
        ),
        "file2.md": FileManifest(
            filename="file2.md",
            content_hash="def456",
            last_indexed=datetime(2025, 11, 1),
            doc_id="doc2"
        )
    }


# Tests for code detection

def test_detect_triple_backticks(code_pattern_examples):
    """Test detection of triple backtick code blocks."""
    from server.utils import detect_code_snippets
    
    assert detect_code_snippets(code_pattern_examples["triple_backticks"]) is True


def test_detect_single_backticks(code_pattern_examples):
    """Test detection of single backtick inline code."""
    from server.utils import detect_code_snippets
    
    assert detect_code_snippets(code_pattern_examples["single_backticks"]) is True


def test_detect_pine_marker(code_pattern_examples):
    """Test detection of Pine Script® marker pattern."""
    from server.utils import detect_code_snippets
    
    assert detect_code_snippets(code_pattern_examples["pine_marker"]) is True


def test_no_code_detection(code_pattern_examples):
    """Test that plain text without code returns False."""
    from server.utils import detect_code_snippets
    
    assert detect_code_snippets(code_pattern_examples["no_code"]) is False


# Tests for chunking

def test_small_document_no_chunking(small_markdown_content):
    """Test that small documents are not chunked."""
    chunks = chunk_document(
        small_markdown_content,
        "test.md",
        chunk_token_threshold=1500,
        overlap_tokens=150
    )
    
    assert len(chunks) == 1
    assert chunks[0][1] is None  # No heading
    assert chunks[0][2] == 0  # Chunk index 0


def test_large_document_chunking(large_markdown_content):
    """Test that large documents are chunked by headings."""
    chunks = chunk_document(
        large_markdown_content,
        "test.md",
        chunk_token_threshold=1500,
        overlap_tokens=150
    )
    
    # Should produce multiple chunks
    assert len(chunks) > 1
    
    # Each chunk should have a heading (except maybe first)
    headings = [chunk[1] for chunk in chunks if chunk[1] is not None]
    assert len(headings) > 0
    
    # Chunk indices should be sequential
    indices = [chunk[2] for chunk in chunks]
    assert indices == list(range(len(chunks)))


def test_chunk_overlap():
    """Test that chunks have proper overlap."""
    content = """## Section 1

This is the first section with some content that should be used for overlap.
It contains multiple sentences to ensure we have enough text.

## Section 2

This is the second section. It should start with overlap from the previous section.

## Section 3

This is the third section with even more content.
"""
    
    chunks = chunk_document(
        content,
        "test.md",
        chunk_token_threshold=50,  # Low threshold to force chunking
        overlap_tokens=20
    )
    
    # Should have multiple chunks
    assert len(chunks) > 1
    
    # Check that later chunks contain overlap marker
    for i in range(1, len(chunks)):
        chunk_content = chunks[i][0]
        # Overlap is indicated by "... " prefix
        if "... " in chunk_content:
            assert chunk_content.index("... ") < 100  # Near the start


# Tests for parsing

def test_parse_small_document(temp_markdown_file):
    """Test parsing a small markdown file."""
    documents = parse_document(temp_markdown_file)
    
    assert len(documents) == 1
    doc = documents[0]
    
    assert doc.source_filename == "test_doc.md"
    assert doc.chunk_index == 0
    assert doc.chunk_count == 1
    assert doc.section_heading is None
    assert doc.token_count > 0
    assert doc.code_snippet is True  # Contains code block
    assert doc.embedding is None  # Not yet embedded


def test_parse_document_with_code_detection(tmp_path):
    """Test that code detection works during parsing."""
    # Document with code
    filepath = tmp_path / "with_code.md"
    filepath.write_text("Here's code: `plot(close)`")
    docs = parse_document(filepath)
    assert docs[0].code_snippet is True
    
    # Document without code
    filepath = tmp_path / "without_code.md"
    filepath.write_text("This is plain text without any code.")
    docs = parse_document(filepath)
    assert docs[0].code_snippet is False


def test_parse_empty_file(tmp_path):
    """Test parsing an empty file."""
    filepath = tmp_path / "empty.md"
    filepath.write_text("")
    
    documents = parse_document(filepath)
    assert len(documents) == 0


def test_parse_generates_deterministic_ids(temp_markdown_file):
    """Test that document IDs are deterministic."""
    docs1 = parse_document(temp_markdown_file)
    docs2 = parse_document(temp_markdown_file)
    
    assert docs1[0].id == docs2[0].id


# Tests for manifest checking

def test_check_manifest_new_file(tmp_path, sample_manifest):
    """Test detection of new files."""
    new_file = tmp_path / "new_file.md"
    new_file.write_text("New content")
    
    files = [new_file]
    new_files, modified_files, unchanged_files = check_manifest(files, sample_manifest)
    
    assert len(new_files) == 1
    assert new_files[0] == new_file
    assert len(modified_files) == 0
    assert len(unchanged_files) == 0


def test_check_manifest_modified_file(tmp_path, sample_manifest):
    """Test detection of modified files."""
    modified_file = tmp_path / "file1.md"
    modified_file.write_text("Modified content")
    
    files = [modified_file]
    new_files, modified_files, unchanged_files = check_manifest(files, sample_manifest)
    
    assert len(new_files) == 0
    assert len(modified_files) == 1
    assert modified_files[0] == modified_file
    assert len(unchanged_files) == 0


def test_check_manifest_unchanged_file(tmp_path, sample_manifest):
    """Test detection of unchanged files."""
    # Create file with same hash as in manifest
    unchanged_file = tmp_path / "file1.md"
    content = "original content"
    unchanged_file.write_text(content)
    
    # Update manifest with correct hash
    content_hash = hash_string(content)
    sample_manifest["file1.md"].content_hash = content_hash
    
    files = [unchanged_file]
    new_files, modified_files, unchanged_files = check_manifest(files, sample_manifest)
    
    assert len(new_files) == 0
    assert len(modified_files) == 0
    assert len(unchanged_files) == 1
    assert unchanged_files[0] == unchanged_file


def test_check_manifest_mixed_files(tmp_path):
    """Test manifest check with mix of new, modified, and unchanged files."""
    manifest = {}
    
    # Create existing file
    existing = tmp_path / "existing.md"
    existing.write_text("existing")
    manifest["existing.md"] = FileManifest(
        filename="existing.md",
        content_hash=hash_string("existing"),
        last_indexed=datetime.now(),
        doc_id="doc1"
    )
    
    # Create modified file
    modified = tmp_path / "modified.md"
    modified.write_text("new content")
    manifest["modified.md"] = FileManifest(
        filename="modified.md",
        content_hash=hash_string("old content"),
        last_indexed=datetime.now(),
        doc_id="doc2"
    )
    
    # Create new file
    new = tmp_path / "new.md"
    new.write_text("new")
    
    files = [existing, modified, new]
    new_files, modified_files, unchanged_files = check_manifest(files, manifest)
    
    assert len(new_files) == 1
    assert len(modified_files) == 1
    assert len(unchanged_files) == 1


# Tests for token overlap extraction

def test_extract_token_overlap_short_text():
    """Test overlap extraction with text shorter than target."""
    text = "Short text"
    overlap = extract_token_overlap(text, 100)
    assert overlap == text


def test_extract_token_overlap_long_text():
    """Test overlap extraction with long text."""
    text = "First sentence. Second sentence. Third sentence. Fourth sentence. Fifth sentence."
    overlap = extract_token_overlap(text, 10)
    
    # Should extract from end
    assert len(overlap) < len(text)
    assert overlap in text
    # Should try to break at sentence boundary
    assert ". " in overlap or overlap.startswith("Fifth")


def test_token_counting_consistency():
    """Test that token counting is consistent."""
    text = "This is a test with some tokens to count."
    
    count1 = count_tokens(text)
    count2 = count_tokens(text)
    
    assert count1 == count2
    assert count1 > 0


# Integration-style tests (optional, can be skipped if no test DB)

@pytest.mark.skip(reason="Requires Supabase test database")
def test_full_indexing_pipeline():
    """Integration test for full indexing pipeline (requires test DB)."""
    # This would test the full index_documents() flow
    # Skipped by default as it requires Supabase setup
    pass


# Optional live Supabase integration tests
import os


@pytest.mark.skipif(
    not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY")),
    reason="Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables"
)
def test_live_supabase_upsert_and_delete():
    """Test upserting a Document to live Supabase and deleting by filename.

    This test runs only when the required Supabase env vars are present.
    It performs a lightweight upsert and cleanup to avoid interfering with
    existing data.
    """
    from server.models import Document
    from server.supabase_client import upsert_documents, delete_documents_by_filename
    from server.utils import generate_doc_id
    from server.embed_client import EMBEDDING_DIM
    from datetime import datetime

    # Build a minimal document
    filename = f"test_live_{int(datetime.now().timestamp())}.md"
    doc_id = generate_doc_id(filename, 0)
    doc = Document(
        id=doc_id,
        content="# Live test\nThis is a live supabase test.",
        source_filename=filename,
        chunk_index=0,
        chunk_count=1,
        section_heading=None,
        token_count=10,
        code_snippet=False,
        metadata={"test": True},
        embedding=[0.0] * EMBEDDING_DIM
    )

    # Upsert to Supabase
    result = upsert_documents([doc])
    assert result.get("success") is True
    assert result.get("count", 0) >= 1

    # Cleanup by deleting documents for the test filename
    del_result = delete_documents_by_filename(filename)
    assert del_result.get("success") is True


@pytest.mark.skipif(
    not (os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_ROLE_KEY")),
    reason="Requires SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables"
)
def test_live_supabase_manifest_roundtrip():
    """Test updating and fetching a manifest entry in live Supabase."""
    from server.supabase_client import update_manifest, fetch_manifest
    from server.models import FileManifest
    from datetime import datetime
    # Ensure a matching documents row exists because `file_manifest.doc_id` has
    # a foreign key constraint referencing `documents.id` in the live DB.
    from server.models import Document
    from server.supabase_client import upsert_documents
    from server.embed_client import EMBEDDING_DIM

    filename = f"manifest_test_{int(datetime.now().timestamp())}.md"
    entry = FileManifest(
        filename=filename,
        content_hash="deadbeef",
        last_indexed=datetime.now(),
        doc_id="doc_live_test"
    )

    # Upsert a minimal document with the same doc_id so the FK constraint
    # on `file_manifest.doc_id` is satisfied.
    doc = Document(
        id=entry.doc_id,
        content="manifest placeholder",
        source_filename=filename,
        chunk_index=0,
        chunk_count=1,
        section_heading=None,
        token_count=1,
        code_snippet=False,
        metadata={"test": True},
        embedding=[0.0] * EMBEDDING_DIM,
    )

    upsert_documents([doc])

    up_result = update_manifest([entry])
    assert up_result.get("success") is True
    assert up_result.get("count", 0) >= 1

    manifest = fetch_manifest()
    assert filename in manifest

    # Cleanup: remove manifest entry and any documents (best-effort)
    from server.supabase_client import delete_documents_by_filename
    delete_documents_by_filename(filename)

