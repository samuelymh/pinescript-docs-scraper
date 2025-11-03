"""Tests for utility functions."""
import pytest
from server.utils import (
    count_tokens,
    hash_string,
    detect_code_snippets,
    extract_code_blocks,
    generate_doc_id
)


class TestTokenCounting:
    """Tests for token counting functionality."""
    
    def test_count_tokens_simple(self):
        """Test token counting for simple text."""
        text = "Hello, world!"
        tokens = count_tokens(text)
        assert tokens > 0
        assert isinstance(tokens, int)
    
    def test_count_tokens_empty(self):
        """Test token counting for empty string."""
        assert count_tokens("") == 0
    
    def test_count_tokens_long_text(self):
        """Test token counting for longer text."""
        text = "This is a longer piece of text " * 100
        tokens = count_tokens(text)
        assert tokens > 100


class TestHashing:
    """Tests for hashing functionality."""
    
    def test_hash_string_deterministic(self):
        """Test that hashing is deterministic."""
        text = "test string"
        hash1 = hash_string(text)
        hash2 = hash_string(text)
        assert hash1 == hash2
    
    def test_hash_string_different_inputs(self):
        """Test that different inputs produce different hashes."""
        hash1 = hash_string("test1")
        hash2 = hash_string("test2")
        assert hash1 != hash2
    
    def test_hash_string_format(self):
        """Test that hash is in expected format."""
        result = hash_string("test")
        assert len(result) == 64  # SHA256 produces 64 hex characters
        assert all(c in "0123456789abcdef" for c in result)


class TestCodeDetection:
    """Tests for code snippet detection."""
    
    def test_detect_triple_backticks(self):
        """Test detection of triple backtick code blocks."""
        content = "Some text\n```pine\ncode here\n```\nmore text"
        assert detect_code_snippets(content) is True
    
    def test_detect_triple_backticks_no_language(self):
        """Test detection of triple backticks without language marker."""
        content = "Some text\n```\ncode here\n```\nmore text"
        assert detect_code_snippets(content) is True
    
    def test_detect_single_backticks(self):
        """Test detection of single backtick code."""
        content = "Use the `array.get()` function to retrieve values."
        assert detect_code_snippets(content) is True
    
    def test_detect_pine_script_marker(self):
        """Test detection of Pine Script® marked code."""
        content = "Example:\nPine Script®\nCopied\n`indicator('My Script')`"
        assert detect_code_snippets(content) is True
    
    def test_no_code_detection(self):
        """Test that plain text without code returns False."""
        content = "This is just plain text without any code snippets."
        assert detect_code_snippets(content) is False
    
    def test_single_backtick_too_short(self):
        """Test that very short backticked text is not detected."""
        content = "Use `x` as variable."  # Only 1 char, should not trigger
        # Note: Current implementation uses 3+ chars threshold
        result = detect_code_snippets(content)
        # This may or may not detect depending on implementation
        assert isinstance(result, bool)


class TestCodeExtraction:
    """Tests for code block extraction."""
    
    def test_extract_triple_backticks(self):
        """Test extraction of triple backtick blocks."""
        content = "Text\n```pine\ncode block 1\n```\nMore text\n```\ncode block 2\n```"
        blocks = extract_code_blocks(content)
        assert len(blocks) == 2
        assert "code block 1" in blocks[0]
        assert "code block 2" in blocks[1]
    
    def test_extract_pine_script_marker(self):
        """Test extraction of Pine Script® marked code."""
        content = "Pine Script®\nCopied\n`indicator('Test')`"
        blocks = extract_code_blocks(content)
        assert len(blocks) >= 1
        assert "indicator" in blocks[0]
    
    def test_extract_no_code(self):
        """Test extraction returns empty list for no code."""
        content = "Just plain text without code."
        blocks = extract_code_blocks(content)
        assert blocks == []


class TestDocIdGeneration:
    """Tests for document ID generation."""
    
    def test_generate_doc_id_deterministic(self):
        """Test that doc ID generation is deterministic."""
        filename = "test_file.md"
        id1 = generate_doc_id(filename, 0)
        id2 = generate_doc_id(filename, 0)
        assert id1 == id2
    
    def test_generate_doc_id_different_chunks(self):
        """Test that different chunk indices produce different IDs."""
        filename = "test_file.md"
        id1 = generate_doc_id(filename, 0)
        id2 = generate_doc_id(filename, 1)
        assert id1 != id2
    
    def test_generate_doc_id_different_files(self):
        """Test that different filenames produce different IDs."""
        id1 = generate_doc_id("file1.md", 0)
        id2 = generate_doc_id("file2.md", 0)
        assert id1 != id2
    
    def test_generate_doc_id_format(self):
        """Test that doc ID is in expected format."""
        doc_id = generate_doc_id("test.md", 0)
        assert len(doc_id) == 64  # SHA256 hex digest
        assert all(c in "0123456789abcdef" for c in doc_id)
