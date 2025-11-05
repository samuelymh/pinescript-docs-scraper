"""Pytest configuration for server tests.

Loads environment variables from the project's `.env` file so tests
that gate on `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` will run
when those values are present in `.env`.

This is intentionally lightweight and only uses `python-dotenv` which
is already listed in `requirements.txt`.
"""
from dotenv import load_dotenv, find_dotenv
"""Pytest configuration for server tests.

Loads environment variables from the project's `.env` file so tests
that gate on `SUPABASE_URL` / `SUPABASE_SERVICE_ROLE_KEY` will run
when those values are present in `.env`.

This module also provides fixtures useful for live/integration tests
that interact with a real Supabase instance. Use these fixtures only
when the required environment variables are available.
"""
from dotenv import load_dotenv, find_dotenv
import time
import pytest


# Load .env from the project root if present so pytest sees creds.
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)


@pytest.fixture
def live_manifest_entry():
    """Create a minimal placeholder Document in Supabase and yield
    `(filename, doc_id)`. Cleans up the document after the test.

    This helps satisfy the foreign-key constraint on `file_manifest.doc_id`.
    """
    # Import lazily so tests that don't use live fixtures don't require
    # Supabase client setup at import time.
    from server.utils import generate_doc_id
    from server.models import Document
    from server.supabase_client import upsert_documents, delete_documents_by_filename
    from server.embed_client import EMBEDDING_DIM

    filename = f"test_fixture_{int(time.time())}.md"
    doc_id = generate_doc_id(filename, 0)

    doc = Document(
        id=doc_id,
        content="placeholder",
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

    try:
        yield filename, doc_id
    finally:
        # Best-effort cleanup: delete documents by filename which will
        # also remove corresponding manifest rows due to ON DELETE CASCADE.
        try:
            delete_documents_by_filename(filename)
        except Exception:
            pass
