import os
import time
from pathlib import Path

import pytest

from google import genai
from google.genai import types

from server.config import get_config


@pytest.mark.integration
def test_gemini_file_search_integration(tmp_path):
    """Integration test: create a file search store, upload a small file, query via FileSearch tool.

    Skips if no Gemini API key is available in config or environment.
    """
    key = None
    try:
        cfg = get_config()
        key = getattr(cfg, "gemini_api_key", None)
    except Exception:
        key = None

    key = key or os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("No Gemini API key configured; skipping integration test")

    client = genai.Client(api_key=key)

    # Create a temporary file search store
    store = client.file_search_stores.create(config={"display_name": f"test-store-{int(time.time())}"})

    sample = tmp_path / "sample.txt"
    sample.write_text("Robert Graves was an English poet, novelist, and critic born in 1895.")

    op = client.file_search_stores.upload_to_file_search_store(
        file=str(sample),
        file_search_store_name=store.name,
        config={"display_name": sample.name},
    )

    # Wait for upload result; handle different client versions
    try:
        if hasattr(op, "result") and callable(op.result):
            upload_res = op.result()
        elif hasattr(op, "wait") and callable(op.wait):
            upload_res = op.wait()
        elif hasattr(op, "metadata"):
            upload_res = op.metadata
        else:
            upload_res = op
    except Exception as e:
        pytest.skip(f"Upload operation failed or timed out: {e}")

    # Query the store using models.generate_content and FileSearch tool
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Can you tell me about Robert Graves?",
            config=types.GenerateContentConfig(
                tools=[
                    types.Tool(
                        file_search=types.FileSearch(
                            file_search_store_names=[store.name]
                        )
                    )
                ]
            ),
        )
    except Exception as e:
        # If model call fails, clean up store and skip the test
        try:
            if hasattr(client.file_search_stores, "delete"):
                client.file_search_stores.delete(store.name)
        finally:
            pytest.skip(f"Model generate_content failed: {e}")

    # Extract text from response (different client versions vary)
    text = ""
    if hasattr(response, "text"):
        text = response.text
    elif hasattr(response, "output_text"):
        text = response.output_text
    else:
        try:
            # Try nested candidates
            text = response.candidates[0].content[0].text
        except Exception:
            text = ""

    # Basic assertion: response should contain some text
    assert text and len(text) > 0

    # Cleanup: attempt to delete the created store
    try:
        if hasattr(client.file_search_stores, "delete"):
            client.file_search_stores.delete(store.name)
    except Exception:
        # don't fail the test on cleanup errors
        pass
