import argparse
import os
import sys
import time
from pathlib import Path
from google import genai

# When running the script directly (e.g. `python server/gemini/upload_files.py`),
# the repository root may not be on sys.path which prevents `server` package
# imports. Ensure the project root is added to sys.path before importing.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
  sys.path.insert(0, str(_repo_root))

from server.config import get_config


def create_client(api_key: str | None = None):
  # Prefer explicit API key passed in; fall back to environment variables
  # First prefer explicit CLI arg, then the app config, then environment vars
  cfg = None
  try:
    cfg = get_config()
  except Exception:
    cfg = None

  key = api_key or (cfg.gemini_api_key if cfg and getattr(cfg, "gemini_api_key", None) else None) or os.environ.get("GENAI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
  if key:
    return genai.Client(api_key=key)
  # If no key found, raise a clear error so caller can show instructions
  raise ValueError(
    "Missing API key. Provide an API key via the --api-key flag or set the GENAI_API_KEY (or GOOGLE_API_KEY) environment variable."
  )


def create_file_search_store(client, display_name: str):
  return client.file_search_stores.create(config={"display_name": display_name})


def upload_file(client, store_name: str, file_path: Path, display_name: str | None = None):
  return client.file_search_stores.upload_to_file_search_store(
    file=str(file_path),
    file_search_store_name=store_name,
    config={"display_name": display_name or file_path.name},
  )


def find_processed_dir() -> Path:
  # project root is two levels up from this script (server/gemini)
  root = Path(__file__).resolve().parents[2]
  return root / "pinescript_docs" / "processed"


def main():
  parser = argparse.ArgumentParser(description="Upload all processed markdown files to a File Search store")
  parser.add_argument("--store-name", help="Display name for the new File Search store", default=None)
  parser.add_argument("--dir", help="Directory with processed markdown files", default=None)
  parser.add_argument("--api-key", help="API key for Google GenAI (or set GENAI_API_KEY env var)")
  parser.add_argument("--dry-run", action="store_true", help="Don't actually upload, just list files")
  args = parser.parse_args()

  processed_dir = Path(args.dir) if args.dir else find_processed_dir()
  if not processed_dir.exists() or not processed_dir.is_dir():
    print(f"Processed directory not found: {processed_dir}")
    sys.exit(2)

  md_files = sorted(processed_dir.rglob("*.md"))
  if not md_files:
    print(f"No markdown files found in {processed_dir}")
    return

  print(f"Found {len(md_files)} markdown files in {processed_dir}")

  if args.dry_run:
    for p in md_files:
      print(p)
    return

  try:
    client = create_client(args.api_key)
  except ValueError as e:
    print(e)
    sys.exit(2)

  display_name = args.store_name or f"pinescript-processed-{int(time.time())}"
  print("Creating File Search Store with display name:", display_name)
  store = create_file_search_store(client, display_name)
  print("Created store:", store.name)

  successes = []
  failures = []

  for p in md_files:
    try:
      print("Uploading:", p)
      op = upload_file(client, store.name, p, display_name=p.name)

      # Different versions of the GenAI client return different operation objects.
      # Try common ways to obtain the final result, falling back to the op itself.
      result = None
      try:
        if hasattr(op, "result") and callable(op.result):
          result = op.result()
        elif hasattr(op, "wait") and callable(op.wait):
          result = op.wait()
        elif hasattr(op, "metadata"):
          result = op.metadata
        else:
          result = op
      except Exception as inner_e:
        print("Warning: couldn't fetch operation result:", inner_e)
        result = op

      uploaded_name = getattr(result, "name", None) or getattr(op, "name", None) or p.name
      print("Uploaded:", p, "->", uploaded_name)
      successes.append(p)
    except KeyboardInterrupt:
      print("Interrupted by user; stopping uploads.")
      break
    except Exception as e:
      print("Failed to upload", p, "->", e)
      failures.append((p, e))

  print("\nUpload summary:")
  print("Successes:", len(successes))
  print("Failures:", len(failures))
  if failures:
    for p, e in failures:
      print(" -", p, ":", e)
    sys.exit(1)


if __name__ == "__main__":
  main()

