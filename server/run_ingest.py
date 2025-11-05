#!/usr/bin/env python3
"""Small CLI to run the ingestion pipeline (scan, parse, embed, index)."""
import argparse
import asyncio
import logging
from pathlib import Path
import sys

# Ensure project root is on path so `server` package imports work when run from repo root.
PROJECT_ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from server.ingest import index_documents


def main():
    parser = argparse.ArgumentParser(description="Run the PineScript docs ingest pipeline")
    parser.add_argument("--full", action="store_true", help="Run a full reindex (clear existing data)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(asctime)s %(levelname)s %(message)s")

    try:
        result = asyncio.run(index_documents(full_reindex=args.full))
        print(result)
    except Exception as e:
        logging.exception("Ingest pipeline failed")
        raise


if __name__ == "__main__":
    main()
