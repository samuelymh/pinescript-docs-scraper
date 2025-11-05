# PineScript V6 Documentation Crawler

A Python-based tool for crawling and processing TradingView's Pine Script V6 documentation, built using the Crawl4Ai framework. This tool extracts, cleans, and organizes the documentation, making it easier to reference and analyze. Crawl4Ai provides the core framework for web crawling, data extraction, and asynchronous processing, making it possible.

## Features

### Crawling

- Automatically extracts documentation from TradingView's Pine Script V6 website using Crawl4Ai
- Efficiently handles navigation through documentation pages
- Supports batch processing with rate limiting
- Maintains a structured extraction schema for consistent results
- Saves individual pages into an `unprocessed/` folder and also writes a combined raw file

### Content Processing

- Cleans and formats documentation content
- Preserves PineScript code blocks with proper syntax highlighting
- Extracts and formats function documentation
- Removes unnecessary navigation elements and formatting
- Processes content into a clean, readable markdown format

### Output Organization

- Creates individual markdown files for each documentation page (raw output)
- Raw (unprocessed) markdown files are saved to `pinescript_docs/unprocessed/`
- A combined raw file `all_docs_{timestamp}.md` is written to `pinescript_docs/`
- The processor reads from `pinescript_docs/unprocessed/` and writes enhanced files to `pinescript_docs/processed/`
- The processor also writes a combined processed file `processed_all_docs.md` into the repository root (same directory as the scripts)
- Tracks failed URLs and crawling statistics
- Preserves original source URLs and timestamps

## Setup

1. Clone the repository:

   ```bash
   git clone git@github.com:samuelymh/pinescript-docs-scraper.git
   cd pinescript-docs-scraper
   ```

2. Install required dependencies:
   ```bash
    # it's recommended to use a virtual environment
    python -m pip install -r requirements.txt
   ```

## Usage

1.  **Crawling Documentation**:

    Run the crawler:

    ```bash
    python 1_scrap_docs.py
    ```

    This script will collect documentation URLs, download content, and save raw markdown files to `pinescript_docs/unprocessed/` and a combined raw `all_docs_{timestamp}.md` to `pinescript_docs/`.

2.  **Processing Documentation**:

    To clean and organize the crawled content, run:

    ```bash
    python 2_process_docs.py
    ```

3.  **Run both (crawl then process) using the orchestrator**:

    A convenience script `3_scrap_and_process.py` was added to run the
    crawler and then the processor in sequence (or to run either step
    individually). It loads the two scripts and invokes their main
    behaviors so you don't need to run them separately.

    ```bash
    # Run crawl then process
    python 3_scrap_and_process.py

    # Only crawl (no processing)
    python 3_scrap_and_process.py --crawl-only

    # Only process (useful if you've already crawled)
    python 3_scrap_and_process.py --process-only

    # Reduce console output
    python 3_scrap_and_process.py --no-verbose
    ```

    This script reads raw markdown files from `pinescript_docs/unprocessed/`, extracts code examples and function documentation, and writes processed versions to `pinescript_docs/processed/`.
    It also writes a combined `processed_all_docs.md` next to the scripts (repository root) for easy access.

## Output Structure

```
pinescript_docs/
├── all_docs_{timestamp}.md           # Combined raw documentation (from crawler)
├── unprocessed/                      # Raw markdown files produced by the crawler
│   └── {index}_{page_name}_{timestamp}.md
├── failed_urls_{timestamp}.txt       # Failed crawl attempts
└── processed/                        # Enhanced content produced by the processor
    └── processed_{page_name}_{timestamp}.md

processed_all_docs.md                  # Combined processed file (written to repository root)
```

## Running the server (RAG API)

The full deployment and operational instructions are maintained in `docs/DEPLOYMENT.md` (operator-focused). For local development and quick commands see `server/STARTUP.md` (developer-focused).

- Quick start (dev): see `server/STARTUP.md`
- Production / deployment: see `docs/DEPLOYMENT.md`

In short: apply SQL migrations in `migrations/`, set required secrets (see `docs/DEPLOYMENT.md`), build the Docker image (optional), and either run the server locally with `uvicorn` for development or use the Docker image / Gunicorn for production-like runs. Use the ingest CLI (`server/run_ingest.py`) for full reindexes to avoid worker timeouts.


## Customization

The crawler and processor can be customized through their respective class initializations:

- `PineScriptDocsCrawler`: Configures crawling behavior, batch size, and extraction schema.
- `PineScriptDocsProcessor`: Customizes content processing and output formatting.

## License

This project is open source and available under the MIT License.

## Error Handling

- Failed URLs are logged with error messages.
- Batch processing ensures resilience to temporary failures.
- Rate limiting helps avoid server overload.
