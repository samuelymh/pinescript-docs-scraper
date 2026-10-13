# PineScript MCP Server

Read-only MCP server over the Pine Script v6 documentation index that this
repository already maintains. It reuses `server/` unchanged.

| | |
|---|---|
| Package | `pinescript_mcp` |
| SDK | official Python MCP SDK `mcp` 2.3.x (**v2** API: `MCPServer`, not `FastMCP`) |
| Transport | stdio (default); Streamable HTTP optional |
| Tools | `search_docs`, `get_context`, `find_examples` |

---

## 1. Install

```bash
python -m pip install -r requirements-mcp.txt
```

`requirements-mcp.txt` adds `mcp[cli]>=2.3,<3` plus the dependency floors the SDK
raises. It does **not** replace `requirements.txt`: the MCP server imports
`server.embed_client`, `server.retriever`, `server.supabase_client` and
`server.utils`, so the server dependencies are still required.

> `mcp` v2 requires `pydantic>=2.12`. `requirements.txt` pins `pydantic>=2.10`,
> so install `requirements-mcp.txt` (which raises the floor) instead of relying
> on the old pin alone.

---

## 2. Configuration

### Required migration

Apply both SQL files in `migrations/` before using the tools:

| Migration | Provides | Needed by |
|---|---|---|
| `0001_create_documents_and_file_manifest.sql` | `documents`, `file_manifest`, `match_documents`, HNSW index | index ingestion, `POST /chat` |
| `0002_match_documents_v2.sql` | `match_documents_v2` (filtered vector search) | all three MCP tools |

Without 0002 the tools fail with a message naming the file, rather than a generic
search error. To apply them, use the Supabase SQL editor or `psql`:

```bash
psql "$DATABASE_URL" -f migrations/0001_create_documents_and_file_manifest.sql
psql "$DATABASE_URL" -f migrations/0002_match_documents_v2.sql
```

Migration 0002 is additive: it creates a new function and does not alter
`match_documents`, so `POST /chat` keeps working exactly as before.

### Environment

The tools read configuration through `server.config.get_config()`, which expects:

| Variable | Required | Purpose |
|---|---|---|
| `SUPABASE_URL` | yes | Supabase project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | yes | service-role key (server-side only) |
| `OPENAI_API_KEY` | yes | query embeddings |
| `EMBEDDING_MODEL` | no | defaults to `text-embedding-3-small` |
| `RAG_VECTOR_TABLE` | no | defaults to `documents` |
| `PROMPT_TOKEN_BUDGET` | no | defaults to `16000`; caps `get_context`'s budget |

### The one gotcha: `PYTHONPATH`

`python -m pinescript_mcp` needs the **repository root** on `sys.path`, because
both `pinescript_mcp` and `server` are top-level packages. MCP clients do not all
support a working-directory setting, so every configuration below sets
`PYTHONPATH` explicitly. Without it you will see
`No module named 'pinescript_mcp'`.

Never commit real credentials in these files. Prefer a per-user client config,
or reference environment variables your shell already exports.

---

## 3. Verify before wiring up a client

```bash
# 1. Tools are advertised? (in-process, no subprocess)
python -c "import pinescript_mcp.app as a; print(a.mcp.name)"

# 2. Inspector, with a UI
mcp dev pinescript_mcp/app.py

# 3. Run the server briefly over stdio (it waits for a client; Ctrl-C to stop)
python -m pinescript_mcp
```

For a real retrieval check the index must be populated (`server/run_ingest.py`)
and reachable; otherwise tool calls return an error result explaining that the
index or embedding service was unreachable.

---

## 4. Client configurations

Windows paths below assume the repo is at `C:\GIT\pinescript-docs-scraper` and
its virtualenv is used. Replace with your own absolute paths. On macOS/Linux the
interpreter is `.venv/bin/python` instead of
`.venv\Scripts\python.exe`.

### 4.1 GitHub Copilot Agent (VS Code)

File: `.vscode/mcp.json` (workspace) — VS Code uses the `servers` key.

```json
{
  "servers": {
    "pinescript-docs": {
      "type": "stdio",
      "command": "${workspaceFolder}\\.venv\\Scripts\\python.exe",
      "args": ["-m", "pinescript_mcp"],
      "env": {
        "PYTHONPATH": "${workspaceFolder}",
        "SUPABASE_URL": "https://your-project.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "your-service-role-key",
        "OPENAI_API_KEY": "sk-your-openai-key"
      }
    }
  }
}
```

On macOS/Linux swap `command` for `${workspaceFolder}/.venv/bin/python`.

### 4.2 Claude Code

Project-scoped file: `.mcp.json` in the repository root.

```json
{
  "mcpServers": {
    "pinescript-docs": {
      "type": "stdio",
      "command": "C:\\GIT\\pinescript-docs-scraper\\.venv\\Scripts\\python.exe",
      "args": ["-m", "pinescript_mcp"],
      "env": {
        "PYTHONPATH": "C:\\GIT\\pinescript-docs-scraper",
        "SUPABASE_URL": "https://your-project.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "your-service-role-key",
        "OPENAI_API_KEY": "sk-your-openai-key"
      }
    }
  }
}
```

Or via the CLI:

```bash
claude mcp add pinescript-docs \
  --env PYTHONPATH=C:\\GIT\\pinescript-docs-scraper \
  --env SUPABASE_URL=https://your-project.supabase.co \
  --env SUPABASE_SERVICE_ROLE_KEY=your-service-role-key \
  --env OPENAI_API_KEY=sk-your-openai-key \
  -- "C:\\GIT\\pinescript-docs-scraper\\.venv\\Scripts\\python.exe" -m pinescript_mcp
```

Verify with `claude mcp list` (the server should report as connected).

### 4.3 Roo Code

Open **MCP Servers → Edit MCP Settings** to open `mcp_settings.json`, then add
under `mcpServers`:

```json
{
  "mcpServers": {
    "pinescript-docs": {
      "command": "C:\\GIT\\pinescript-docs-scraper\\.venv\\Scripts\\python.exe",
      "args": ["-m", "pinescript_mcp"],
      "env": {
        "PYTHONPATH": "C:\\GIT\\pinescript-docs-scraper",
        "SUPABASE_URL": "https://your-project.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "your-service-role-key",
        "OPENAI_API_KEY": "sk-your-openai-key"
      },
      "disabled": false,
      "alwaysAllow": [],
      "timeout": 60
    }
  }
}
```

`alwaysAllow` is intentionally empty: `get_context` can return thousands of
tokens, so approve it deliberately rather than auto-approving every call.

### 4.4 Cline

Open the MCP Servers panel → **Configure MCP Servers** to open
`cline_mcp_settings.json`, then add under `mcpServers`:

```json
{
  "mcpServers": {
    "pinescript-docs": {
      "command": "C:\\GIT\\pinescript-docs-scraper\\.venv\\Scripts\\python.exe",
      "args": ["-m", "pinescript_mcp"],
      "env": {
        "PYTHONPATH": "C:\\GIT\\pinescript-docs-scraper",
        "SUPABASE_URL": "https://your-project.supabase.co",
        "SUPABASE_SERVICE_ROLE_KEY": "your-service-role-key",
        "OPENAI_API_KEY": "sk-your-openai-key"
      },
      "disabled": false,
      "autoApprove": [],
      "timeout": 60
    }
  }
}
```

### 4.5 Continue.dev

Continue supports MCP servers with the same `mcpServers` shape in its config
(`config.yaml` / `config.json`). Because Continue is model-provider oriented,
the MCP entry is the supported route for this server; the REST `/chat` endpoint
is not OpenAI-API-compatible.

---

## 5. Tools

### `search_docs`

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `query` | string | — | Search query, natural language or code |
| `top_k` | int | 8 | 1–50 hits |
| `min_score` | float | -1.0 | Cosine floor; -1.0 means no floor |
| `source` | string? | null | Case-insensitive substring of the page filename |
| `include_code` | bool | false | Only chunks flagged as containing code |

Returns `[{doc_id, source, section, chunk_index, chunk_count, score, has_code, tokens, excerpt, truncated}]`.

### `get_context`

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `query` | string | — | Topic to assemble context for |
| `token_budget` | int | 8000 | 256–20000, additionally capped by `PROMPT_TOKEN_BUDGET` |
| `top_k` | int | 12 | Chunks considered before trimming |

Returns `{query, passages[], total_tokens, budget, documents_considered}` where
each passage carries the full text plus `doc_id`, `source`, `section`, `score`.

### `find_examples`

| Argument | Type | Default | Meaning |
|---|---|---|---|
| `topic` | string | — | What the example should demonstrate |
| `limit` | int | 5 | 1–20 examples |
| `min_score` | float | -1.0 | Cosine floor of the containing chunk |

Returns `[{doc_id, source, section, score, language, line_count, truncated, code}]`,
one code block per matching chunk, capped at 2000 characters.

---

## 6. Behaviour worth knowing

- **Scores are raw cosine similarity**, in `[-1, 1]`. Negative values are
  legitimate. `min_score` defaults to `-1.0` (no filtering) precisely so that a
  default cannot silently discard valid hits.
- **Filtering happens in the database.** `min_score`, `source` and `include_code`
  are passed to `match_documents_v2`, so the RPC returns at most `top_k` rows that
  already satisfy every criterion. This matters for correctness, not just speed: a
  query that filtered after `LIMIT` could return fewer hits than requested when the
  filter was selective. `code_snippet` is true for 89% of chunks, so
  `include_code` only trims a small tail — it is not a substitute for
  extracting actual code blocks.
- **`search_docs` returns excerpts, `get_context` returns full passages.** Call
  `get_context` only when the excerpt is insufficient: every returned token is
  one the caller cannot use for anything else.
- **Hybrid/BM25 search is deliberately not used.** `server.retriever.bm25_search`
  reads a `rank` field that the Supabase client never returns, so its scores are
  always 0 and the 0.3 BM25 weight contributes nothing. The tools use pure vector
  search until that is fixed.
- **The index covers the user manual only**, not the Language Reference. There are
  no per-function signatures in the corpus, so do not present a retrieved example
  as an authoritative signature.
- **stdio means stdout is the wire.** The entry point logs to stderr. Do not add
  `print()` calls to the server path.

### Measuring the filtering change

`scripts/benchmark_filtering.py` compares the two designs. Its payload figures use
real corpus text and a real pgvector encoding; the row counts come from the live
code path. `--live` measures both RPCs against a configured Supabase project.

---

## 7. Optional: Streamable HTTP

For a shared deployment, mount the ASGI app inside the existing FastAPI app so
both share one process. Two things are easy to get wrong: the host lifespan must
enter the session manager, and a deployed hostname must be allowlisted or every
request is rejected with `421 Misdirected Request`.

```python
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from starlette.routing import Mount

from pinescript_mcp.app import mcp
# ... existing FastAPI app and its lifespan ...

@asynccontextmanager
async def lifespan(app) -> AsyncIterator[None]:
    async with mcp.session_manager.run():   # required: a mounted app's own lifespan never runs
        yield

app.router.routes.append(Mount("/mcp-root", app=mcp.streamable_http_app(
    transport_security=TransportSecuritySettings(
        allowed_hosts=["mcp.example.com", "mcp.example.com:*"],
        allowed_origins=["https://app.example.com"],
    ),
)))
```

Remote deployments should be authenticated. The current REST login endpoint is
not suitable for public use (see [SECURITY-REVIEW.md](../SECURITY-REVIEW.md),
findings SEC-01 and SEC-02); stdio needs no authentication at all, which is why
it is the recommended starting point.
