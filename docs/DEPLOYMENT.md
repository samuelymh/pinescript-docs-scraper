# Deployment Guide

This document outlines recommended steps to deploy the PineScript RAG Server in production.

**Audience:** operators and maintainers. This file is the canonical production/deployment guide. For local developer instructions and quick run commands, see `server/STARTUP.md`.

## Required environment variables

Set these as secrets in your hosting provider (Render, Fly, Docker, etc.):

- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key (server-side only)
- `OPENAI_API_KEY` - OpenAI API key (server-side only)
- `ADMIN_API_KEY` - Admin key for internal endpoints (optional but recommended)
- `JWT_SECRET` - HMAC secret for HS256 JWT verification (or configure JWKS_URL)
- `JWT_ALGORITHM` - e.g. `HS256` or `RS256` (defaults to `HS256`)
- `JWKS_URL` - (optional) URL to fetch JWKS for RS256 verification
- `JWT_AUDIENCE` - (optional) expected JWT audience
- `JWT_ISSUER` - (optional) expected JWT issuer
- `RAG_VECTOR_TABLE` - Supabase table name for documents (default: `documents`)

Keep secrets out of client code and never expose service role keys to browsers.

## Docker (local) quickstart

Build the image (project root):

```bash
docker build -t pinescript-rag-server:latest .
```

Run with environment variables (example):

```bash
docker run -p 8000:8000 \
  -e SUPABASE_URL="$SUPABASE_URL" \
  -e SUPABASE_SERVICE_ROLE_KEY="$SUPABASE_SERVICE_ROLE_KEY" \
  -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  -e JWT_SECRET="$JWT_SECRET" \
  -e ADMIN_API_KEY="$ADMIN_API_KEY" \
  pinescript-rag-server:latest
```

Healthcheck endpoint available at `GET /status`.

## Recommended production runtime

For simple deployments (Render / Fly / Docker): use Gunicorn with Uvicorn workers. The container image ships a startup command that runs a small number of workers and honors `--env` environment variables.

If using a process manager (systemd, k8s), prefer multiple worker processes and a readiness/liveness probe that calls `/status`.

## Running migrations and indexing

The repository contains two migrations. **Apply them in this order.**

| Migration | Provides | Required by |
|---|---|---|
| `migrations/0001_create_documents_and_file_manifest.sql` | `documents`, `file_manifest`, `match_documents`, HNSW index | Index ingestion and `POST /chat` |
| `migrations/0002_match_documents_v2.sql` | `match_documents_v2` (filtered vector search) | **The MCP tools** |

Example (Supabase SQL editor): copy/paste each file and run it, `0001` first. With `psql`:

```bash
psql "$DATABASE_URL" -f migrations/0001_create_documents_and_file_manifest.sql
psql "$DATABASE_URL" -f migrations/0002_match_documents_v2.sql
```

> **Deploying the MCP server? Migration 0002 is mandatory and must be applied before the
> code.** See [MCP server deployment](#mcp-server-deployment).

- Administrative indexing options:

  - Background (non-blocking from the API):

    ```bash
    curl -sS -X POST "http://localhost:8000/internal/index?background=true" -H "X-Admin-Key: $ADMIN_API_KEY" | jq .
    ```

  - Synchronous (one-off, recommended for full reindexes): run the ingest CLI inside the image to avoid request worker timeouts:

    ```bash
    # run inside a running container
    docker exec -it <container> python /app/server/run_ingest.py --full --log-level INFO

    # or as a one-off container
    docker run --rm \
      --env-file .env \
      -v "$(pwd)/pinescript_docs/processed:/app/pinescript_docs/processed:ro" \
      pinescript-rag-server:latest \
      python /app/server/run_ingest.py --full --log-level INFO
    ```

  - Note: the API also supports `POST /internal/index` with `?background=false` but long-running synchronous requests may be killed by Gunicorn unless you increase the worker `--timeout`.

## MCP server deployment

The MCP server is a separate process from the FastAPI application: it speaks the Model
Context Protocol over stdio, launched by the client (`python -m pinescript_mcp`). See
`docs/MCP.md` for client configuration. This section covers the part that is specific to
a production deployment: the migration prerequisite, verification and rollback.

### Prerequisite: apply migration 0002 first

**`migrations/0002_match_documents_v2.sql` MUST be applied to the Supabase project
before the MCP code is deployed.** The MCP tools delegate all filtering to that
function and have no application-side fallback.

Deploy the code before the migration and **all three MCP tools fail**:

| Tool | Behaviour without migration 0002 |
|---|---|
| `search_docs` | Fails with an error result naming the missing migration |
| `get_context` | Fails identically |
| `find_examples` | Fails identically |

The failure is explicit rather than silent — a `MigrationRequiredError` after a single
attempt, with the migration filename in the message — but it is a failure, not a
degraded mode. Nothing partial works.

Order of operations:

1. Apply `migrations/0001_create_documents_and_file_manifest.sql` if not already applied.
2. Apply `migrations/0002_match_documents_v2.sql`.
3. Verify with the four checks below.
4. Install the MCP dependencies: `python -m pip install -r requirements-mcp.txt`.
5. Deploy or restart the MCP server.
6. Configure the client per `docs/MCP.md`.

Steps 3 and 4 are not optional. Without step 4 the server cannot start at all, and
without step 3 you find out about a problem from a user instead of from a query.

### Verifying the deployment

`match_documents_v2` requires a query embedding, so the checks below use an embedding
already stored in the table as the query vector. The chunk it came from must come back
as the top hit with a similarity near 1, which makes these checks self-validating: if
the function returns nothing at all, something is structurally wrong.

**Check 1 — the function exists and returns rows.**

```sql
SELECT m.id, m.source_filename, m.code_snippet, round(m.similarity::numeric, 4) AS similarity
FROM match_documents_v2(
       query_embedding  => (SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1),
       match_threshold  => -1.0,
       match_count      => 5,
       filter_source    => NULL,
       filter_code_only => false
     ) AS m;
```

Expected: up to 5 rows, the top one with a similarity close to 1.

- `ERROR: function public.match_documents_v2(...) does not exist` → the migration was not applied.
- `permission denied for function match_documents_v2` → the client is not connecting as
  `service_role`. See the note on permissions below.

**Check 2 — the source filter.**

The corpus filenames all begin with `processed_`, so a filter on that substring must
match everything and therefore return the same count as no filter:

```sql
SELECT
  (SELECT count(*) FROM match_documents_v2(
     query_embedding  => (SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1),
     match_threshold  => -1.0,
     match_count      => 20,
     filter_source    => NULL,
     filter_code_only => false))            AS unfiltered,
  (SELECT count(*) FROM match_documents_v2(
     query_embedding  => (SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1),
     match_threshold  => -1.0,
     match_count      => 20,
     filter_source    => 'processed_',
     filter_code_only => false))            AS all_match;
```

Expected: `all_match` = `unfiltered`. A smaller number means the filter is excluding
rows it should keep.

Then confirm it discriminates, using a selective substring:

```sql
SELECT count(*) AS total,
       count(*) FILTER (
         WHERE strpos(lower(source_filename), 'strategies') > 0
       ) AS matching
FROM match_documents_v2(
       query_embedding  => (SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1),
       match_threshold  => -1.0,
       match_count      => 20,
       filter_source    => 'strategies',
       filter_code_only => false
     );
```

Expected: `total` = `matching`. (`SELECT source_filename FROM documents LIMIT 5` shows
real filenames if you want a different substring.)

> Caveat: filters are applied after the HNSW index walk, so a very selective filter can
> return fewer rows — or none — even though matching chunks exist further from the query
> vector. That is expected behaviour, not a broken filter, and it is why the assertion
> above is "every returned row matches" rather than "at least one row is returned".

**Check 3 — the code-only filter.**

```sql
SELECT count(*) AS total,
       count(*) FILTER (WHERE code_snippet IS TRUE) AS with_code
FROM match_documents_v2(
       query_embedding  => (SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1),
       match_threshold  => -1.0,
       match_count      => 20,
       filter_source    => NULL,
       filter_code_only => true
     );
```

Expected: `total` = `with_code`. Note that `code_snippet` is true for roughly 89% of the
current corpus, so this filter removes only a small tail: a `total` close to the
unfiltered count is correct, not a failure.

**Check 4 — the threshold is actually applied.**

```sql
SELECT
  (SELECT count(*) FROM match_documents_v2(
     query_embedding  => (SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1),
     match_threshold  => -1.0,
     match_count      => 20,
     filter_source    => NULL,
     filter_code_only => false))  AS no_floor,
  (SELECT count(*) FROM match_documents_v2(
     query_embedding  => (SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1),
     match_threshold  => 0.99,
     match_count      => 20,
     filter_source    => NULL,
     filter_code_only => false))  AS strict_floor;
```

Expected: `strict_floor` strictly less than `no_floor`, typically 1 (only the row whose
own embedding was used as the query).

**If the two counts are equal, the threshold is not being applied.** That is precisely
the defect migration 0002 exists to fix, and it means an older version of the function
is installed. `match_threshold` is a *similarity* floor (higher is better), not a
distance.

**Optional — confirm the vector index is used.**

```sql
EXPLAIN ANALYZE
SELECT * FROM match_documents_v2(
  query_embedding  => (SELECT embedding FROM documents WHERE embedding IS NOT NULL LIMIT 1),
  match_threshold  => -1.0,
  match_count      => 10,
  filter_source    => NULL,
  filter_code_only => false
);
```

Look for an index scan on `idx_documents_embedding`. On a small corpus PostgreSQL may
prefer a sequential scan, which is correct behaviour rather than a problem; it becomes
relevant past roughly 100k chunks.

**Note on permissions.** Migration 0002 revokes execute on the new function from
`PUBLIC` and grants it to `service_role` only, so it is not publicly callable with the
anon key. The application authenticates with `SUPABASE_SERVICE_ROLE_KEY` and therefore
works. If your deployment calls this RPC under a different role, grant it explicitly:

```sql
-- Replace my_api_role with the role your PostgREST requests actually run as.
GRANT EXECUTE ON FUNCTION public.match_documents_v2(
  vector, double precision, integer, text, boolean
) TO my_api_role;
```

### Rollback

Rolling back is one statement and loses no data. It removes only the MCP retrieval path:
`documents`, `file_manifest`, the `match_documents` function and the HNSW index are all
untouched.

```sql
DROP FUNCTION IF EXISTS public.match_documents_v2(
  vector, double precision, integer, text, boolean
);
```

Then redeploy the previous MCP server version, or stop the MCP clients. If you leave the
new code running without the function, the tools fail with an explicit
`MigrationRequiredError` instead of returning wrong results — an acceptable intermediate
state if you need to disable the feature immediately.

Re-applying the migration restores the function. It is written with `CREATE OR REPLACE`,
so running it again is safe and does not modify existing data.

To roll back the entire MCP feature, also revert the code that introduced
`pinescript_mcp/` and `search_documents_v2`. The FastAPI application, `POST /chat` and
every endpoint listed above are unaffected by all of the above.

## Monitoring and logging

- The server logs to stdout/stderr and honors `LOG_LEVEL` (or `log_level` in config). Configure your host to capture logs.
- Health checks: `GET /status` returns `documents_count` and `last_index_time` (if Supabase is configured).

## Secrets and security

- Use HS256 (`JWT_SECRET`) or RS256 (`JWKS_URL`) for verifying JWTs. Prefer RS256 in distributed deployments with centralized auth.
- Do not expose Supabase service role key to clients. Keep it server-side only.

## Example `.env` (minimal)

Create a `.env` file at the repo root with the following values for local testing:

```
SUPABASE_URL=https://your.supabase.url
SUPABASE_SERVICE_ROLE_KEY=service-role-key
OPENAI_API_KEY=sk-xxx
ADMIN_API_KEY=change-me
JWT_SECRET=test-secret
```

After modifying the `Dockerfile`, rebuild the image to pick up `GUNICORN_CMD_ARGS` defaults:

```bash
docker build -t pinescript-rag-server:latest .
```

Then run with mounts or env overrides as needed (see `server/STARTUP.md` for examples).
