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

- Apply migrations: the repo contains SQL migrations in `migrations/` — apply them to your Supabase project.

- Example (Supabase SQL editor): copy/paste `migrations/0001_create_documents_and_file_manifest.sql` and run it.

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
