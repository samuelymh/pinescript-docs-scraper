# Security Review — PineScript RAG Server

| | |
|---|---|
| **Repository** | `samuelymh/pinescript-docs-scraper` (fork of `OdynBrouwer/pinescript-docs-scraper`) |
| **Revision reviewed** | `50c3229` (branch `main`, 15 commits ahead of `origin/main`) |
| **Scope** | `server/` (FastAPI app, auth, ingest, retrieval, clients), `migrations/`, `scripts/`, `Dockerfile`, `requirements.txt`, `.github/workflows/` |
| **Review type** | Static review plus local verification (test suite executed, chunking pipeline executed against the real corpus) |
| **Findings** | 2 critical, 6 high, 8 medium, 4 low |

## Method and verification

The review combined manual code reading with executable verification:

- Full test suite executed twice — once without environment configuration (**29/78 failures**) and once with dummy credentials (**8/78 failures**, of which 2 require live Supabase). One failure reproduces a real runtime bug (SEC-20 in the related-findings section).
- The ingestion/chunking pipeline was executed against the real corpus: 73 processed files, 3.19 MB, 884,619 tokens, 594 chunks after heading-based chunking, 18 chunks above the embedding-model limit (max 17,832 tokens).
- The repository was scanned for committed credentials (`sk-…`, `eyJ…`, private keys, service-role keys) — **no secrets found** in tracked files, including the CI logs committed under `log/`.

This document covers security findings. Integrity and availability findings that would block a production deployment are listed separately in [Related production findings](#related-production-findings).

---

## Summary

| Severity | Count |
|---|---|
| 🔴 CRITICAL | 2 |
| 🟠 HIGH | 6 |
| 🟡 MEDIUM | 8 |
| ⚪ LOW | 4 |

## Findings

| # | Severity | File | Lines | Finding | Confidence |
|---|----------|------|-------|---------|------------|
| SEC-01 | 🔴 CRITICAL | migrations/0001_create_documents_and_file_manifest.sql | 1-122 | Row Level Security is never enabled, so `documents`, `file_manifest` and the `match_documents` RPC are exposed to the public `anon` role for read **and write** | 8/10 |
| SEC-02 | 🔴 CRITICAL | server/app.py | 86-144 | `POST /api/login` issues a full JWT based on an email address alone — no password, no email verification, no rate limit | 9/10 |
| SEC-03 | 🟠 HIGH | server/app.py | 60-66 | `allow_origins=["*"]` combined with `allow_credentials=True` | 9/10 |
| SEC-04 | 🟠 HIGH | server/app.py | 147-176 | Unauthenticated `/status` discloses LLM/embedding model configuration and index statistics, and triggers an exact-count DB scan | 8/10 |
| SEC-05 | 🟠 HIGH | server/app.py | 244-330 | No `typ` claim distinguishes access from refresh tokens; access tokens are accepted at `/api/refresh`, and refresh tokens cannot be revoked | 8/10 |
| SEC-06 | 🟠 HIGH | server/app.py | 332-380 | `POST /internal/index?full=true` wipes the entire index with no confirmation, scoping or audit trail | 8/10 |
| SEC-07 | 🟠 HIGH | server/embed_client.py, server/llm_client.py, server/supabase_client.py | 44-49, 118-130, 247-318 | No request timeouts on OpenAI or Supabase clients; retries fire on non-retryable errors | 7/10 |
| SEC-08 | 🟠 HIGH | server/app.py | 240, 328 | Internal exception text returned to clients via `detail=str(e)` | 9/10 |
| SEC-09 | 🟡 MEDIUM | server/app.py, server/config.py | 123, 297 / 38-56 | `jwt_expiration_seconds` is not a configuration field, so the environment variable is silently ignored and tokens always live 3600 s | 9/10 |
| SEC-10 | 🟡 MEDIUM | server/auth.py | 64-79 | Admin key compared with `!=` (not constant-time); no rotation, key id or audit trail | 7/10 |
| SEC-11 | 🟡 MEDIUM | server/app.py | 179-242 | No per-user cost quota — the only limit is 10 requests/minute against a ~18k-token prompt | 7/10 |
| SEC-12 | 🟡 MEDIUM | server/auth.py | 32-58 | The email allowlist is the only authorization layer; there are no roles, so one leaked inbox grants full API access | 8/10 |
| SEC-13 | 🟡 MEDIUM | server/llm_client.py | 115-131 | `conversation_history` roles are not validated, so a client can inject a `system` message and override the system prompt | 7/10 |
| SEC-14 | 🟡 MEDIUM | server/llm_client.py | 133-140 | Retrieved context and the user query are concatenated into a single user message with no delimiters and no grounding instruction | 7/10 |
| SEC-15 | 🟡 MEDIUM | server/app.py, server/auth.py | 55-58, 147, 179, 332 / 179-243 | Rate limiting is per-worker in-memory, missing on `/api/login` and `/api/refresh`, and does not trust proxy headers | 7/10 |
| SEC-16 | 🟡 MEDIUM | server/app.py | 318-330 | Dead, unreachable exception handler retains a copy of the error-leaking pattern and obscures the control flow | 9/10 |
| SEC-17 | ⚪ LOW | server/auth.py | 82-97, 99-177 | JWKS fetch performs blocking `requests.get()` inside async request paths; no refresh on unknown `kid`, no locking, no negative caching | 6/10 |
| SEC-18 | ⚪ LOW | Dockerfile | 1-40 | Container runs as root, `.git` is copied into the image, and the HEALTHCHECK cannot succeed | 8/10 |
| SEC-19 | ⚪ LOW | requirements.txt | 1-30 | Unpinned `>=` ranges, no lock file, and the deprecated `asyncio` backport listed as a server dependency | 7/10 |
| SEC-20 | ⚪ LOW | log/ | — | CI logs (13 text files plus `logs_54775165524.zip`) are committed; verified free of secret patterns in this revision, but they remain a recurring leak vector and repository bloat | 6/10 |

---

## Detailed findings

### SEC-01 — Row Level Security is not enabled (🔴 CRITICAL)

**Impact.** Supabase exposes every table in the `public` schema through PostgREST and grants privileges to the `anon` and `authenticated` roles by default. The migration creates `documents` and `file_manifest` without `ALTER TABLE … ENABLE ROW LEVEL SECURITY`, and `match_documents` is a `SECURITY INVOKER` function executable by `anon`. Anyone holding the project's public anon key — which is designed to be embedded in client applications — can read, insert, update and delete the entire document index, and call the vector-search RPC directly.

**Evidence.** [migrations/0001_create_documents_and_file_manifest.sql](migrations/0001_create_documents_and_file_manifest.sql) contains no `ENABLE ROW LEVEL SECURITY` statement and no `REVOKE EXECUTE ON FUNCTION`. The schema document [SUPABASE_SCHEMA.md](SUPABASE_SCHEMA.md) documents the RLS policies that *should* exist, but they were never added to the migration.

**Remediation.**

```sql
ALTER TABLE public.documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.file_manifest ENABLE ROW LEVEL SECURITY;

-- The API connects with the service role, which bypasses RLS.
CREATE POLICY service_role_all ON public.documents
    FOR ALL TO service_role USING (true) WITH CHECK (true);
CREATE POLICY service_role_all ON public.file_manifest
    FOR ALL TO service_role USING (true) WITH CHECK (true);

-- No anonymous access at all unless explicitly required.
REVOKE ALL ON public.documents, public.file_manifest FROM anon, authenticated;
REVOKE EXECUTE ON FUNCTION public.match_documents(vector, double precision, integer) FROM anon, authenticated;
```

Add this as a new migration (`0002_enable_rls.sql`) rather than editing `0001`, then verify with the anon key that `select` and `delete` both fail.

---

### SEC-02 — Password-less identity issuance (🔴 CRITICAL)

**Impact.** `POST /api/login` accepts an email address, checks it against `ALLOWED_EMAILS`, and returns an access and refresh token. There is no password, no email ownership verification, no one-time code and no rate limit. Anyone who knows or guesses an allowlisted address (corporate domains, published contributor emails, support addresses) obtains a valid identity and full chat access. This converts the allowlist from an access control into a hint.

**Evidence.** [server/app.py:86-144](server/app.py:86) — the handler validates the email format with a regex, calls `is_email_allowed()`, and signs a token with no further proof of identity. It is the only endpoint without a rate-limit decorator aside from `/api/refresh`.

**Remediation.** Replace the endpoint with a real identity provider (Supabase Auth, Entra ID, Auth0, Google) and verify the token issuer via the existing JWKS path. If a first-party flow is required, issue a single-use magic link or OTP that proves mailbox control, add a per-IP and per-email rate limit, and record login attempts. Until then, treat this endpoint as a public credential dispenser and disable it in production.

---

### SEC-03 — Wildcard CORS with credentials (🟠 HIGH)

**Impact.** `allow_origins=["*"]` together with `allow_credentials=True` is rejected by browsers and, where it is tolerated, allows any origin to make credentialed requests. Combined with SEC-02, any visited page could silently obtain tokens on a user's behalf.

**Evidence.** [server/app.py:60-66](server/app.py:60) — the block still carries the comment `# TODO: Configure for production`.

**Remediation.** Enumerate allowed origins from configuration, for example `allow_origins=config.cors_origins`, and set `allow_credentials=False` unless cookie-based auth is genuinely required. Add the origins to `config.py` and `config.example.env`.

---

### SEC-04 — Unauthenticated status endpoint discloses configuration (🟠 HIGH)

**Impact.** `/status` is reachable without authentication and returns the configured LLM and embedding models, context and chunking parameters, and the indexed document count and last index time. It also performs `count="exact"` on `documents` and an ordered query on `file_manifest` on every call, making it a cheap amplification vector against the database (30 requests/minute per key).

**Evidence.** [server/app.py:147-176](server/app.py:147) and [server/supabase_client.py:206-244](server/supabase_client.py:206).

**Remediation.** Split liveness from readiness: keep `/health` public and trivial, and require authentication for `/status`. Reduce the public payload to `{"status": "ok"}`. Switch to `count="planned"` or maintain a rolling counter instead of an exact count.

---

### SEC-05 — Access and refresh tokens are interchangeable (🟠 HIGH)

**Impact.** Both tokens are signed by the same key with the same claims and no `typ` field. An access token is therefore accepted by `/api/refresh` and can be exchanged indefinitely for new tokens, so a single leaked short-lived token yields permanent access. Refresh tokens cannot be revoked: there is no `jti`, no server-side rotation record and no reuse detection. Additionally, if `JWT_ALGORITHM=RS256` is configured without `JWT_SECRET`, `/api/refresh` raises an unhandled error and returns 500.

**Evidence.** [server/app.py:244-330](server/app.py:244) — the refresh payload is built identically to the access payload, and the only subject check is the email allowlist.

**Remediation.** Add a `typ` claim (`access` / `refresh`), reject mismatch at both verification points, store refresh token identifiers server-side with rotation and reuse detection, and shorten the refresh lifetime. Fail fast at startup when `jwt_algorithm` and `jwt_secret` are inconsistent.

---

### SEC-06 — Destructive admin operation without safeguards (🟠 HIGH)

**Impact.** A single authenticated call — `POST /internal/index?full=true` — invokes `clear_all_data()`, which deletes every row from `file_manifest` and `documents`. There is no confirmation parameter, no dry-run, no scoped target and no audit record. Combined with `background=true`, the deletion proceeds with no request-level feedback, and concurrent invocations can interleave deletes and upserts because there is no indexing lock.

**Evidence.** [server/app.py:332-380](server/app.py:332) and [server/supabase_client.py:321-355](server/supabase_client.py:321).

**Remediation.** Require an explicit confirmation flag for destructive runs, add a PostgreSQL advisory lock around the pipeline to serialize indexing, emit an audit log entry with the caller identity and before/after counts, and prefer rebuild-into-new-identifier followed by a swap over delete-then-rebuild.

---

### SEC-07 — No request timeouts, retries on non-retryable errors (🟠 HIGH)

**Impact.** Neither the OpenAI nor the Supabase client is constructed with a timeout, so a stalled upstream holds a worker until the Gunicorn timeout kills it. On a four-worker deployment this is a straightforward denial of service. The embedding retry policy is `retry_if_exception_type((Exception,))`, so authentication failures and malformed requests are retried three times with exponential backoff, delaying and obscuring the real error.

**Evidence.** [server/embed_client.py:44-49](server/embed_client.py:44), [server/llm_client.py:118-130](server/llm_client.py:118), [server/supabase_client.py:247-318](server/supabase_client.py:247).

**Remediation.** Pass explicit `timeout=` values to the OpenAI and Supabase clients, restrict `retry_if_exception_type` to `openai.APIConnectionError`, `openai.RateLimitError` and `openai.InternalServerError`, and add a circuit breaker plus a per-request deadline.

---

### SEC-08 — Internal error details leaked to clients (🟠 HIGH)

**Impact.** Both the chat and refresh handlers convert any exception into `HTTPException(status_code=500, detail=str(e))`, exposing internal messages — SDK errors, database errors, and potentially upstream response bodies — to the caller.

**Evidence.** [server/app.py:240](server/app.py:240) and [server/app.py:328](server/app.py:328).

**Remediation.** Log the exception with a correlation id and return a generic message plus that id. Add a global exception handler so future endpoints cannot reintroduce the pattern.

---

### SEC-09 — Token lifetime configuration silently ignored (🟡 MEDIUM)

**Impact.** `jwt_expiration_seconds` does not exist on the `Config` model, although `jwt_refresh_expiration_seconds` does. Both the login and refresh handlers read it through `getattr(config, "jwt_expiration_seconds", 3600)`, so the fallback is always used and an operator setting `JWT_EXPIRATION_SECONDS` — including to a *shorter* value for a security reason — sees no effect.

**Evidence.** [server/app.py:123](server/app.py:123), [server/app.py:297](server/app.py:297), [server/config.py:38-56](server/config.py:38).

**Remediation.** Add the field to `Config` with an explicit default, remove the `getattr` fallbacks, and assert at startup that the access lifetime is strictly shorter than the refresh lifetime.

---

### SEC-10 — Admin key comparison and lifecycle (🟡 MEDIUM)

**Impact.** The admin key is compared with `!=` and validated as a plain header value, so comparison timing leaks information about the expected value. The key has no identifier, no expiry and no rotation path, and successful and failed admin calls are logged only as unstructured messages.

**Evidence.** [server/auth.py:64-79](server/auth.py:64).

**Remediation.** Use `secrets.compare_digest`, support a comma-separated set of accepted keys with ids to allow rotation without downtime, and log every admin operation with the key id, caller address and outcome.

---

### SEC-11 — No cost ceiling per user (🟡 MEDIUM)

**Impact.** Each chat request embeds the query and sends up to 16,000 tokens of context to GPT-4o. At roughly USD 0.06 per request, the 10/minute limit permits on the order of USD 900 per day per authenticated user — and SEC-02 makes obtaining such a user straightforward.

**Evidence.** [server/app.py:179-242](server/app.py:179), `prompt_token_budget` in [server/config.py](server/config.py).

**Remediation.** Enforce a per-user daily token or cost budget, record token usage per request in a durable store, alert on anomalies, and bound `conversation_history` length and item size (currently unbounded).

---

### SEC-12 — Single-layer authorization (🟡 MEDIUM)

**Impact.** Authorization consists solely of "is this email in `ALLOWED_EMAILS`". There is no distinction between administrative, read-only and interactive use, and every allowed identity can trigger reindexing and deletion.

**Evidence.** [server/auth.py:32-58](server/auth.py:32) and the reuse of the same allowlist for `/chat` and, indirectly, for administrative intent.

**Remediation.** Introduce roles (for example `admin`, `user`) as a claim or server-side mapping, and gate `/internal/*` on an administrative role instead of a shared static key.

---

### SEC-13 — Conversation history roles are not validated (🟡 MEDIUM)

**Impact.** Client-supplied history entries are forwarded with their `role` intact, so a caller can insert `{"role": "system", "content": "…"}` and override the system prompt for their own session.

**Evidence.** [server/llm_client.py:115-131](server/llm_client.py:115).

**Remediation.** Restrict accepted roles to `user` and `assistant` at the request model, and reject or coerce anything else.

---

### SEC-14 — Prompt construction is injection-friendly (🟡 MEDIUM)

**Impact.** Retrieved passages and the user query are joined into one message, and the system prompt does not state that the context is untrusted reference material. Because retrieved text is derived from crawled web content that is re-scraped automatically every three days, hostile or corrupted content would be treated as instructions.

**Evidence.** [server/llm_client.py:133-140](server/llm_client.py:133).

**Remediation.** Delimit context in explicit markers, instruct the model to treat retrieved content as data and never as instructions, require answers to be grounded in the provided sources, and keep the user query in its own message.

---

### SEC-15 — Rate limiting is not enforceable in the deployed topology (🟡 MEDIUM)

**Impact.** SlowAPI is configured with its default in-memory storage, so limits are per worker process — a four-worker container permits four times the configured rate, and horizontal scaling multiplies it further. `/api/login` and `/api/refresh` have no limit at all. `get_remote_address` does not consult proxy headers, and the container command does not enable proxy header handling, so behind a load balancer all clients can share a single bucket.

**Evidence.** [server/app.py:55-58](server/app.py:55), limits at [server/app.py:147](server/app.py:147), [:179](server/app.py:179), [:332](server/app.py:332), and [server/auth.py:179-243](server/auth.py:179).

**Remediation.** Move limiter storage to Redis, add limits to the authentication endpoints, enable trusted proxy headers explicitly, and document the expected client address behaviour for the deployment platform.

---

### SEC-16 — Unreachable duplicate exception handler (🟡 MEDIUM)

**Impact.** The `/api/refresh` handler ends with a second `except HTTPException` / `except Exception` block that can never execute because it follows `return` inside the same `try`. It also carries a stale copy of the error-leaking pattern, so a future edit that moves code around can resurrect it.

**Evidence.** [server/app.py:318-330](server/app.py:318).

**Remediation.** Delete lines 318-330 and cover `/api/refresh` with tests for the error paths.

---

### SEC-17 — JWKS handling in async paths (⚪ LOW)

**Impact.** `_fetch_jwks` uses blocking `requests.get(..., timeout=5)` and is called from `async def verify_jwt_token` and from the synchronous rate-limit key function, blocking the event loop on cache misses. An unknown `kid` is rejected for up to the one-hour TTL because the cache is never refreshed on a miss; there is no lock around concurrent refreshes and no negative caching.

**Evidence.** [server/auth.py:82-97](server/auth.py:82), [server/auth.py:99-177](server/auth.py:99).

**Remediation.** Use `httpx.AsyncClient` with a timeout, force a cache refresh when a `kid` is unknown (with rate limiting on refreshes), guard the cache with a lock, and add an asyncio-based TTL cache.

---

### SEC-18 — Container hardening and health check (⚪ LOW)

**Impact.** The image runs as root, `COPY . /app` includes `.git` (full history and remotes), and the HEALTHCHECK invokes `curl`, which is absent from `python:3.11-slim`, so every container is reported unhealthy after the start period. The health check also targets `/status`, which depends on Supabase availability, conflating liveness with dependency health.

**Evidence.** [Dockerfile](Dockerfile).

**Remediation.** Add a non-root `USER`, exclude `.git` and documentation artefacts in `.dockerignore`, point the HEALTHCHECK at `/health`, and either install a health binary or use a Python one-liner instead of `curl`. Publish the scraper dependencies as a separate image from the server.

---

### SEC-19 — Dependency hygiene (⚪ LOW)

**Impact.** Every dependency uses a `>=` range with no lock file, so builds are not reproducible and a compromised or breaking upstream release is pulled automatically. The `asyncio` backport is listed as a server dependency although it is documented as unnecessary and unsupported on modern Python. `openai>=1.3.0` is resolved to 2.x locally while the code passes the deprecated `max_tokens` parameter.

**Evidence.** [requirements.txt](requirements.txt) and [server/llm_client.py:126](server/llm_client.py:126).

**Remediation.** Split scraper and server dependencies, pin versions with a lock file produced by `uv` or `pip-tools`, remove `asyncio`, and migrate to `max_completion_tokens`. Add a dependency scanning step to CI.

---

### SEC-20 — Build logs committed to the repository (⚪ LOW)

**Impact.** Thirteen CI log files and one log archive are tracked under `log/`. They were scanned during this review and contain no credential patterns at this revision, but CI logs routinely capture environment dumps and command output, and `log/6_Debug info.txt` explicitly gathers debug output. Committed logs also make the repository heavy and make secrets hard to purge later.

**Evidence.** `log/0_generate.txt`, `log/generate/*.txt`, `log/logs_54775165524.zip`.

**Remediation.** Remove the directory from version control, add it to `.gitignore`, and ensure any CI step that dumps environment information redacts it.

---

## Related production findings

The following issues are outside the security scope but were confirmed while reviewing. They affect data integrity and availability and would block a production deployment. Details are in the accompanying architecture and readiness audit.

| # | Severity | Area | Finding |
|---|---|---|---|
| PROD-01 | 🔴 CRITICAL | Index integrity | Processed filenames embed a fresh timestamp on every scrape run, so the manifest diff classifies all files as *new*; files that disappear are never deleted, so orphaned duplicate chunks accumulate in `documents` on every three-day cycle ([server/ingest.py:344-350](server/ingest.py:344), [server/ingest.py:412-423](server/ingest.py:412)) |
| PROD-02 | 🟠 HIGH | Retrieval | Hybrid search is a no-op: Supabase `text_search` returns no `rank`, so every BM25 score is 0 and the 0.3 weight contributes nothing while zero-score documents can still be injected into the context ([server/retriever.py:105-118](server/retriever.py:105), [server/retriever.py:204-216](server/retriever.py:204)) |
| PROD-03 | 🟠 HIGH | Retrieval | `match_threshold` is declared but never used in `match_documents`; there is no relevance floor at all, and the client passes `1 - similarity_threshold` as if it were a distance ([migrations/0001…sql:72](migrations/0001_create_documents_and_file_manifest.sql:72), [server/supabase_client.py:276](server/supabase_client.py:276)) |
| PROD-04 | 🟠 HIGH | Availability | Indexing runs synchronously on the event loop (`index_documents` is async but contains no `await`), and `background=true` schedules it with `asyncio.create_task` without retaining a reference — the worker is blocked and the task can be garbage collected ([server/app.py:360](server/app.py:360), [server/ingest.py:361](server/ingest.py:361)) |
| PROD-05 | 🟠 HIGH | Scalability | All chunks are upserted in a single PostgREST request, and `fetch_manifest` has no pagination, so the manifest is silently truncated by the PostgREST row cap ([server/supabase_client.py:80](server/supabase_client.py:80), [server/supabase_client.py:154](server/supabase_client.py:154)) |
| PROD-06 | 🟠 HIGH | Correctness | A `logger` assignment inside `create_chat_completion` shadows the module-level logger, so the `except` branch raises `UnboundLocalError` and masks every LLM failure ([server/llm_client.py:79](server/llm_client.py:79), [server/llm_client.py:152](server/llm_client.py:152)) |
| PROD-07 | 🟡 MEDIUM | Data quality | The processor appends every code block a second time under `## Code Examples`, and all markdown links (including source URLs) are stripped, destroying provenance ([2_process_docs.py:44-52](2_process_docs.py:44), [2_process_docs.py:85-100](2_process_docs.py:85)) |
| PROD-08 | 🟡 MEDIUM | Testability | The test suite is not hermetic (29/78 failures without a `.env`), contains stale expectations, and there is no server CI workflow — the existing workflow only re-scrapes documentation and pushes to `main` |
| PROD-09 | ⚪ LOW | Repository state | An unfinished merge with 70 unresolved paths is present in the working tree, and tracked artefacts include `__pycache__/*.pyc`, a duplicated `processed_all_docs.md` index entry, a log archive, and two full generations of the document corpus |

---

## Posture assessment

The codebase shows genuine security intent: deny-by-default allowlisting, an admin key gate on internal endpoints, the Supabase service role kept server-side, algorithm pinning during JWT decoding, `.env` excluded from version control, and no credentials in tracked files. Passwords are absent by design, which is defensible — but the design stops at proving that an email address is *known* rather than *owned*.

Two findings are deployment blockers. SEC-01 exposes the database to any holder of the public anon key, and SEC-02 allows anyone who knows an allowlisted address to obtain a valid identity. Both are small, well-understood changes. The remaining high-severity findings are hardening work: strict CORS, authenticated status, token separation with revocation, destructive-operation safeguards, timeouts, and error-message hygiene.

**Security score: 4/10.** The structure is sound; the defaults are not.

## Recommended remediation order

1. **SEC-01** — enable RLS and revoke anonymous access (migration `0002`), then verify with the anon key.
2. **PROD-01** — fix index identity and deletion detection; without this the data the server protects degrades continuously.
3. **SEC-02** — replace or disable `/api/login`; move to a real identity provider.
4. **SEC-05** — add `typ`, rotation and revocation for refresh tokens.
5. **SEC-08, SEC-16** — remove error leakage and the dead handler.
6. **SEC-03, SEC-04, SEC-15** — lock down CORS, the status endpoint, and rate limiting.
7. **SEC-06, SEC-07** — safeguards on destructive operations and timeouts on all upstream clients.
8. **SEC-09 through SEC-14, SEC-17 through SEC-20** — configuration correctness, roles, token accounting, prompt hardening, JWKS handling, container hardening, dependency pinning, repository hygiene.

## Verification checklist

```sql
-- RLS must be enabled on both tables
SELECT tablename, rowsecurity FROM pg_tables
WHERE schemaname = 'public' AND tablename IN ('documents', 'file_manifest');

-- The RPC must not be executable by anonymous roles
SELECT grantee, privilege_type FROM information_schema.routine_privileges
WHERE routine_name = 'match_documents';
```

```bash
# Unit tests must pass without any .env present
python -m pytest -q

# No credentials in tracked files
git grep -n -I -E "sk-[A-Za-z0-9]{20,}|eyJhbGciOi|BEGIN (RSA|OPENSSH) PRIVATE" -- .
```

## Scope and limitations

This was a static review with local execution of the existing test suite and the chunking pipeline. Supabase and OpenAI were not reachable during the review, so the RLS exposure (SEC-01) and the missing relevance filtering (PROD-03) were established by reading the migration and client code rather than by exercising a live project. No penetration testing, dependency CVE scan or infrastructure review of the hosting platform was performed. The `Dockerfile` was not built; the HEALTHCHECK finding is based on the contents of `python:3.11-slim`.
