# Verificatierapport: MCP Server readiness

**Doel:** controleren — niet analyseren — welke bouwstenen voor een MCP-server daadwerkelijk in de repository aanwezig zijn, met per tool een exacte mapping naar herbruikbare functies.

**Revisie:** `49352fe` (branch `main`), plus de branches `origin/feat/rag-pinescript` en `origin/feat/geminin-rag`.
**Datum:** 13 oktober 2026.
**Verwante documenten:** [implementatieplan_mcp.md](./implementatieplan_mcp.md), [SECURITY-REVIEW.md](./SECURITY-REVIEW.md).

Status: ✅ aanwezig · ⚠️ gedeeltelijk aanwezig · ❌ ontbreekt

---

# Werkelijk gevalideerd

*Bijgewerkt op 13 oktober 2026, nadat de MCP-MVP is geïmplementeerd. Sectie 1 tot en
met 7 hieronder beschrijven de oorspronkelijke readiness-inventarisatie op revisie
`49352fe`; de conclusies daaruit over `search_docs`, `get_context` en `find_examples`
zijn inmiddels ingehaald door de implementatie — zie
[Wijziging ten opzichte van deze inventarisatie](#wijziging-ten-opzichte-van-deze-inventarisatie).*

## Unit tests

66 tests draaien zonder netwerk en zonder Supabase, via de in-memory client
`Client(mcp)` van de SDK (geen subprocess, geen poort, geen transport).

Eén van die 66 is afhankelijk van een optionele dependency: de SQL-parservalidatie
draait alleen als `pglast` geïnstalleerd is (gedeclareerd in `requirements-dev.txt`).
Zonder dat pakket is het resultaat **65 passed, 1 skipped, 0 failed** — de test slaat
over via `pytest.importorskip` in plaats van te falen. Zie
[Testresultaten](#testresultaten).

| Bestand | Tests | Wat het dekt |
|---|---|---|
| `test_mvp_tools.py` | 37 | toolregistratie, contract-lock, gedrag per tool, foutafhandeling, stdio-subprocess |
| `test_migration_0002.py` | 16 | SQL-contract van de migratie tegen de client |
| `test_search_documents_v2.py` | 13 | clientgedrag: parameters, retries, migratiedetectie |

Wat concreet bewezen wordt:

- **Registratie en schema's.** Exact drie tools (`search_docs`, `get_context`,
  `find_examples`), elk met een bruikbare beschrijving, een inputschema met
  `required` en grenzen, en een outputschema.
- **De contract-lock.** De volledige inputschema's zijn vastgezet op eigenschappen,
  `required` en defaults (`top_k = 8`, `min_score = -1.0`, `source = None`,
  `include_code = False`, `token_budget = 8000`, `limit = 5`), plus de outputmodellen
  per tool. Elke latere contractbreuk faalt hier.
- **Gedrag per tool.** Provenance-mapping (`doc_id`, `source`, `section`, `score`,
  `chunk_count`), excerptafkapping met `truncated`-vlag, negatieve cosine-scores die
  behouden blijven, respecteren van `top_k`, budgetclamping op `PROMPT_TOKEN_BUDGET`,
  extractie van het langste codeblok, en het overslaan van niet-bruikbare blokken.
- **Filter-delegatie.** Elke tool stuurt de criteria naar de RPC
  (`similarity_threshold`, `source_filename`, `code_only`, `limit`) en vraagt exact
  `top_k` rijen — de assertie dat er géén over-fetch meer is.
- **Foutafhandeling.** Een upstream-storing wordt een `ToolError` zonder de ruwe
  upstreamtekst; ontbrekende configuratie geeft een actionable melding; een
  ontbrekende migratie noemt expliciet `0002_match_documents_v2.sql`; een ongeldige
  parameter bereikt de toolbody niet.
- **stdio-entrypoint.** Een echt subprocess (`python -m pinescript_mcp`) start,
  identificeert zich als `pinescript-docs` en adverteert de drie tools.

## Contract tests

De migratie kan niet worden uitgevoerd in deze omgeving, dus de SQL is op een andere
manier vastgezet: de tests lezen de migratietekst en vergelijken die met de client.
Commentaar wordt eerst gestript, zodat een `DROP FUNCTION IF EXISTS`-vermelding in de
rollbackinstructie niet als signatuur kan doorgaan.

| Eigenschap | Assertie |
|---|---|
| Parameternamen SQL == payloadsleutels | `{'query_embedding','match_threshold','match_count','filter_source','filter_code_only'}` aan beide kanten |
| Threshold wordt écht toegepast | `1 - (d.embedding <=> query_embedding) >= match_threshold` in de body — dit is precies de v1-bug die niet terug mag komen |
| Source-filter is letterlijk | `strpos(lower(...), lower(...)) > 0`, en géén `ILIKE` en géén `||` in de body (wildcard-injectie) |
| Code-filter wordt toegepast | `filter_code_only IS FALSE OR d.code_snippet IS TRUE` |
| Embedding-kolom komt niet terug | afwezig in `RETURNS TABLE` |
| Uitvoeringsrechten | `REVOKE EXECUTE ... FROM PUBLIC` en `GRANT ... TO service_role` aanwezig |
| v1 is onaangeraakt | zelfde signatuur, `match_threshold` nog steeds niet in de body, embedding nog in de output |
| v2 herdefinieert v1 niet | `FUNCTION public.match_documents(` komt niet voor in 0002 |
| Retrybeleid | ontbrekende functie faalt na 1 poging; transiente fout na 3 |
| Geen misclassificatie | een onbereikbare database wordt niet als migratieprobleem gemeld |

## Parser-validatie

`pglast` 8.5 (libpg_query — de daadwerkelijke PostgreSQL-parser) is gebruikt om de
syntax te controleren:

| Bestand | Resultaat |
|---|---|
| `0002_match_documents_v2.sql` | 4 statements parsen: `CREATE FUNCTION`, `COMMENT ON FUNCTION`, `REVOKE`, `GRANT` |
| functiebody van 0002 | parseert zelfstandig als 1 statement (mogelijk omdat de functie `LANGUAGE sql` is) |
| `0001_...sql` | 11 statements parsen — ongewijzigd |

`libpg_query` parst de body van een `LANGUAGE sql`-functie **niet** mee als onderdeel
van het `CREATE FUNCTION`-statement (het is daar een stringliteral), dus die body is
apart gevalideerd. Voor plpgsql-functies kan dat niet; de body van de triggerfunctie in
0001 is daarom niet op die manier te controleren. Dat is geen fout in 0001.

`pglast` is een **optionele ontwikkeldependency**, gedeclareerd in
`requirements-dev.txt`; het is niet nodig voor de runtime. De bijbehorende test gebruikt
`pytest.importorskip` en slaat over wanneer het pakket ontbreekt. Dat is geverifieerd
door het pakket te simuleren als afwezig (stub op `PYTHONPATH` die `ImportError` gooit):
het resultaat was 15 passed en 1 skipped, met de reden
"pglast is an optional SQL syntax validator".

## Benchmark-validatie

`scripts/benchmark_filtering.py` vergelijkt de oude en de nieuwe aanpak. De uitvoer
maakt expliciet onderscheid tussen gemeten en gemodelleerde getallen.

| Grootheid | Uitkomst | Aard |
|---|---|---|
| Bytes per rij (v1 → v2) | 22.626 → 6.447 | **gemeten** — echte corpus­tekst en echte pgvector-codering |
| Rijen per query bij `top_k = 8` | 32 → 8 | **gemeten** — live codepad met een gestubde RPC |
| Rijen per query bij `top_k = 3` | 12 → 3 | **gemeten** |
| Aandeel van de embedding in een rij | 71,5% | **gemeten** |
| Totale bytebesparing | 92,9% (= 4× rijen × 3,5× per rij) | **gemeten**, met de factorisering apart gerapporteerd |
| Tekort onder de oude aanpak bij een source-filter | 0,4 hits tegenover 8 | **gemodelleerd** — hangt af van de selectiviteit tegen de echte embedding-verdeling |
| Serverlatency | niet gerapporteerd | **niet gemeten** — er is geen database |

Alleen de *waarden* van de embeddingvector zijn synthetisch (er is geen live index om
ze uit te lezen); de coderingskosten per waarde zijn dat niet. De benchmark kan met
`--live` tegen een echte Supabase worden gedraaid, maar dat is in deze omgeving niet
gedaan.

# Niet gevalideerd

## Echte uitvoering op Supabase

**Migratie 0002 is nooit uitgevoerd.** Er is in deze omgeving geen Docker, geen `psql`
en geen Postgres-service. Wat daarmee onbewezen blijft:

- of de query de bedoelde rijen teruggeeft bij echte data;
- of `filter_source` met een echte bestandsnaam het verwachte aantal rijen geeft;
- of de planner de HNSW-index gebruikt, en of `hnsw.ef_search`/`iterative_scan` nodig
  is (pgvector past filters toe na de indexwandeling, dus een zeer selectief filter
  kan minder dan `match_count` rijen geven — dat staat als noot 1 in de migratie);
- wat de latency van beide RPC's is;
- of de `REVOKE`/`GRANT`-regels in het Supabase-project het gewenste effect hebben,
  met de service-role key als enige toegang.

De tools zijn dus nooit end-to-end met een echte index gedraaid: de retrieval is
steeds op het niveau van de service gestubd. Dat is een bewuste afweging — de
alternatieven waren een testproject opzetten of de validatie uitstellen — maar het
betekent dat de eerste uitvoering op een echte database de eerste echte test is.

## Productiebelasting

Niets is gemeten onder belasting: geen gelijktijdige queries, geen gedrag van de
OpenAI-embeddings onder rate limits, geen gedrag van HNSW bij herhaald filteren, geen
geheugengebruik. De benchmark meet één query per meting, niet een reeks.

Ook niet gemeten: het effect van de gerealiseerde 92,9% payloadreductie op de
daadwerkelijke responstijd van een client, en het gedrag van de MCP-server bij
langdurige sessies.

## RLS

Row Level Security is niet aangezet en dus ook niet getest. Er is geen verificatie dat
`documents` en `file_manifest` ontoegankelijk zijn voor de anon-rol, en evenmin dat de
nieuwe `REVOKE`/`GRANT` op `match_documents_v2` in de praktijk het bedoelde effect
heeft. Dit blijft open als **SEC-01** in [SECURITY-REVIEW.md](./SECURITY-REVIEW.md) en
is bewust uitgesteld.

# Testresultaten

```
tests/mcp                               66 passed, 0 failed          ~5,5 s
  test_mvp_tools.py                     37 passed
  test_migration_0002.py                16 passed
  test_search_documents_v2.py           13 passed

tests/mcp  (zonder pglast)              65 passed, 1 skipped, 0 failed
  test_migration_0002.py                15 passed, 1 skipped

server/tests                            8 failed, 68 passed, 2 skipped   ~30 s
```

**66 passed, 0 failed** in de MCP-suite met `pglast` geïnstalleerd; zonder dat pakket
**65 passed, 1 skipped, 0 failed**. Het aantal failures is in beide gevallen nul — de
parservalidatie is de enige test die van een optionele dependency afhangt. De 8 failures
in de server-suite zijn
ongewijzigd ten opzichte van de baseline van vóór de implementatie: dezelfde 8
testnamen, hetzelfde aantal. Ze zijn dus geen regressie van deze wijziging. Het zijn
vooraf bestaande problemen in `server/tests` — onder meer een stale verwachting dat
`/chat` 501 teruggeeft (het is geïmplementeerd), twee live-Supabase-tests zonder
credentials, en de `logger`-shadowingbug in `llm_client.create_chat_completion`
(PROD-06).

## Waarom de suite hermetisch is

Een autouse fixture `block_outbound_calls` in `tests/mcp/conftest.py` vervangt
`generate_single_embedding` en `search_documents_v2` door een guard die een
`AssertionError` geeft met de instructie welke fixture te gebruiken. Tests die data
nodig hebben overschrijven die guard met `fake_index` of `failing_index`.

De aanleiding is bug (d) uit
[implementatieplan_mcp.md](./implementatieplan_mcp.md#d-outbound-openai-call-bug): een
test stubbde alleen de RPC en niet de embedding, waardoor de suite met de
placeholder-key `sk-test` een **echte** call naar `api.openai.com` deed. Dat leverde
een 401 op en maakte de test traag (~30 s), netwerkafhankelijk, en liet een uitgaand
verzoek en een API-key-achtige waarde over het net gaan — alle drie ongewenst in een
unit test. Erger: de test faalde daardoor om de verkeerde reden, zodat de eigenlijke
assertie niets bewees.

Nu faalt een vergeten stub expliciet in plaats van stil naar buiten te bellen. Het
effect is meetbaar: de suite ging van ~30 s naar ~5,5 s en raakt het netwerk niet.

De enige uitzondering is de stdio-test: die start een subprocess, maar roept geen
tools aan en doet dus geen outbound call.

---

## 1. Verificatiemethode

Elke regel in dit rapport komt uit een uitgevoerde zoekopdracht of bestandslezing, niet uit eerdere analyse. Gebruikte commando's:

```bash
# Endpoints
grep -n '@app\.(get|post|put|delete|patch)' server/
# Supabase-aanroepen
grep -n '\.rpc\(|\.text_search\(|\.table\(|embeddings\.create|chat\.completions\.create' **/*.py
# Functie-inventaris per module
grep -n '^(async )?def ' server/*.py
# MCP-aanwezigheid (ook op andere branches)
git grep -n -i -E '\bmcp\b|modelcontextprotocol|FastMCP|MCPServer'
# Review/lint/repainting-logica, repo-breed en op alle branches
git grep -n -i -E 'def .*(review|lint|rule|validate|analyze)|repaint|severity'
# Filteroperaties op Supabase
grep -n '\.eq\(|\.neq\(|\.select\(|\.order\(|\.limit\(' server/supabase_client.py
# SQL-functies
grep -n 'CREATE OR REPLACE FUNCTION' migrations/*.sql
# gebruik van utils-functies in productiecode
grep -n 'extract_code_blocks|estimate_embedding_cost|...' server/**/*.py
```

Tellingen die deze controle opleverden:

```
publieke functies per module:  app.py 7 | auth.py 6 | config.py 1 | embed_client.py 6
                               ingest.py 7 | llm_client.py 3 | retriever.py 5
                               run_ingest.py 1 | supabase_client.py 8 | utils.py 7
                                    → 51 functies totaal
Supabase RPC-aanroepen:        1   (match_documents)
SQL-functies in migraties:     2   (update_updated_at_column, match_documents)
migratiebestanden:             1   (0001)
text_search-aanroepen:         1
embeddings.create-aanroepen:   1
chat.completions.create:       1
MCP-implementatie:             0
review/lint/repaint-logica:    0
```

---

## 2. Inventarisatiematrix

### 2.1 FastAPI endpoints

| Functionaliteit | Bestand | Functie | Status |
|---|---|---|---|
| Health check (licht, zonder externe calls) | `server/app.py:70` | `health()` | ✅ aanwezig |
| Login (JWT minten op e-mailadres) | `server/app.py:85` | `api_login()` | ✅ aanwezig |
| Status (config + indexstatistieken) | `server/app.py:146` | `get_status()` | ✅ aanwezig |
| Chat (volledige RAG-pipeline) | `server/app.py:178` | `chat()` | ✅ aanwezig |
| Refresh token exchange | `server/app.py:243` | `api_refresh()` | ✅ aanwezig |
| Admin: indexering triggeren | `server/app.py:331` | `trigger_indexing()` | ✅ aanwezig |
| Applicatie-lifespan (logging) | `server/app.py:32` | `lifespan()` | ✅ aanwezig |
| CORS-middleware | `server/app.py:60` | `add_middleware(CORSMiddleware)` | ⚠️ gedeeltelijk (wildcard) |
| Rate-limit exception handler | `server/app.py:57` | `add_exception_handler()` | ✅ aanwezig |
| **Zoekendpoint (`/search`) voor MCP-tools** | — | — | ❌ ontbreekt |
| **MCP-endpoint (`/mcp`) of stdio-entrypoint** | — | — | ❌ ontbreekt |

**Totaal: 6 zakelijke endpoints.** Retrieval is uitsluitend bereikbaar *binnen* `/chat`; er is geen endpoint dat alleen documenten teruggeeft.

### 2.2 Retrieval-functies

| Functionaliteit | Bestand | Functie | Status |
|---|---|---|---|
| Vector search → RetrievedDocument-objecten | `server/retriever.py:57` | `vector_search()` | ✅ aanwezig |
| Full-text search (BM25-imitaties) | `server/retriever.py:92` | `bm25_search()` | ⚠️ gedeeltelijk (score altijd 0) |
| Hybrid search (gewogen fusie) | `server/retriever.py:178` | `hybrid_search()` | ⚠️ gedeeltelijk (BM25-helft dood) |
| Context assembly binnen tokenbudget | `server/retriever.py:130` | `assemble_context()` | ✅ aanwezig |
| Code-fence-bewuste content-trimming | `server/retriever.py:21` | `_trim_content_for_budget()` | ✅ aanwezig |
| Metadata filtering (bron, sectie, code) | — | — | ❌ ontbreekt |
| Relevantiedrempel toepassen | — | — | ❌ ontbreekt |
| Re-ranking / RRF | — | — | ❌ ontbreekt |
| Document ophalen op ID | — | — | ❌ ontbreekt |
| Bronnenlijst / distinct sources | — | — | ❌ ontbreekt |

### 2.3 Vector search

| Functionaliteit | Bestand | Functie | Status |
|---|---|---|---|
| Similarity search via Supabase RPC | `server/supabase_client.py:247` | `search_similar_documents()` | ✅ aanwezig |
| RPC-aanroep `match_documents` | `server/supabase_client.py:297` | `client.rpc("match_documents", …)` | ✅ aanwezig |
| SQL-functie `match_documents` | `migrations/0001_…sql:70` | `match_documents()` | ⚠️ gedeeltelijk (`match_threshold` ongebruikt) |
| HNSW-index (cosine) | `migrations/0001_…sql:31` | `idx_documents_embedding` | ✅ aanwezig |
| Similarity-score (`1 - distance`) | `migrations/0001_…sql:111` | `ORDER BY d.embedding <=> query_embedding` | ✅ aanwezig |
| Retry met backoff rond de RPC | `server/supabase_client.py:280` | retry-lus (3 pogingen) | ✅ aanwezig |
| `section_heading` doorgeven aan de retriever | `server/retriever.py:75` | — | ❌ ontbreekt (RPC levert het, `vector_search` laat het vallen) |
| `code_snippet` doorgeven aan de retriever | `server/retriever.py:75` | — | ❌ ontbreekt (idem) |
| `chunk_count` doorgeven aan de retriever | `server/retriever.py:75` | — | ❌ ontbreekt (idem) |

### 2.4 Embedding

| Functionaliteit | Bestand | Functie | Status |
|---|---|---|---|
| OpenAI-client (singleton) | `server/embed_client.py:29` | `init_openai_client()` | ✅ aanwezig |
| Batch-embeddings met retry | `server/embed_client.py:51` | `generate_embeddings_batch()` | ✅ aanwezig |
| Gegroepeerde embeddings (batch 100) | `server/embed_client.py:100` | `generate_embeddings_chunked()` | ✅ aanwezig |
| Query-embedding (één tekst) | `server/embed_client.py:144` | `generate_single_embedding()` | ✅ aanwezig |
| Embedding-dimensie per model | `server/embed_client.py:163` | `get_embedding_dimension()` | ⚠️ gedeeltelijk (niet gebruikt in productiepad) |
| Kostenraming | `server/embed_client.py:185` | `estimate_embedding_cost()` | ⚠️ gedeeltelijk (alleen in ingest) |
| Alternatief: Google File Search store | `origin/feat/geminin-rag:server/gemini/upload_files.py` | `create_file_search_store()`, `upload_file()` | ⚠️ alleen op branch, niet op `main` |

### 2.5 Document-indexering

| Functionaliteit | Bestand | Functie | Status |
|---|---|---|---|
| Bestanden scannen | `server/ingest.py:31` | `scan_documents()` | ✅ aanwezig |
| Manifest-diff (nieuw/gewijzigd/ongewijzigd) | `server/ingest.py:315` | `check_manifest()` | ✅ aanwezig |
| Document parsen + metadata | `server/ingest.py:179` | `parse_document()` | ✅ aanwezig |
| Volledige pipeline-orchestratie | `server/ingest.py:361` | `index_documents()` | ⚠️ gedeeltelijk (blocking, zie §6) |
| Upsert naar Supabase | `server/supabase_client.py:40` | `upsert_documents()` | ⚠️ gedeeltelijk (één request, alles) |
| Manifest lezen | `server/supabase_client.py:142` | `fetch_manifest()` | ⚠️ gedeeltelijk (geen paginering) |
| Manifest schrijven | `server/supabase_client.py:95` | `update_manifest()` | ⚠️ gedeeltelijk (`chunk_count` hardcoded 1) |
| Verwijderen op bestandsnaam | `server/supabase_client.py:174` | `delete_documents_by_filename()` | ✅ aanwezig |
| Volledige reset | `server/supabase_client.py:321` | `clear_all_data()` | ⚠️ gedeeltelijk (misleidende tellingen) |
| Statistieken | `server/supabase_client.py:206` | `get_document_stats()` | ✅ aanwezig |
| Verwijderde bronnen opruimen | — | — | ❌ ontbreekt |
| CLI-ingest | `server/run_ingest.py:16` | `main()` | ✅ aanwezig |

### 2.6 Supabase RPC- en table-aanroepen

| Functionaliteit | Bestand | Functie | Status |
|---|---|---|---|
| RPC: vector search | `server/supabase_client.py:297` | `client.rpc("match_documents", payload)` | ✅ aanwezig |
| RPC met filters (bron/sectie/code) | — | — | ❌ ontbreekt |
| RPC voor full-text rank | — | — | ❌ ontbreekt |
| `documents` upsert | `server/supabase_client.py:80` | `.table(rag_vector_table).upsert(…)` | ✅ aanwezig |
| `documents` delete per bron | `server/supabase_client.py:192` | `.delete().eq("source_filename", …)` | ✅ aanwezig |
| `documents` count | `server/supabase_client.py:220` | `.select("id", count="exact")` | ✅ aanwezig |
| `documents` delete all | `server/supabase_client.py:343` | `.delete().neq("id", "")` | ⚠️ gedeeltelijk |
| `file_manifest` upsert/select/delete | `server/supabase_client.py:127, 154, 337` | `.upsert()` / `.select("*")` / `.delete().neq()` | ✅ aanwezig |
| `file_manifest` laatste indexering | `server/supabase_client.py:226` | `.order("last_indexed", desc=True).limit(1)` | ✅ aanwezig |
| Directe `text_search` op `documents` | `server/retriever.py:105` | `.select("*").text_search("content", …)` | ⚠️ gedeeltelijk (geen rank, geen index) |
| SQL-trigger `updated_at` | `migrations/0001_…sql:36` | `update_updated_at_column()` | ✅ aanwezig |
| RLS-policies | — | — | ❌ ontbreekt |

### 2.7 Chunking-logica

| Functionaliteit | Bestand | Functie | Status |
|---|---|---|---|
| Chunking op H1/H2-koppen | `server/ingest.py:58` | `chunk_document()` | ⚠️ gedeeltelijk (alleen H1/H2) |
| Overlap-extractie uit vorige chunk | `server/ingest.py:145` | `extract_token_overlap()` | ✅ aanwezig |
| Harde split bij model-limiet | `server/ingest.py:250` | `split_text_by_token_limit()` | ✅ aanwezig |
| Tokentelling | `server/utils.py:25` | `count_tokens()` | ✅ aanwezig |
| Code-detectie | `server/utils.py:56` | `detect_code_snippets()` | ⚠️ gedeeltelijk (89% van chunks = true) |
| Code-blokken extraheren | `server/utils.py:86` | `extract_code_blocks()` | ⚠️ aanwezig maar **ongebruikt in productie** |
| Deterministische doc-ID | `server/utils.py:111` | `generate_doc_id()` | ⚠️ gedeeltelijk (op bestandsnaam met timestamp) |
| Content-hash | `server/utils.py:44` | `hash_string()` | ✅ aanwezig |
| Semantische / recursieve splitter | — | — | ❌ ontbreekt |
| Chunkkwaliteitsvalidatie | — | — | ❌ ontbreekt |

### 2.8 Auth-logica

| Functionaliteit | Bestand | Functie | Status |
|---|---|---|---|
| JWT-verificatie (dependency) | `server/auth.py:99` | `verify_jwt_token()` | ✅ aanwezig |
| HS256 via gedeeld secret | `server/auth.py:120` | `jwt.decode(…, algorithms=[…])` | ✅ aanwezig |
| RS256 via JWKS | `server/auth.py:82, 148` | `_fetch_jwks()` | ⚠️ gedeeltelijk (sync `requests` in async pad) |
| Allowlist (deny-by-default) | `server/auth.py:32, 49` | `_parse_allowed_emails()`, `is_email_allowed()` | ✅ aanwezig |
| Admin-key verificatie | `server/auth.py:64` | `verify_admin_key()` | ⚠️ gedeeltelijk (`!=`, geen rotatie) |
| Rate-limit key (per JWT-sub) | `server/auth.py:179` | `get_rate_limit_key()` | ⚠️ gedeeltelijk (in-memory, per worker) |
| Tokens uitgeven | `server/app.py:85` | `api_login()` | ⚠️ gedeeltelijk (geen wachtwoord) |
| Refresh-flow | `server/app.py:243` | `api_refresh()` | ⚠️ gedeeltelijk (geen `typ`, geen revocatie) |
| Rollen / autorisatieniveaus | — | — | ❌ ontbreekt |
| MCP-auth (OAuth 2.1 / API-tokens) | — | — | ❌ ontbreekt |

---

## 3. Per-tool readiness

> **Deels ingehaald.** `search_docs`, `get_context` en `find_examples` zijn inmiddels
> geïmplementeerd en getest; de status hieronder beschrijft de situatie vóór die
> implementatie en geldt nog onverkort voor `explain_function`, `review_pinescript` en
> `optimize_strategy`.

| Tool | Kernonderdeel aanwezig? | Status | Blokkerend? |
|---|---|---|---|
| `search_docs` | Embedding ✅ + vector search ✅ + RPC ✅ | ⚠️ **bijna gereed** — mist filtering, drempel en sectie-provenance | Nee (werkbaar, maar zonder drempel en filters) |
| `get_context` | `assemble_context` ✅ + budgettrimming ✅ | ✅ **gereed** | Nee |
| `find_examples` | `extract_code_blocks` ⚠️ + vector search ✅ | ⚠️ **gedeeltelijk** — extractie bestaat maar wordt nergens gebruikt; `code_snippet`-vlag is onbruikbaar als filter | Nee |
| `explain_function` | — | ❌ **ontbreekt** — geen functiecatalogus, en de Language Reference is niet gecrawld | **Ja** |
| `review_pinescript` | — | ❌ **ontbreekt** — nul regels review-/lintlogica in de hele repo, op geen enkele branch | **Ja** |
| `optimize_strategy` | RAG-infrastructuur ✅ | ⚠️ **gedeeltelijk** — alleen RAG-gebaseerd advies mogelijk; geen codeanalyse | Nee (RAG-versie wel) |

### Ondersteunende bouwstenen die de tools nodig hebben

| Bouwsteen | Status | Toelichting |
|---|---|---|
| `get_chunk` (document op ID) | ❌ ontbreekt | Geen enkele functie haalt een document op basis van `id`; alleen delete/upsert per bestandsnaam of bron |
| Bronnenlijst (`list_sources`) | ❌ ontbreekt | Geen `distinct`-query of equivalent |
| Kleine payload-vormgeving | ❌ ontbreekt | Geen enkele functie kort `content` in tot een begrensde hit-representatie |
| MCP-server / transport | ❌ ontbreekt | Geen SDK-afhankelijkheid, geen tool-registratie, geen stdio- of HTTP-entrypoint |

---

## 4. Exacte herbruikbare functies per tool

### 4.1 `search_docs`

| Herbruikbaar | Bestand | Functie | Status | Opmerking |
|---|---|---|---|---|
| Embedding van de query | `server/embed_client.py:144` | `generate_single_embedding()` | ✅ direct herbruikbaar | — |
| Vector search | `server/retriever.py:57` | `vector_search()` | 🔧 uitbreiden | Moet `section_heading`/`code_snippet`/`chunk_count` doorgeven |
| RPC-aanroep | `server/supabase_client.py:247` | `search_similar_documents()` | 🔧 uitbreiden | `similarity_threshold` wordt nu niet doorgegeven aan de DB |
| Hybrid fusie | `server/retriever.py:178` | `hybrid_search()` | ⚠️ niet gebruiken in v1 | BM25-scores zijn altijd 0 |
| Tokentelling | `server/utils.py:25` | `count_tokens()` | ✅ direct herbruikbaar | Voor payload-begroting |
| Config | `server/config.py:138` | `get_config()` | ✅ direct herbruikbaar | `retrieval_top_k`, `hybrid_bm25_weight` |

**Nieuw:** `min_score`-filter, `include_code`-filter, payload-vormgeving, `DocHit`-model.

### 4.2 `get_context`

| Herbruikbaar | Bestand | Functie | Status | Opmerking |
|---|---|---|---|---|
| Context assembly | `server/retriever.py:130` | `assemble_context()` | ✅ direct herbruikbaar | Neemt `token_budget` al als parameter |
| Budgettrimming | `server/retriever.py:21` | `_trim_content_for_budget()` | ✅ direct herbruikbaar | Privéfunctie; mag als intern gelden |
| Prompt-tokenschatting | `server/utils.py:129` | `estimate_prompt_tokens()` | ✅ direct herbruikbaar | — |
| Retrieval | `server/retriever.py:57` / `:178` | `vector_search()` / `hybrid_search()` | ✅ / ⚠️ | Zelfde voorbehoud als bij `search_docs` |

**Nieuw:** niets wezenlijks. Dit is de goedkoopste tool.

### 4.3 `find_examples`

| Herbruikbaar | Bestand | Functie | Status | Opmerking |
|---|---|---|---|---|
| Code-blokken extraheren | `server/utils.py:86` | `extract_code_blocks()` | ⚠️ bestaat, **ongebruikt** | Wordt alleen door tests aangeroepen; dit wordt de eerste productiegebruiker |
| Code-detectie | `server/utils.py:56` | `detect_code_snippets()` | ✅ direct herbruikbaar | Alleen als signaal, niet als filter |
| Retrieval | `server/retriever.py:57` | `vector_search()` | ✅ direct herbruikbaar | — |
| Tokentelling | `server/utils.py:25` | `count_tokens()` | ✅ direct herbruikbaar | Voor het capen van codeblokken |

**Nieuw:** ranking op code-dichtheid, captioning, `ExampleHit`-model, taalfilter.

### 4.4 `explain_function`

| Herbruikbaar | Bestand | Functie | Status | Opmerking |
|---|---|---|---|---|
| RPC/vector search | `server/retriever.py:57` | `vector_search()` | ✅ direct herbruikbaar | Levert alleen **prose**; de manual noemt 338 functies maar zonder signaturen |
| Parings-/chunklogica | `server/ingest.py:179` | `parse_document()` | 🔧 herbruikbaar | Nodig voor de catalogus-ETL |
| Embeddings voor de catalogus | `server/embed_client.py:100` | `generate_embeddings_chunked()` | ✅ direct herbruikbaar | — |
| JSON-generator | `scripts/md_to_json.py` | `build_language_reference()` | ❌ **niet bruikbaar** | Schrijft letterlijk `functions: {}` en `variables: {}` — lege skeletten |
| Tokentelling | `server/utils.py:25` | `count_tokens()` | ✅ direct herbruikbaar | — |

**Nieuw:** scraper voor `reference/functions/*` en `reference/variables/*`, catalogus-ETL, aparte tabel/index, `FunctionInfo`-model. **Dit is de enige tool met een ontbrekende databron.**

### 4.5 `review_pinescript`

| Herbruikbaar | Bestand | Functie | Status | Opmerking |
|---|---|---|---|---|
| — | — | — | ❌ **niets** | Geverifieerd met `git grep -n -i` op de termen `review`, `lint`, `rule`, `validate`, `analyze`, `repaint` en `severity`: in de hele repo (alle branches) uitsluitend treffers in [SECURITY-REVIEW.md](./SECURITY-REVIEW.md) |
| Optioneel: LLM voor toelichting | `server/llm_client.py:39` | `create_chat_completion()` | 🔧 herbruikbaar | Alleen voor suggestieteksten, niet voor detectie |
| Systeemprompt | `server/llm_client.py:16` | `build_system_prompt()` | 🔧 herbruikbaar | Moet een Pine-review-instructie worden |

**Nieuw:** lexer/scope-detectie, rule engine, ~15 regels, `Finding`/`ReviewReport`-modellen. Volledig nieuw werk.

### 4.6 `optimize_strategy`

| Herbruikbaar | Bestand | Functie | Status | Opmerking |
|---|---|---|---|---|
| Retrieval van optimalisatiedocumentatie | `server/retriever.py:57` | `vector_search()` | ✅ direct herbruikbaar | Pagina `48_profiling-and-optimization` bestaat in de corpus |
| Context assembly | `server/retriever.py:130` | `assemble_context()` | ✅ direct herbruikbaar | — |
| LLM-aanroep | `server/llm_client.py:39` | `create_chat_completion()` | ✅ direct herbruikbaar | Neemt al `conversation_history` |
| Systeemprompt | `server/llm_client.py:16` | `build_system_prompt()` | 🔧 herbruikbaar | Moet een optimalisatie-instructie worden |
| Modelkeuze | `server/llm_client.py:25` | `choose_model()` | ✅ direct herbruikbaar | — |

**Nieuw:** prompt, doel-interpretatie, verwijzingen naar bronpagina's. **Geen** echte codeanalyse.

---

## 5. Wat ontbreekt (geverifieerd)

| # | Ontbrekend onderdeel | Bewijs |
|---|---|---|
| 1 | MCP-implementatie | `git grep -i '\bmcp\b'` → uitsluitend `scripts/md_to_json.py`, dat JSON wegschrijft naar een **externe** server (`../mcp-server-pinescript`) |
| 2 | Functiecatalogus | `scripts/md_to_json.py` schrijft `functions: {}` en `variables: {}`; corpus bevat geen `reference/functions` |
| 3 | Review-/lint-/repaintlogica | Nul treffers in de hele repo op alle branches |
| 4 | Document ophalen op ID | Geen `.eq("id", …)`; alleen delete/upsert op `source_filename` |
| 5 | Bronnenlijst | Geen `distinct`-query of equivalent |
| 6 | Filter-RPC | `match_documents` heeft geen filterparameters (`migrations/0001_…sql:70-73`) |
| 7 | Werkende relevantiedrempel | `match_threshold` wordt aangenomen maar nooit gebruikt (`migrations/0001_…sql:112`) |
| 8 | Sectie-provenance in retrieval | `retriever.py` bevat **geen enkele** verwijzing naar `section_heading`, `chunk_count` of `code_snippet` (grep: 0 treffers) |
| 9 | Sectie-provenance in het DTO | `RetrievedDocument` (`server/models.py:142`) heeft 7 velden; `section_heading` ontbreekt |
| 10 | `/search`-endpoint | Alleen `/health`, `/api/login`, `/status`, `/chat`, `/api/refresh`, `/internal/index` |
| 11 | Kleine-payload-representatie | Geen functie kort hits in tot begrensde velden |
| 12 | MCP-auth | Geen OAuth 2.1 / tokenflow voor MCP-clients |
| 13 | RLS | Geen `ENABLE ROW LEVEL SECURITY` in de migratie |

---

## 6. Belangrijke correcties op eerdere aannames

Deze controle leverde vier punten op die het plan materieel beïnvloeden:

1. **`extract_code_blocks` bestaat wel, maar wordt nergens gebruikt.** Geverifieerd: alleen `tests/test_utils.py` roept het aan. `find_examples` is dus niet "75% klaar door hergebruik", maar "de bouwsteen ligt klaar en moet voor het eerst in productie worden aangesloten".
2. **De sectie-metadata gaat verloren in de retriever, niet in de database.** De RPC retourneert `section_heading`, `chunk_count` en `code_snippet` wél (`migrations/0001_…sql:81-83`), maar `vector_search()` (`server/retriever.py:75`) bouwt een `RetrievedDocument` met slechts 7 velden. De reparatie is dus klein en lokaal.
3. **`review_pinescript` en `find_repainting_risks` hebben geen enkele bestaande basis.** Niet "20%", maar 0%. De 9 regels van de JS-server (`iamrichardD/mcp-server-pinescript`) zijn de enige externe referentie — die staan in een andere repository.
4. **De andere branches bevatten geen extra retrievalcode.** `retriever.py` en `supabase_client.py` hebben op `main`, `origin/feat/rag-pinescript` en `origin/feat/geminin-rag` **identieke functiesets**. De Gemini-branch voegt een alternatieve RAG-route toe (Google File Search store), geen extra retrievalfuncties.

---

## 7. Conclusie

**Klaar om te bouwen zonder blokkades:** `get_context`, `search_docs`, `find_examples`.
**Niet klaar wegens ontbrekende code:** `review_pinescript` (volledig nieuw).
**Niet klaar wegens ontbrekende data:** `explain_function` (Language Reference is nooit gecrawld).
**Gedeeltelijk:** `optimize_strategy` (RAG-versie kan direct, echte analyse niet).

De infrastructuur voor de drie retrieval-tools is compleet: embedding, vector search, RPC, context assembly en budgettrimming zijn allemaal aanwezig en functioneren. De ontbrekende onderdelen zitten consequent in drie categorieën: **filtering en drempels** (in de RPC), **payload-vormgeving** (in de MCP-laag) en **domeinkennis** (review-regels en de functiecatalogus).

Voor een werkende MCP-server met `search_docs`, `get_context` en `find_examples` is geen enkele wijziging aan de database of aan `server/` nodig buiten de twee kleine uitbreidingen voor sectie-provenance (§5, items 8 en 9). Voor de volledige toolset van zes is één scraper-uitbreiding en één nieuwe rule engine nodig.

## Wijziging ten opzichte van deze inventarisatie

De inventarisatie hierboven beschrijft de situatie op revisie `49352fe`, vóór de
implementatie. Deze tabel legt vast wat daarvan inmiddels achterhaald is.

| Toen vastgesteld | Nu | Toelichting |
|---|---|---|
| §5.7: werkende relevantiedrempel ontbreekt — `match_threshold` wordt genegeerd | ⚠️ opgelost in code, niet uitgevoerd | `match_documents_v2` past de drempel toe (`>= match_threshold`). De migratie is geschreven en statisch gevalideerd, maar nooit op een database uitgevoerd |
| §5.6: filter-RPC ontbreekt — `match_documents` heeft geen filterparameters | ⚠️ opgelost in code, niet uitgevoerd | `filter_source` en `filter_code_only` zijn toegevoegd; idem qua uitvoering |
| §5.8/5.9 en §2.3: sectie-provenance gaat verloren in `vector_search` | ✅ opgelost | Niet door `RetrievedDocument` uit te breiden, maar door in de MCP-laag de ruwe RPC-rijen te lezen. `server/models.py` en `server/retriever.py` bleven daardoor onaangeraakt |
| §2.6 en §4.1: `search_similar_documents` is het herbruikbare retrievalpad | ❌ niet gevolgd | De MCP-laag gebruikt een nieuwe functie `search_documents_v2`; `search_similar_documents` bleef ongemoeid zodat `/chat` niet raakt |
| §2.7 en §4.3: `extract_code_blocks` bestaat maar wordt nergens in productie gebruikt | ✅ opgelost | `find_examples` is de eerste productiegebruiker, via `service._iter_code_blocks` |
| §5.4: document ophalen op ID ontbreekt | ❌ nog open | `get_chunk` bestaat nog niet; `search_docs` levert alleen een excerpt |
| §5.13: RLS ontbreekt | ❌ nog open | Bewust uitgesteld als SEC-01. Alleen de nieuwe functie is afgeschermd met `REVOKE`/`GRANT` |
| §4.4: `explain_function` vereist een scraper-uitbreiding | ❌ nog open | Ongewijzigd; de Language Reference is nog steeds niet gecrawld |
| §4.5: `review_pinescript` heeft nul bestaande basis | ❌ nog open | Ongewijzigd; er bestaat nog geen rule engine of lintcode |
| §7 slotsom: "geen enkele wijziging aan de database of aan `server/` nodig" | ❌ achterhaald | Er is juist wél een databasewijziging (0002) en één additieve functie in `server/supabase_client.py` nodig gebleken om de filtering database-side te krijgen |
| §2.5/§2.6: `index_documents` en `upsert_documents` functioneren maar zijn blocking en ongepagineerd | ❌ nog open | Buiten scope gelaten; PROD-04 en PROD-05 in [SECURITY-REVIEW.md](./SECURITY-REVIEW.md) |
| §6.1: de index vervuilt per scrape-cyclus (PROD-01) | ❌ nog open | Niet aangeraakt; blijft het grootste data-integriteitsrisico |
