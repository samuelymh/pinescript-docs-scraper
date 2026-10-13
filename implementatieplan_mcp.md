# Implementatieplan: PineScript MCP Server

**Doel:** deze repository omvormen tot een MCP-server met de **officiële Python MCP SDK** (`mcp`), bruikbaar in GitHub Copilot Agent, Claude Code, Roo Code, Cline en Continue.dev.

**Uitgangspunt:** de bestaande FastAPI-app, Supabase-index en retrieval-code blijven functioneren. De MCP-laag komt er als een dunne, nieuwe package bovenop.

**Versie:** opgesteld op 13 oktober 2026 tegen revisie `49352fe`.
Verwante documenten: [SECURITY-REVIEW.md](./SECURITY-REVIEW.md), [SUPABASE_SCHEMA.md](./SUPABASE_SCHEMA.md), [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md), [server/STARTUP.md](./server/STARTUP.md).

---

# Status na implementatie

*Bijgewerkt op 13 oktober 2026. Dit hoofdstuk beschrijft de werkelijk gerealiseerde
situatie. De secties 0 tot en met 9 daarna zijn het oorspronkelijke plan en blijven
staan als historisch uitgangspunt; waar plan en werkelijkheid uiteenlopen staat dat
expliciet in [Geplande versus werkelijke status](#geplande-versus-werkelijke-status).*

## Voltooid

### Gerealiseerde MCP tools

| Tool | Status | Filtering | Output |
|---|---|---|---|
| `search_docs` | ✅ werkt | database-side (similarity, source, code-only) | `list[DocHit]`, excerpt ≤ 500 tekens |
| `get_context` | ✅ werkt | geen filters; tokenbudget als plafond | `ContextPack`, volledige passages |
| `find_examples` | ✅ werkt | database-side (`code_only=true`) + Python-extractie van codeblokken | `list[ExampleHit]`, code ≤ 2000 tekens |

`review_pinescript` en `explain_function` zijn conform afspraak uitgesteld. De tools
zijn synchroon gedefinieerd, zodat de SDK ze op een worker thread draait en de
blocking OpenAI- en Supabase-calls de event loop niet blokkeren.

### Gerealiseerde infrastructuur

| Onderdeel | Omvang | Toelichting |
|---|---|---|
| `pinescript_mcp/` | 6 modules, 709 regels | `app.py` (MCPServer + registratie via `add_tool`), `service.py`, `tools.py`, `schemas.py`, `__init__.py`, `__main__.py` |
| `migrations/0002_match_documents_v2.sql` | 151 regels | nieuwe RPC met werkende threshold en filters; `match_documents` onaangeraakt |
| `server/supabase_client.py` | +123 / −0 regels | additief: `search_documents_v2()`, `MigrationRequiredError`, `_is_missing_function_error()`. Nul verwijderingen |
| `scripts/benchmark_filtering.py` | 388 regels | vergelijkt oude en nieuwe aanpak; `--live` meet tegen een echte Supabase |
| `requirements-mcp.txt` | 24 regels | `mcp[cli]>=2.3,<3` plus de verhoogde floors (`pydantic>=2.12`, `anyio>=4.9`, `typing-extensions>=4.13`) |
| `requirements-dev.txt` | 6 regels | optionele ontwikkeldependency `pglast>=8.5` voor de SQL-parservalidatie |
| `docs/MCP.md` | 330 regels | installatie, migratietabel, configuratie voor 4 clients, gedrag, HTTP-mount |
| `docs/DEPLOYMENT.md` | +223 / −2 regels | waarvan 208 regels de nieuwe MCP-deploysectie: verplichte uitrolvolgorde, vier verificatiequeries met benoemde argumenten, rollback |
| `tests/mcp/` | 4 bestanden, 955 regels | 66 tests (65 + 1 skipped zonder `pglast`) |

> Regelaantallen in dit document zijn **fysieke regels, inclusief lege regels**, zoals
> geteld met `@(Get-Content <bestand>).Count`. Let op: `Measure-Object -Line` telt lege
> regels niet mee en geeft daardoor een 10 tot 40 procent lager getal.

De SDK-versie is `mcp` **2.3.0** met de v2-API (`from mcp.server import MCPServer`);
`FastMCP` bestaat in v2 niet meer.

### Uitgevoerde tests

| Suite | Resultaat | Duur |
|---|---|---|
| `tests/mcp` (met `pglast`) | **66 passed, 0 failed** | ~5,5 s |
| `tests/mcp` (zonder `pglast`) | 65 passed, 1 skipped, 0 failed | ~5,5 s |
| `server/tests` | 8 failed, 68 passed, 2 skipped | ~30 s |

De parservalidatie is de enige test die van een optionele dependency afhangt; het aantal
failures is nul in beide gevallen.

De 8 server-failures zijn ongewijzigd ten opzichte van de baseline van vóór de
implementatie (identieke testnamen) en zijn dus geen regressie. Zie
[verificatie_mcp.md](./verificatie_mcp.md) voor de volledige toelichting.

Verdeling van de MCP-suite: 37 tests in `test_mvp_tools.py` (registratie,
contract-lock, gedrag per tool, foutafhandeling, stdio-subprocess), 16 in
`test_migration_0002.py` (SQL-contract), 13 in `test_search_documents_v2.py`
(client-gedrag, retries, migratiedetectie).

## Gevonden bugs

Vijf fouten die tijdens de implementatie zijn gevonden en opgelost. Vier daarvan
zaten in de nieuwe code, één in de onderbouwing.

### a. `min_score`-bug

**Oorzaak.** `min_score` stond standaard op `0.0` en werd toegepast als
`if score < floor: continue`. Cosine similarity van genormaliseerde OpenAI-embeddings
ligt in `[-1, 1]`, dus een default van `0.0` gooide élke rij met een negatieve score
weg — en dat is een legitiem deel van het bereik.

**Impact.** Stil dataverlies. De tool gaf minder resultaten zonder foutmelding, dus
dit was niet te zien in logs of metrics en zou in productie onopgemerkt zijn
gebleven. Een agent zou concluderen dat de documentatie het antwoord niet bevat.

**Oplossing.** Default naar `DEFAULT_MIN_SCORE = -1.0`, de onderkant van het
cosine-bereik, met een docstring die uitlegt waarom. Daarnaast is de `or`-valkuil
verwijderd: `float(min_score or 0.0)` zou `None` naar `0.0` hebben omgezet en zo de
bug hebben herintroduceerd; nu is het
`float(min_score) if min_score is not None else DEFAULT_MIN_SCORE`. Vastgelegd met
`test_keeps_negative_cosine_scores` en met de contract-lock op de default `-1.0`.

### b. Regex-signatuur-bug

**Oorzaak.** De testhelper `signature_parameters()` zocht met
`match_documents_v2\((.*?)\)\s*RETURNS` over de volledige migratietekst. De
rollback-instructie bovenaan die migratie bevat
`DROP FUNCTION IF EXISTS public.match_documents_v2(vector, double precision, integer, text, boolean);`
— dat is de eerste treffer in het bestand, dus de helper las een commentaarregel als
"de signatuur".

**Impact.** De helper gaf `['vector', 'double precision', ...]` terug in plaats van
de parameternamen. Dat is de test die SQL/client-drift moet betrappen; een test die
de verkeerde tekst parseert kan een echte drift juist goedkeuren.

**Oplossing.** `strip_sql_comments()` verwijdert `--`- en `/* */`-commentaar vóór het
parsen, en de zoekopdracht is verankerd op `FUNCTION\s+(?:public\.)?name\(` zodat een
`DROP FUNCTION IF EXISTS`-vermelding niet als definitie kan doorgaan. De
drift-test zelf is ongewijzigd in intentie.

### c. ILIKE-test-bug

**Oorzaak.** `assert "ILIKE" not in body.upper()` faalde omdat de functiebody een
commentaar bevat dat uitlegt waarom `strpos` in plaats van ILIKE wordt gebruikt. De
assertie testte de documentatie in plaats van de code.

**Impact.** Een testfout die de code onterecht afkeurde — een verkeerd signaal dat
uitnodigt tot het "repareren" van correcte code. Erger: de assertie dekte de
eigenschap die er echt toe doet niet af.

**Oplossing.** Commentaar wordt gestript vóór de assertie, dus er wordt op code
getest. Daarnaast is een scherpere assertie toegevoegd: er mag geen `||`
(stringconcatenatie) in de body staan, want dat is de constructie waarmee een
caller-supplied `%` alsnog een wildcard zou worden.

### d. Outbound OpenAI-call-bug

**Oorzaak.** De test `test_missing_migration_is_reported_actionably` stubbde alleen
`search_documents_v2`, maar `_fetch_rows` embedt de query éérst. De embedding-call
ging dus naar de echte OpenAI-API met de placeholder-key `sk-test`.

**Impact.** Drie problemen tegelijk: de suite deed een uitgaand verzoek dat niet in
een unit test thuishoort, werd daardoor netwerkafhankelijk en traag (~30 s), en de
test faalde om de verkeerde reden (generieke upstream-fout in plaats van de
migratiemelding) waardoor de eigenlijke assertie niets bewees. Dit is het gevaarlijkste
van de vijf, omdat het patroon zich in elke nieuwe test herhaalt.

**Oplossing.** De test gebruikt nu `fake_index`, dat beide calls stubt. Belangrijker:
een **autouse fixture** `block_outbound_calls` vervangt beide outbound calls door een
guard die hard faalt met een instructie welke fixture te gebruiken. Elke test die
voortaan een stub vergeet, faalt expliciet in plaats van stil naar buiten te bellen.
De suite is nu hermetisch en duurt ~5,5 s. Zie
[verificatie_mcp.md](./verificatie_mcp.md#waarom-de-suite-hermetisch-is).

### e. Benchmarklabel-bug

**Oorzaak.** Het label `embedding_share_of_legacy` werd berekend als
`1 - new_bytes / legacy_bytes`. Dat is de *totale* bytebesparing, niet het aandeel van
de embedding-kolom. Die totale besparing is het product van twee effecten: 4× minder
rijen én een kleinere rij.

**Impact.** Een onjuist getal in precies het onderdeel dat een ontwerpbeslissing moet
onderbouwen. "92,9% is het aandeel van de embedding" leest als "de embedding is bijna
de hele payload", terwijl het per-rij-aandeel 71,5% is en de rest van de besparing uit
de rijreductie komt. Een beslissing op dat getal zou de verkeerde optimalisatie kiezen.

**Oplossing.** Per-rij-meting toegevoegd (`row_bytes_v1`, `row_bytes_v2`) en twee
gescheiden grootheden gerapporteerd: 71,5% embedding-aandeel per rij en 92,9% totale
besparing, met de factorisering expliciet vermeld (4× rijen × 3,5× per rij). De
benchmark onderscheidt nu ook in de kop welke getallen gemeten zijn en welke
gemodelleerd.

## Bekende beperkingen

### Filtering in Python — *correctie: filtering is verplaatst naar SQL*

Het plan voorzag filtering in Python met een over-fetch. Dat is **niet** de
eindtoestand: de score-, source- en code-filters worden door `match_documents_v2` in
PostgreSQL toegepast en `pinescript_mcp/service.py` gooit niets meer weg. De
`continue`-takken voor die drie filters zijn verwijderd.

Wat resteert is een andere beperking: **migratie 0002 is nog niet toegepast op enige
database.** De code delegeert de filtering dus wel, maar dat pad is nooit uitgevoerd.
Gevolg voor de operationele volgorde: tegen een database zonder 0002 mislukken alle
drie de tools met een `MigrationRequiredError` en een melding die het migratiebestand
noemt. Dat is een bewuste keuze (duidelijke fout boven stilzwijgende degradatie), maar
het betekent dat **code en migratie samen uitgerold moeten worden**. Zie
[Aanbevolen vervolgstappen](#aanbevolen-vervolgstappen), P1.

### Over-fetch-strategie — *correctie: verwijderd, op één uitzondering na*

`SEARCH_OVERFETCH = 4` en `CONTEXT_OVERFETCH = 2` zijn verwijderd. De RPC past het
predicaat toe vóór `LIMIT`, dus de client vraagt exact `top_k` rijen en er wordt niets
gediscard.

Eén uitzondering blijft, met een andere reden: `EXAMPLE_SCAN_FACTOR = 2` in
`find_examples`. De vlag `code_snippet` is ook waar voor chunks met alleen *inline*
code (tussen enkele backticks), en die leveren geen fenced block op voor
`extract_code_blocks`. Omdat het kiezen van het beste codeblok zich niet als
SQL-predicaat laat uitdrukken, wordt er een kleine surplus gescand. Zonder die factor
zou een pagina met prose-zware chunks stil minder voorbeelden opleveren. Dit is dus
geen filter-over-fetch; er is een test die het onderscheid aantoont
(`test_tolerates_code_flagged_chunks_without_fenced_blocks`).

### Ontbreken van RLS

Row Level Security staat nog steeds niet aan op `documents` en `file_manifest`, en de
`match_documents`-RPC is nog steeds uitvoerbaar door de publieke rollen. Dit is
**SEC-01 uit [SECURITY-REVIEW.md](./SECURITY-REVIEW.md) en is bewust uitgesteld** naar
een eigen taak, omdat het een keuze over de hele Supabase-configuratie vereist en niet
alleen over de MCP-laag.

Wat wél is gedaan: de nieuwe functie herhaalt de fout niet. PostgreSQL geeft nieuwe
functies standaard `EXECUTE` aan `PUBLIC`, wat in Supabase `anon` betekent; daarom
bevat 0002 expliciet `REVOKE EXECUTE ... FROM PUBLIC` en
`GRANT EXECUTE ... TO service_role`. Zonder die twee regels zou deze wijziging een
tweede publiek aanroepbare vectorsearch hebben toegevoegd.

### Ontbreken van productievalidatie van migratie 0002

De migratie is **statisch** gevalideerd, niet uitgevoerd:

- `pglast` (libpg_query, de echte PostgreSQL-parser) parst het bestand als 4
  statements en de functiebody apart — geen syntaxfouten.
- Contracttests houden de SQL-tekst en de client-payload tegen elkaar: de
  parameternamen moeten exact overeenkomen, de body moet `match_threshold` echt
  gebruiken, de source-filter mag geen wildcardconstructie zijn, en de embedding-kolom
  mag niet terugkomen.
- `match_documents` (v1) blijft aantoonbaar onaangeraakt: dezelfde signatuur, dezelfde
  body die `match_threshold` negeert, en de embedding-kolom nog in de output.

Wat daar **niet** mee bewezen is: dat de query de juiste rijen teruggeeft, dat de
planner de HNSW-index gebruikt, en wat de latency is. Er is in deze omgeving geen
Docker, geen `psql` en geen Postgres-service, dus uitvoering was onmogelijk. Zie
[verificatie_mcp.md](./verificatie_mcp.md#niet-gevalideerd).

## Aanbevolen vervolgstappen

### P1 — vóór enige vorm van gebruik met een echte index

1. **Migratie 0002 uitvoeren op een testproject en verifiëren met echte data.** Minimaal:
   een query met `filter_source` waarin je het aantal rijen controleert, een query met
   `code_only = true`, een query met een hoge `match_threshold` (moet minder rijen
   geven dan zonder), en `EXPLAIN ANALYZE` om te zien of de HNSW-index gebruikt wordt.
   Pas daarna naar productie.
2. **Uitrolvolgorde vastleggen.** 0002 moet toegepast zijn vóór de nieuwe code
   draait, anders falen alle drie de tools. Zet dit in
   [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md) en in de CI/deploymentstap.
3. **De merge afronden.** De repository zit nog midden in een merge met 70
   conflictpaden; committen van deze wijzigingen is daardoor geblokkeerd. Zie
   [COMMIT_PREVIEW.md](./COMMIT_PREVIEW.md).

### P2 — kwaliteit en veiligheid van de bestaande functionaliteit

4. **RLS aanzetten (SEC-01)**, inclusief `REVOKE EXECUTE` op de bestaande
   `match_documents` en policies voor beide tabellen. Dit is de laatste
   deployment-blocker uit de securityreview.
5. **Hybrid search repareren of verwijderen.** `bm25_search` leest een `rank`-veld dat
   de Supabase-client nooit teruggeeft, dus de score is altijd 0 en de
   `hybrid_bm25_weight` van 0.3 doet niets. Dit vraagt een `tsvector`-kolom met
   GIN-index en `ts_rank_cd`, of het expliciet uitzetten van hybrid search.
6. **Drempel en re-ranking.** `min_score` filtert nu, maar er is nog geen
   diversiteits- of rerankingstap; de context kan near-duplicates bevatten.
7. **`get_chunk` toevoegen** en `RetrievedDocument` uitbreiden. Nu levert `search_docs`
   alleen een excerpt; een agent die de volledige chunk wil moet `get_context`
   gebruiken, wat duurder is dan nodig.

### P3 — uitbreiding en consolidatie

8. **`review_pinescript` en `find_repainting_risks`** op één gedeelde rule engine.
   Er bestaat nog geen enkele regel of lintcode in de repository.
9. **`explain_function`**, wat eerst een scraper-uitbreiding vereist: de Language
   Reference (`reference/functions/*`) is nooit gecrawld. De 73 pagina's zijn volledig
   de User Manual, dus er zijn geen signaturen beschikbaar.
10. **Repositoryhygiëne**: `log/`, `__pycache__`-bestanden, de dubbele
    `processed_all_docs.md` en de dubbele docs-generatie opruimen; `requirements.txt`
    splitsen in scraper- en serverdependencies met een lockbestand.
11. **Streamable HTTP + auth**, pas wanneer er een remote deployment nodig is. stdio
    heeft geen authenticatie nodig en is voor lokale clients de juiste keuze.

---

# Geplande versus werkelijke status

## Afwijkingen van het plan

| Plan | Werkelijkheid | Waarom |
|---|---|---|
| `RetrievedDocument` uitbreiden met `section_heading`, `chunk_count`, `code_snippet` | **Niet gedaan** — `server/models.py` is onaangeraakt | `match_documents` levert die velden al; `service._fetch_rows` leest de ruwe RPC-rijen rechtstreeks. Dat geeft dezelfde provenance met nul wijzigingen in bestaande modellen |
| `vector_search` uitbreiden | **Niet gedaan** — `server/retriever.py` is onaangeraakt | Idem; `vector_search` wordt niet gebruikt door de MCP-laag |
| Filtering in Python met 4×/2× over-fetch | **Vervangen** door SQL-filtering in `match_documents_v2`; over-fetch verwijderd | De database kan het predicaat vóór `LIMIT` toepassen, wat het systematische tekort wegneemt in plaats van het te compenseren |
| 3 wijzigingen in bestaande bestanden | **1 additieve wijziging**: `server/supabase_client.py` +123/−0 | Alleen de nieuwe RPC-clientfunctie was nodig; `search_similar_documents` bleef ongemoeid zodat `/chat` niet raakt |
| Uitgebreide directorystructuur (`rules/`, 9 toolbestanden, `scrape_reference.py`, `build_function_catalog.py`) | **Minimale platte package**: 6 modules, 3 tools | `review_pinescript` en `explain_function` zijn uitgesteld; de bijbehorende mappen zijn toen nog niet nodig |
| `requirements.txt` splitsen in scraper/server/mcp | **Alleen `requirements-mcp.txt` toegevoegd**; `requirements.txt` onaangeraakt | Splitsen is een aparte refactor; de MCP-dependencies zijn nu gescheiden zonder bestaande manifests te raken |
| Fase 1 als 1,5 dag | **Uitgevoerd**, inclusief Fase 2 (migratie 0002 + filtering) | De filtering bleek de grootste correctheidswinst en was zonder database te implementeren |
| `match_threshold` "werkend maken" in de bestaande RPC | **Nieuwe functie `match_documents_v2`**; v1 bewust ongemoeid | v1 is het live contract voor `/chat`; het aanpassen zou bestaand gedrag veranderen |

## Gerealiseerde structuur

```
pinescript-docs-scraper/
├── pinescript_mcp/                    # NIEUW  709 regels
│   ├── __init__.py                    17
│   ├── __main__.py                    32   stdio entrypoint
│   ├── app.py                         63   MCPServer + registratie
│   ├── schemas.py                     85   DocHit / ExampleHit / Passage / ContextPack
│   ├── service.py                    329   retrieval, delegeert filtering naar SQL
│   └── tools.py                      183   de drie tools
├── tests/mcp/                         # NIEUW  955 regels
│   ├── conftest.py                   115   fake index + outbound guard
│   ├── test_mvp_tools.py             468   37 tests
│   ├── test_migration_0002.py        210   16 tests (SQL-contract)
│   └── test_search_documents_v2.py   162   13 tests (client)
├── migrations/
│   ├── 0001_create_documents_and_file_manifest.sql   ONGEWIJZIGD
│   └── 0002_match_documents_v2.sql    NIEUW  151
├── scripts/benchmark_filtering.py     NIEUW  388
├── docs/MCP.md                        NIEUW  330
├── docs/DEPLOYMENT.md                 +223 / −2  (MCP-deploysectie, 208 regels)
├── requirements-mcp.txt               NIEUW   24
├── requirements-dev.txt               NIEUW    6
└── server/supabase_client.py          +123 / −0 (439 regels)
```

---

## 0. Geverifieerde uitgangspunten

Alles in deze sectie is gecontroleerd, niet aangenomen. Deze feiten bepalen het plan.

| Feit | Bewijs | Gevolg voor het plan |
|---|---|---|
| **`mcp` 2.3.0** is de actuele SDK; v2 heeft `FastMCP` → **`MCPServer`** hernoemd | PyPI-metadata (release 2.3.0, 2026-10-02) + what's-new: *"`FastMCP` is now `MCPServer`… the old import path is gone rather than deprecated"* | Alle online tutorials met `from mcp.server.fastmcp import FastMCP` zijn **verouderd**. Gebruik `from mcp.server import MCPServer` |
| Transportopties horen op **`run()`**, niet op de constructor | `MCPServer("x", port=9000)` → `TypeError` | `host`/`port`/`transport_security` gaan naar `run()` of `streamable_http_app()` |
| **Synchrone `def`-handlers draaien op een worker thread** | Migration guide: *"Sync handlers run on a worker thread"* | **Gunstig**: de blocking OpenAI/Supabase-code kan ongewijzigd sync blijven, zonder de event loop te blokkeren |
| Dependency floors v2: `pydantic>=2.12`, `anyio>=4.9`, `typing-extensions>=4.13`, `sse-starlette>=3`, `opentelemetry-api>=1.28`, `httpx2>=2.5.0`, `mcp-types==<exact>` | Migration guide, dependency-tabel | Repo pint `pydantic>=2.10` → **moet omhoog**. De SDK gebruikt `httpx2`, dus de bestaande `httpx 0.28.1` van Supabase blijft conflictvrij |
| Gemounte ASGI-app vereist `mcp.session_manager.run()` in de **host**-lifespan | ASGI-docs, "Check"-callout | [`server/app.py`](./server/app.py) heeft al een `lifespan`; daar moet de manager in |
| Achter een echte hostname **moet** `TransportSecuritySettings` met allowlist | ASGI-docs: *"every request is rejected with 421 Misdirected Request"* | Verplicht item in de deploy-fase, anders faalt remote gebruik stil |
| Foutmodel: `ToolError` (model ziet het) vs `MCPError` (protocol) | SDK-docs "Handling errors" | Bepaalt de foutstrategie per tool |
| Return-annotatie = **output schema** | SDK-docs "Structured Output" | Tools retourneren Pydantic-modellen, geen `dict` |
| Testen kan in-memory: `Client(mcp)` | SDK-docs "Get started" | Geen subprocess of poort nodig in tests |

### Geverifieerde repo-eigenschappen

```
python (venv)                      3.12.10
mcp geïnstalleerd                  nee
pydantic (geïnstalleerd)           2.12.5   → voldoet aan mcp>=2.3 floor
httpx (geïnstalleerd)              0.28.1   → blijft nodig voor Supabase/TestClient
tests zonder .env                  29/78 gefaald
tests met dummy-env                8/78 gefaald
openstaande merge                  70 conflictpaden, MERGE_HEAD aanwezig
```

### ⚠️ Belangrijkste corpusbevinding

De corpusinhoud is getest, niet aangenomen:

```
total processed pages:              73
does corpus mention 'reference/functions'?   False
failed_urls file                    leeg (0 regels) → niets "faalde", de sectie is nooit gevonden
distinct namespaced functions *mentioned*    338
top: request.security 412×, color.new 392×, str.tostring 363×, ta.sma 186×
```

De 73 pagina's zijn **volledig de User Manual**. De **Language Reference** (`reference/functions/...`, `reference/variables/...`) is nooit gecrawld — de crawler loopt alleen nav-links vanaf `/pine-script-docs/welcome/` (zie [`1_scrap_docs.py`](./1_scrap_docs.py)) en die pagina's zitten achter een aparte, JS-gedreven boom.

**Concreet gevolg:** `explain_function` kan **niet** uit deze corpus worden gebouwd. De bestaande JS-server (`iamrichardD/mcp-server-pinescript`) adverteert 457 functies + 427 variabelen — die data komt uit precies de referentiepagina's die hier ontbreken. Dit is de enige tool die een **scraper-uitbreiding** vereist, niet alleen MCP-glue.

---

## 1. Directorystructuur

> **Niet meer actueel.** Dit is de oorspronkelijk geplande structuur, inclusief
> `rules/`, negen toolbestanden en de scraper-uitbreiding. De werkelijk gerealiseerde,
> veel kleinere structuur staat in
> [Gerealiseerde structuur](#gerealiseerde-structuur).

Nieuwe **top-level package** `pinescript_mcp/`, die `server/` importeert. Reden: `server/` blijft onaangeroerd (belangrijk gezien de open merge), en de MCP-server is los deploybaar met een eigen dependency-set.

```
pinescript-docs-scraper/
├── server/                          # ONGEWIJZIGD (op 3 kleine uitbreidingen na)
├── pinescript_mcp/                  # ← NIEUW
│   ├── __init__.py
│   ├── __main__.py                  # stdio entrypoint: python -m pinescript_mcp
│   ├── app.py                       # MCPServer-instantie + registratie + ASGI-mount
│   ├── service.py                   # ← service-laag: één retrieval-pad voor API én MCP
│   ├── formatting.py                # payload-verkleining, excerpting, provenance
│   ├── schemas.py                   # Pydantic output-modellen (= MCP output schema's)
│   ├── errors.py                    # ToolError/MCPError-mapping
│   ├── resources.py                 # pinescript:// resources
│   ├── prompts.py                   # @mcp.prompt() templates
│   └── tools/
│       ├── __init__.py              # register_all(mcp)
│       ├── search_docs.py
│       ├── find_examples.py
│       ├── get_context.py
│       ├── get_chunk.py             # ← tegenhanger van de kleine payloads
│       ├── explain_function.py
│       ├── review_pinescript.py
│       ├── optimize_strategy.py
│       └── find_repainting_risks.py
├── pinescript_mcp/rules/            # ← NIEUW: gedeelde rule engine
│   ├── __init__.py
│   ├── engine.py                    # Rule, Finding, run_rules()
│   ├── pine_parser.py               # lichtgewicht lexer/scope-detectie
│   └── rules/
│       ├── naming.py
│       ├── repainting.py            # voedt ZOWEL review als find_repainting_risks
│       ├── security_calls.py
│       └── performance.py
├── scripts/
│   ├── scrape_reference.py          # ← NIEUW: crawlt reference/functions|variables
│   └── build_function_catalog.py    # ← NIEUW: bouwt de catalogus
├── tests/mcp/                       # ← NIEUW
│   ├── test_tools_search.py
│   ├── test_tools_examples.py
│   ├── test_tools_review.py
│   └── test_server_smoke.py
├── requirements-mcp.txt             # ← NIEUW
├── requirements-server.txt          # ← refactor van requirements.txt
├── requirements-scraper.txt         # ← refactor van requirements.txt
└── docs/MCP.md                      # ← NIEUW: client-config per IDE
```

---

## 2. Nieuwe bestanden

> **Niet meer actueel.** Dit is de oorspronkelijk geplande omvang (±3.100 regels, inclusief
> de rule engine en de catalogus-ETL). De werkelijk gerealiseerde bestanden en hun
> aantallen staan in [Gerealiseerde infrastructuur](#gerealiseerde-infrastructuur) en
> [Gerealiseerde structuur](#gerealiseerde-structuur).

| Bestand | Doel | Regels (schatting) | Hergebruikt |
|---|---|---|---|
| `pinescript_mcp/__main__.py` | stdio entrypoint + `if __name__ == "__main__"` guard | ~15 | — |
| `pinescript_mcp/app.py` | `MCPServer(...)`, tool-registratie, `streamable_http_app()` | ~60 | — |
| `pinescript_mcp/service.py` | **Service-laag**: `search()`, `context()`, `examples()` | ~180 | `retriever`, `embed_client`, `supabase_client` |
| `pinescript_mcp/formatting.py` | excerpting, payload-cap, provenance-opbouw | ~120 | `utils.count_tokens` |
| `pinescript_mcp/schemas.py` | `DocHit`, `ExampleHit`, `ContextPack`, `Finding`, `FunctionInfo` | ~160 | `models.RetrievedDocument` |
| `pinescript_mcp/errors.py` | exception → `ToolError`/`MCPError` | ~40 | — |
| `pinescript_mcp/resources.py` | `pinescript://docs` (lijst), `pinescript://doc/{slug}` | ~80 | `supabase_client` |
| `pinescript_mcp/prompts.py` | review/optimize-prompttemplates | ~50 | `llm_client.build_system_prompt` |
| `pinescript_mcp/tools/*.py` (9×) | één tool per bestand | 40–200 elk | zie §5 |
| `pinescript_mcp/rules/engine.py` | `Rule`-protocol, `run_rules()`, severity | ~150 | — |
| `pinescript_mcp/rules/pine_parser.py` | regel-gebaseerde Pine-lexer (geen volledige parser) | ~200 | — |
| `pinescript_mcp/rules/rules/*.py` | ~15 regels, elk 20–50 regels | ~500 | — |
| `scripts/scrape_reference.py` | crawlt `reference/functions` + `reference/variables` | ~250 | patroon uit `1_scrap_docs.py` |
| `scripts/build_function_catalog.py` | signaturen/params/returns → JSON + embeddings | ~300 | `ingest.parse_document`, `embed_client` |
| `tests/mcp/*` | in-memory `Client(mcp)` tests | ~400 | bestaande teststijl |
| `requirements-mcp.txt` | MCP deps | ~10 | — |
| `docs/MCP.md` | config voor 5 clients | ~250 | — |

**Totaal nieuw: ±3.100 regels**, waarvan ~900 voor de rule engine en ~550 voor de catalogus-ETL.

---

## 3. Dependencies

`requirements-mcp.txt` (nieuw):

```
# Official Python MCP SDK (v2 API: MCPServer, not FastMCP)
mcp[cli]>=2.3,<3

# Transitive floors that the repo currently under-pins
pydantic>=2.12          # repo pins >=2.10 -> RESOLUTION FAILURE with mcp>=2
anyio>=4.9
typing-extensions>=4.13

# Shared with server/
openai>=1.3.0
supabase>=2.0.0
tiktoken>=0.5.0
tenacity>=8.2.0
python-dotenv>=1.0
```

Aandachtspunten:

- **`pydantic>=2.12` is verplicht.** [`requirements.txt`](./requirements.txt) zegt `pydantic>=2.10`. Met `mcp[cli]>=2.3` erbij faalt de resolver als die floor niet omhoog gaat. De venv heeft al 2.12.5, dus lokaal gaat het goed — in CI niet.
- De SDK vervangt `httpx`/`httpx-sse` door **`httpx2`**. De bestaande `httpx` (voor Supabase en `TestClient`) blijft nodig en conflicteert niet.
- `opentelemetry-api` en `mcp-types` komen automatisch mee. Pin `mcp-types` **nooit** zelf (exact-pinned aan de SDK-versie).
- `mcp[cli]` geeft de CLI: `mcp dev` (Inspector), `mcp run`, `mcp install`, `mcp version`.
- Python ≥3.10 vereist; de venv heeft 3.12.10.

---

## 4. Volledige MCP tool lijst

Ontwerpprincipe: **kleine payloads**. `RetrievedDocument.content` kan 17.832 tokens zijn; dat blaast het contextvenster van Cline/Roo/Copilot op. Alle zoektools geven daarom alleen `id`, `source`, `section`, `score`, `has_code` en een excerpt van ~300 tekens; volledige tekst komt via `get_chunk`.

| # | Tool | Input (type hints → schema) | Output (Pydantic → output schema) | Payload |
|---|---|---|---|---|
| 1 | `search_docs` | `query: str, top_k: int = 8, source: str \| None, min_score: float = 0.0, include_code: bool = False` | `list[DocHit]` | ~2 KB |
| 2 | `find_examples` | `topic: str, language_version: str = "6", limit: int = 5` | `list[ExampleHit]` met `code` + `caption` | ~4 KB |
| 3 | `get_context` | `query: str, token_budget: int = 8000, top_k: int = 12` | `ContextPack` (doc-id's + scores + budgetgebruik) | ~3 KB |
| 4 | `get_chunk` | `doc_id: str` | `ChunkFull` (volledige content) | tot 40 KB |
| 5 | `explain_function` | `name: str, namespace: str \| None, version: str = "6"` | `FunctionInfo` (signatuur, params, returns, voorbeeld, beschikbaarheid) | ~3 KB |
| 6 | `review_pinescript` | `code: str, severity: str = "warning", include_suggestions: bool = True` | `ReviewReport` (`list[Finding]`) | ~3 KB |
| 7 | `optimize_strategy` | `code: str, goal: str \| None, include_references: bool = True` | `OptimizationReport` | ~5 KB |
| 8 | `find_repainting_risks` | `code: str` | `list[Finding]` met regelnummers + waarom het repaint | ~2 KB |
| 9 | *(meta)* `list_sources` | — | `list[SourceSummary]` | ~1 KB |

Plus niet-tool primitieven die clients gratis krijgen:

- **Resources**: `pinescript://docs` (catalogus), `pinescript://doc/{slug}` (volledige pagina), `pinescript://function/{name}` (na fase 4).
- **Prompts**: `pinescript-review`, `pinescript-optimize` (voor slash-commands in Claude Code/Copilot).
- **Context**: `ctx: Context` parameter voor logging/progress. In v2 is `get_context()` **verwijderd** — declareer `ctx: Context`.

---

## 5. Mapping naar bestaande code

Legenda: **✅ ongewijzigd** · **🔧 kleine wijziging** · **❌ nieuw**

### 5.1 Tools

| Tool | Herbruikt | Nieuw te schrijven |
|---|---|---|
| `search_docs` | ✅ `embed_client.generate_single_embedding`<br>✅ `retriever.vector_search`<br>🔧 `supabase_client.search_similar_documents`<br>🔧 `retriever.hybrid_search` | payload-shaping, `min_score`-filter (client-side zolang de RPC geen threshold heeft), `include_code`-filter |
| `find_examples` | ✅ `utils.extract_code_blocks`<br>✅ `utils.detect_code_snippets`<br>✅ `vector_search` | extractie + ranking van codeblokken, captioning |
| `get_context` | ✅ `retriever.assemble_context`<br>✅ `retriever._trim_content_for_budget`<br>✅ `utils.estimate_prompt_tokens` | budget-parameter-mapping |
| `get_chunk` | 🔧 `supabase_client` (nieuwe `fetch_document_by_id`) | nieuwe queryfunctie + RLS-check |
| `explain_function` | ✅ `vector_search` (alleen prose)<br>⚠️ `scripts/md_to_json.py` genereert **lege** `functions: {}` | **volledige catalogus-ETL** + scraper-uitbreiding (§0) |
| `review_pinescript` | ❌ niets | rule engine + ~15 regels |
| `optimize_strategy` | ✅ `search_docs`-pad<br>✅ `llm_client.create_chat_completion` | prompt, doel-interpretatie |
| `find_repainting_risks` | ✅ repainting-pagina via `vector_search` | heuristieken (deelt engine met `review_pinescript`) |

### 5.2 Wijzigingen in bestaande bestanden (3 gerichte ingrepen)

| Bestand | Wijziging | Waarom onvermijdelijk |
|---|---|---|
| [`server/models.py`](./server/models.py) | **`RetrievedDocument` uitbreiden** met `section_heading: str \| None`, `chunk_count: int`, `code_snippet: bool` | De RPC retourneert deze velden wél, maar `vector_search` gooit ze weg. Zonder `section_heading` heeft **geen enkele** MCP-tool bruikbare provenance |
| [`server/retriever.py`](./server/retriever.py) | Mapping in `vector_search` uitbreiden met diezelfde 3 velden | Idem |
| [`migrations/`](./migrations) | Nieuwe `0002_search_rpc_filters.sql`: `match_documents` **met werkende `match_threshold`** + filters (`p_source`, `p_code_only`) | De huidige RPC negeert `match_threshold` volledig en kan niet filteren. Elke MCP-tool heeft filtering nodig |

**Niet aanraken** (buiten scope, wel blokkerend — zie §8): de index-integriteitsbug, blocking ingest, BM25.

### 5.3 Hergebruik expliciet *niet* mogelijk — en waarom

| Wat | Probleem | Gevolg |
|---|---|---|
| `models.Source` als MCP-output | `similarity_score: float = Field(..., ge=0.0, le=1.0)` — ruwe cosine kan **negatief** zijn | Gebruik een **nieuw** `DocHit`-model; hergebruik van `Source` geeft `ValidationError` |
| `retriever.bm25_search` voor hybrid | Supabase `text_search` retourneert geen `rank` → score altijd 0 → `hybrid_bm25_weight=0.3` doet niets | In v1 **vector-only** shippen; hybrid pas na de BM25-fix (§8) |
| `documents.code_snippet` als filter | 531/594 chunks = `true` (89%) → geen discriminerend vermogen | Filter op basis van `extract_code_blocks()`-resultaat, niet op de vlag |
| `/chat`-pipeline in [`server/app.py`](./server/app.py) | Pipeline zit ín de endpoint-handler, niet in een service | Verplaats naar `pinescript_mcp/service.py` zodat API en MCP hetzelfde pad delen; anders divergeren ze |

---

## 6. Implementatievolgorde

### Fase 0 — Ontgrendelen (0,5 dag) — *harde voorwaarde*

1. Open merge afronden: 70 conflictpaden + `UU processed_all_docs.md`. Zonder dit kan er niet normaal gecommit worden.
2. `requirements.txt` splitsen in scraper/server/mcp; `pydantic>=2.12`.
3. Package-skelet + `mcp dev` rooktest met één triviale tool.

### Fase 1 — Read-only MVP (1,5 dag)

4. `schemas.py` + `formatting.py` + `errors.py`.
5. `service.py` met het gedeelde retrieval-pad.
6. `search_docs` → `get_context` → `get_chunk` → `find_examples`.
7. Test met `Client(mcp)` in-memory; `mcp dev` voor handmatige inspectie.

→ **Levert 4 werkende tools over stdio**, bruikbaar in Claude Code, Copilot, Roo en Cline.

### Fase 2 — Retrieval-kwaliteit (1–2 dagen)

8. Migratie `0002`: RPC met werkende `match_threshold` + filters; RLS erbij.
9. `RetrievedDocument` uitbreiden; `vector_search`-mapping.
10. BM25 repareren (`tsvector` + GIN + `ts_rank_cd`) → hybrid écht aanzetten.
11. `min_score`-drempel + absolute confidence (niet de huidige deling door max, die de top-hit altijd 1.0 maakt).

### Fase 3 — Rule engine + domeintools (3–5 dagen)

12. `rules/pine_parser.py` + `engine.py`.
13. ~15 regels: naming (`shorttitle`-lengte, `precision`), `request.security` zonder `lookahead`, `varip`, `ta.pivothigh` op hogere TF, `barstate.isconfirmed`, `strategy.*` zonder risk-management.
14. `review_pinescript` → `find_repainting_risks` (zelfde engine) → `optimize_strategy`.

### Fase 4 — Functiecatalogus (3–5 dagen) — *vereist scraper-uitbreiding*

15. `scripts/scrape_reference.py`: crawlt `reference/functions/*` + `reference/variables/*`.
16. `scripts/build_function_catalog.py`: signaturen/params/returns → JSON + embeddings → nieuwe tabel.
17. `explain_function` + resource `pinescript://function/{name}`.

### Fase 5 — Deploy & clients (1–2 dagen)

18. `streamable_http_app()` mounten in de bestaande FastAPI-app; **`mcp.session_manager.run()` in de host-lifespan**.
19. `TransportSecuritySettings(allowed_hosts, allowed_origins)` — anders 421 bij elke remote request.
20. CORS incl. `allow_headers` met `Mcp-*` en `expose_headers=["Mcp-Session-Id"]`.
21. Auth voor remote; apart image; `docs/MCP.md` met configs voor 5 clients.

---

## 7. Tijdschatting

> **Uitkomst.** Fase 0 en 1 zijn uitgevoerd, en Fase 2 is grotendeels meegenomen: de
> migratie met `match_documents_v2` en de omzetting naar database-side filtering zijn
> gerealiseerd, inclusief benchmark en contracttests. Fase 3 en 4 zijn niet gestart
> (conform afspraak uitgesteld); Fase 5 is gedocumenteerd in
> [docs/MCP.md](./docs/MCP.md) maar niet uitgerold. Het resterende werk zit in
> [Aanbevolen vervolgstappen](#aanbevolen-vervolgstappen).

| Fase | Scope | Tijd |
|---|---|---|
| 0 | Merge + packaging | 0,5 dag |
| 1 | 4 read-only tools, stdio | **1,5 dag** |
| 2 | Filters, threshold, BM25, RLS | 1–2 dagen |
| 3 | Rule engine + 3 domeintools | 3–5 dagen |
| 4 | Scraper + catalogus + `explain_function` | 3–5 dagen |
| 5 | HTTP-mount, transport security, auth, docs | 1–2 dagen |
| | **Totaal** | **≈10–16 dagen** |

**Bruikbare MCP-server: 1,5 dag. Alles: ~2–3 weken.**

---

## 8. Risico's en beslissingen

| # | Risico | Impact | Mitigatie |
|---|---|---|---|
| R1 | **Open merge blokkeert elke commit** (70 paden) | Blokkerend | Fase 0 vóór alles |
| R2 | **`pydantic>=2.10` in `requirements.txt` vs `mcp` floor 2.12** | Resolutiefout in CI | Floor optrekken; split requirements |
| R3 | **`explain_function` heeft geen databron** (referentie niet gecrawld) | Verkeerde verwachting | Communiceer expliciet; plan de scraper-uitbreiding vooraf in |
| R4 | **Payload-grootte** (17k-token chunks) | Context-venster klapt dicht | Klein-by-default + `get_chunk` voor volledige tekst |
| R5 | **58% kans op indexvervuiling per scrape-cyclus** (filenaam-timestamp + geen delete-detectie) | Tools antwoorden uit verouderde duplicaten | Fix vóór publiek gebruik; anders serveer je rommel |
| R6 | **Hybrid search levert verkeerde scores** (BM25 altijd 0) | Slechte ranking in `search_docs` | v1 vector-only, hybrid pas na fase 2 |
| R7 | **Remote auth**: huidige `/api/login` is een e-mailadres-only tokenfabriek | Onacceptabel publiek | stdio heeft géén auth nodig — begin daar; remote pas met OAuth 2.1/PKCE |
| R8 | **421 Misdirected Request** bij deploy | "Werkt lokaal niet remote" | `TransportSecuritySettings` in fase 5 |
| R9 | **Sync retry met `time.sleep`** in `search_similar_documents` blokkeert een worker thread tot ~3 s | Vertraging bij upstream-storingen | Bewust accepteren (worker thread, niet de loop); timeouts toevoegen |

### Twee aanbevolen ontwerpkeuzes

1. **Één service-laag.** Verplaats de retrieval-pipeline uit [`server/app.py`](./server/app.py) naar `pinescript_mcp/service.py` en laat zowel `/chat` als alle MCP-tools daar doorheen gaan. Anders ontstaan twee implementaties die uit elkaar lopen — precies het patroon dat deze repo al heeft met `auth.py` versus `app.py`.
2. **Één rule engine, twee tools.** `review_pinescript` en `find_repainting_risks` delen ~80% van hun logica. Door de engine eerst te bouwen (stap 12–13) leveren twee tools bijna de prijs van één. Zonder die keuze betaal je de regels twee keer.

---

## 9. Verificatie

Rooktest na fase 1 (geen poort, geen subprocess):

```python
import pytest
from mcp import Client
from pinescript_mcp.app import mcp

@pytest.mark.anyio
async def test_search_docs_returns_hits():
    async with Client(mcp) as client:
        result = await client.call_tool("search_docs", {"query": "ta.sma moving average", "top_k": 3})
        assert result.is_error is False
        assert len(result.structured_content["result"]) <= 3
```

Handmatige inspectie:

```bash
mcp dev pinescript_mcp/app.py     # MCP Inspector
mcp run pinescript_mcp/app.py --transport streamable-http --port 8000
```

Controle dat de pydantic-floor klopt (moet ≥2.12 zijn):

```bash
pip install -r requirements-mcp.txt
python -c "import pydantic, mcp.version; print(pydantic.VERSION)"
```
