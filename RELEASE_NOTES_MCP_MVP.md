# MCP MVP Release

| | |
|---|---|
| **Onderdeel** | PineScript MCP Server (`pinescript_mcp/`) |
| **Versie** | 0.1.0 |
| **Basisrevisie** | `49352fe` |
| **Datum** | 13 oktober 2026 |
| **Status** | Werkend, getest, **niet gecommit** en **niet uitgerold op een echte database** |
| **SDK** | officiële Python MCP SDK `mcp` 2.3.0, v2-API (`MCPServer`) |

## Samenvatting

Deze release voegt een read-only MCP-server toe bovenop de bestaande PineScript
RAG-stack. Drie tools stellen de geïndexeerde Pine Script v6-documentatie beschikbaar
aan MCP-clients zoals GitHub Copilot Agent, Claude Code, Roo Code en Cline:
`search_docs`, `get_context` en `find_examples`.

De belangrijkste technische wijziging zit niet in de MCP-laag maar in de database: de
relevantedrempel en de filters worden nu door PostgreSQL toegepast via een nieuwe
functie `match_documents_v2`, in plaats van in Python nadat er te veel rijen waren
opgehaald. Dat lost een correctheidsprobleem op — een selectief filter kon minder
resultaten geven dan gevraagd — en reduceert de hoeveelheid overgedragen data met
**92,9%** (gemeten).

De bestaande FastAPI-applicatie is niet gewijzigd. `POST /chat` blijft exact werken:
de oude RPC `match_documents` is byte-for-byte onaangeraakt gelaten en de enige
wijziging in `server/` is een additieve functie (+123 regels, nul verwijderingen).

**Belangrijkste voorbehoud:** migratie 0002 is geschreven en statisch gevalideerd, maar
**nooit uitgevoerd op een echte database**. Er was in de ontwikkelomgeving geen Docker,
geen `psql` en geen Postgres. De tools zijn nooit end-to-end tegen een echte index
gedraaid. De eerste uitvoering op een echte database is daarmee ook de eerste echte test.

## Nieuwe functionaliteit

### MCP-server

- Nieuwe top-level package `pinescript_mcp/` (6 modules, 709 regels) met een
  `MCPServer`-instantie, toolregistratie, outputmodellen en een stdio-entrypoint.
- Twee transportvormen: `python -m pinescript_mcp` voor stdio (lokaal, geen
  authenticatie nodig) en een gedocumenteerde mount van de ASGI-app in de bestaande
  FastAPI-app voor remote gebruik.
- Serverinstructies die de client vertellen wanneer welke tool te gebruiken en dat de
  index de User Manual is, niet de Language Reference.
- Gestructureerde output: de return-annotatie van elke tool is het outputschema, zodat
  clients zowel tekst als een JSON-object krijgen.

### Database

- **`match_documents_v2`** met een werkende `match_threshold` (similariteit, hoger is
  beter), `filter_source` (letterlijke case-insensitive substring) en
  `filter_code_only`.
- De `embedding`-kolom wordt niet meer teruggegeven: 71,5% van de bytes per rij werd
  door niemand gelezen.
- `REVOKE EXECUTE ... FROM PUBLIC` plus `GRANT EXECUTE ... TO service_role`, zodat de
  nieuwe functie niet — zoals PostgreSQL standaard doet — publiek aanroepbaar wordt.

### Client en tooling

- `server/supabase_client.search_documents_v2()` (additief) met een retrybeleid dat
  onderscheid maakt tussen een onbereikbare database en een niet-geïnstalleerde
  migratie. Het laatste geeft `MigrationRequiredError` na één poging in plaats van drie.
- `scripts/benchmark_filtering.py`: vergelijkt de oude en nieuwe aanpak, onderscheidt
  expliciet gemeten van gemodelleerde getallen, en kan met `--live` tegen een echte
  Supabase worden gedraaid.
- `docs/MCP.md`: installatie, migratietabel, en configuratie voor vier clients.
- `docs/DEPLOYMENT.md`: nieuwe sectie **MCP server deployment** met de verplichte
  uitrolvolgorde (migratie 0002 vóór de code), vier verificatiequeries en een
  rollbackprocedure.
- `requirements-mcp.txt`: gescheiden MCP-dependencies met de door de SDK verhoogde
  floors.
- `requirements-dev.txt`: optionele ontwikkeldependency (`pglast>=8.5`) voor de
  SQL-parservalidatie.

## Geïmplementeerde MCP tools

### `search_docs`

Zoekt chunks en geeft provenance plus een excerpt.

| Parameter | Type | Default | Betekenis |
|---|---|---|---|
| `query` | str | — | Zoekopdracht, natuurlijke taal of code |
| `top_k` | int | 8 | 1–50 resultaten |
| `min_score` | float | -1.0 | Cosine-ondergrens; -1.0 betekent geen filter |
| `source` | str? | null | Substring van de paginanaam, bv. `strategies` |
| `include_code` | bool | false | Alleen chunks met de code-vlag |

Output: `doc_id`, `source`, `section`, `chunk_index`, `chunk_count`, `score`,
`has_code`, `tokens`, `excerpt` (≤ 500 tekens), `truncated`.

### `get_context`

Assembleert volledige passages binnen een tokenbudget.

| Parameter | Type | Default | Betekenis |
|---|---|---|---|
| `query` | str | — | Onderwerp |
| `token_budget` | int | 8000 | 256–20000, extra begrensd door `PROMPT_TOKEN_BUDGET` |
| `top_k` | int | 12 | Beschouwde chunks vóór trimming |

Output: `query`, `passages[]` (met `content`, `doc_id`, `source`, `section`, `score`,
`tokens`), `total_tokens`, `budget`, `documents_considered`.

### `find_examples`

Levert kopieerbare codevoorbeelden.

| Parameter | Type | Default | Betekenis |
|---|---|---|---|
| `topic` | str | — | Wat het voorbeeld moet demonstreren |
| `limit` | int | 5 | 1–20 voorbeelden |
| `min_score` | float | -1.0 | Cosine-ondergrens van de bevattende chunk |

Output per hit: `doc_id`, `source`, `section`, `score`, `language`, `line_count`,
`truncated`, `code` (≤ 2000 tekens).

## Verbeteringen

| Verbetering | Effect | Aard |
|---|---|---|
| Filtering naar de database verplaatst | Een selectief filter kan niet langer minder resultaten geven dan gevraagd, doordat het predicaat vóór `LIMIT` wordt toegepast | gemeten |
| Over-fetch verwijderd | 32 → 8 rijen per zoekopdracht bij `top_k = 8`; 12 → 3 bij `top_k = 3` | gemeten |
| Embedding-kolom niet meer in de response | 71,5% minder bytes per rij | gemeten |
| Totale payloadreductie | 92,9% minder bytes per query (4× rijen × 3,5× per rij) | gemeten |
| Toolcontracten vastgezet | Een contract-lock test faalt bij elke wijziging van eigenschappen, `required`, defaults of outputmodellen | — |
| Testsuite hermetisch | Elke niet-gestubde outbound call faalt expliciet; de suite raakt het netwerk niet meer en ging van ~30 s naar ~5,5 s | gemeten |
| Nieuwe functie niet publiek aanroepbaar | Voorkomt een tweede instantie van SEC-01 | — |

## Opgeloste bugs

Vijf fouten gevonden en opgelost tijdens de implementatie. De volledige analyse staat
in [implementatieplan_mcp.md](./implementatieplan_mcp.md#gevonden-bugs).

| Bug | Kern | Impact |
|---|---|---|
| **`min_score`-default** | Default `0.0` gooide alle negatieve cosine-scores weg, terwijl cosine in `[-1, 1]` ligt | Stil dataverlies: minder resultaten zonder foutmelding, onzichtbaar in logs. Default naar `-1.0` |
| **Regex-signatuur** | De testhelper las een `DROP FUNCTION ...`-vermelding in commentaar als de functiesignatuur | De drift-test tussen SQL en client kon een echte drift goedkeuren. Opgelost met commentaarstripping en verankering op `FUNCTION <naam>(` |
| **ILIKE-assertie** | `assert "ILIKE" not in body` faalde op een *commentaar* dat uitlegt waarom `strpos` wordt gebruikt | Vals-positieve testfout die uitnodigde tot het "repareren" van correcte code. Nu wordt op code getest, plus een assertie op het ontbreken van `\|\|` |
| **Outbound OpenAI-call** | Een test stubbde de RPC maar niet de embedding, en deed dus een echte call met de placeholder-key | Netwerkafhankelijke, trage test die faalde om de verkeerde reden én een uitgaand verzoek lekte. Nu een autouse guard die elke vergeten stub hard laat falen |
| **Benchmarklabel** | `1 - new/legacy` werd gepresenteerd als het embedding-aandeel, maar is de totale besparing | Verkeerd getal in de onderbouwing van een ontwerpbeslissing. Nu twee gescheiden grootheden: 71,5% per rij en 92,9% totaal |

## Testresultaten

```
tests/mcp                     66 passed, 0 failed        ~5,5 s
  test_mvp_tools.py           37 passed
  test_migration_0002.py      16 passed
  test_search_documents_v2.py 13 passed

tests/mcp  (zonder pglast)    65 passed, 1 skipped, 0 failed
  test_migration_0002.py      15 passed, 1 skipped

server/tests                  8 failed, 68 passed, 2 skipped   ~30 s
```

66 passed met `pglast` geïnstalleerd; 65 passed en 1 skipped zonder dat pakket. Het
aantal failures is nul in beide gevallen — de SQL-parservalidatie is de enige test die
van een optionele dependency afhangt. Die dependency staat in `requirements-dev.txt`.

De 8 server-failures zijn **ongewijzigd ten opzichte van de baseline** van vóór deze
wijziging: identieke testnamen en aantallen. Het zijn vooraf bestaande problemen
(stale 501-verwachting, twee live-Supabase-tests zonder credentials, en de
`logger`-shadowingbug in `llm_client`), geen regressie.

Wat de MCP-suite aantoont: toolregistratie en -schema's, de bevroren toolcontracten,
gedrag per tool, dat filtering aan de database wordt gedelegeerd, dat er geen
over-fetch meer is, foutafhandeling zonder interne details te lekken, en dat
`python -m pinescript_mcp` als echt subprocess start en de drie tools adverteert.

## Bekende beperkingen

1. **Migratie 0002 is nooit uitgevoerd.** Geen enkele query is tegen een echte
   Postgres gedraaid. Of de query de juiste rijen teruggeeft, of de HNSW-index wordt
   gebruikt en wat de latency is, is onbekend. Statische validatie met de echte
   PostgreSQL-parser (`pglast`) dekt syntax, niet semantiek.
2. **Uitrolkoppeling.** De code vereist 0002: tegen een database zonder de migratie
   mislukken alle drie de tools met een `MigrationRequiredError`. Dit is de eerste
   wijziging in dit project waarbij code en schema samen uitgerold moeten worden.
3. **Geen RLS.** SEC-01 uit [SECURITY-REVIEW.md](./SECURITY-REVIEW.md) staat nog open:
   de hele index is leesbaar en beschrijfbaar met de publieke anon-key. Bewust
   uitgesteld; alleen de nieuwe functie is afgeschermd.
4. **De tools hebben geen authenticatie en geen rate limiting** (SEC-21). Over stdio is
   dat geen blootstelling; via Streamable HTTP is het een ongeauthenticeerde leesroute.
5. **`search_docs` levert alleen een excerpt.** Er is nog geen `get_chunk`; wie de
   volledige chunk wil moet `get_context` gebruiken, wat duurder is dan nodig.
6. **Hybrid search staat uit.** `bm25_search` leest een `rank`-veld dat de client nooit
   teruggeeft, dus de score is altijd 0. De tools gebruiken puur vector search.
7. **Geen re-ranking of diversiteit.** De context kan near-duplicates bevatten.
8. **`find_examples` scant 2× het gevraagde aantal rijen**, omdat `code_snippet` ook
   waar is voor chunks met alleen inline code. Die factor is niet met echte data
   gekalibreerd.
9. **De index vervuilt per scrape-cyclus** (PROD-01): bestandsnamen bevatten een verse
   timestamp, dus elke scrape-ronde voegt nieuwe duplicaten toe zonder de oude op te
   ruimen. De tools antwoorden daardoor op termijn uit verouderde duplicaten.
10. **De index dekt alleen de User Manual**, niet de Language Reference. Er zijn dus
    geen functiesignaturen beschikbaar.

## Openstaande werkpunten

| # | Werkpunt | Prioriteit |
|---|---|---|
| 1 | Migratie 0002 uitvoeren op een testproject en verifiëren met echte data (rijcounts, `EXPLAIN ANALYZE`) | P1 |
| 2 | ~~Uitrolvolgorde code/migratie vastleggen in `docs/DEPLOYMENT.md`~~ **gedaan**; resterend: de volgorde ook in CI afdwingen | P1 |
| 3 | De open merge afronden (70 conflictpaden) zodat er gecommit kan worden | P1 |
| 4 | RLS aanzetten plus `REVOKE EXECUTE` op `match_documents` (SEC-01) | P2 |
| 5 | Hybrid search repareren (`tsvector` + GIN + `ts_rank_cd`) of expliciet uitzetten | P2 |
| 6 | `get_chunk` toevoegen en `RetrievedDocument` uitbreiden | P2 |
| 7 | Re-ranking en diversiteit (MMR) toevoegen | P2 |
| 8 | Timeouts op de OpenAI- en Supabase-clients (SEC-07) | P2 |
| 9 | `review_pinescript` en `find_repainting_risks` op één rule engine | P3 |
| 10 | `explain_function`, na een scraper-uitbreiding voor de Language Reference | P3 |
| 11 | Repositoryhygiëne: `log/`, `__pycache__`, dubbele docs-generatie, requirements splitsen met lockbestand (`requirements-dev.txt` bestaat nu; `requirements.txt` nog niet gesplitst) | P3 |
| 12 | Streamable HTTP met authenticatie, alleen wanneer remote gebruik nodig is | P3 |

## Aanbevolen volgende sprint

**Hoofddoel: de migratie bewijzen en de release vastleggen.**

1. **Merge afronden en committen** (0,5 dag). Zonder dit staat al het werk untracked en
   is het niet herstelbaar. Zie [COMMIT_PREVIEW.md](./COMMIT_PREVIEW.md).
2. **Migratie 0002 uitvoeren op een testproject** (0,5 dag). Verifiëren met vier
   queries: geen filter, `source`-filter, `code_only`, en een hoge drempel. Plus
   `EXPLAIN ANALYZE` om te zien of de HNSW-index daadwerkelijk wordt gebruikt. Dit is
   de enige manier om de belangrijkste onbekende weg te nemen.
3. **De drie tools end-to-end draaien tegen die index** (0,5 dag), via `mcp dev` en via
   één echte client. Tot nu toe is de retrieval altijd gestubd geweest.
4. **`find_examples`-scanfactor kalibreren** met echte data (0,5 dag): meet hoeveel
   chunks met `code_snippet = true` daadwerkelijk een fenced block opleveren, en stel
   de factor daarop af in plaats van op een aanname.
5. **SEC-01 (RLS) oppakken** (0,5 dag) als er ruimte is. Dit is de laatste
   deployment-blocker en het is een kleine, geïsoleerde wijziging in een nieuwe
   migratie.

Daarna, in de volgende sprint: hybrid search repareren of verwijderen, `get_chunk`
toevoegen, en re-ranking. Die drie bepalen samen of de retrieval kwalitatief goed
genoeg is voor agentgebruik, wat de huidige MVP nog niet heeft bewezen.
