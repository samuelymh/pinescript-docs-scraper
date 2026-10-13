# Architecture Status — PineScript RAG & MCP

*Stand: 13 oktober 2026, revisie `49352fe` plus de MCP-MVP-wijzigingen. Dit document
beschrijft de volledige pijplijn van scraper tot MCP-tool: wat werkt, hoe volwassen het
is, welke technische schuld er ligt, en wat er als volgende verbeterd moet worden.*

## Pijplijnoverzicht

```
Scraper → Processing → Chunking → Embeddings → Supabase → Retrieval → Context Assembly → MCP Tools
  ✅          ⚠️          ⚠️          ✅           ⚠️          ✅            ⚠️              ✅
 6/10       4/10        4/10        7/10         5/10        7/10          6/10            7/10
```

| Stap | Status | Volwassenheid | Grootste tekortkoming |
|---|---|---|---|
| Scraper | ✅ draait | 6/10 | Ziet de Language Reference niet; afhankelijk van de navigatie-DOM |
| Processing | ⚠️ fragiel | 4/10 | Vernietigt bron-URL's en dupliceert codeblokken |
| Chunking | ⚠️ rudimentair | 4/10 | Alleen H1/H2; 18 chunks boven de modellimiet; 89% code-vlag |
| Embeddings | ✅ solide | 7/10 | Retryt op niet-herhaalbare fouten; geen modelversie per chunk |
| Supabase | ⚠️ deels | 5/10 | Geen RLS; upsert in één request; geen tekstindex |
| Retrieval | ✅ correct | 7/10 | Geen re-ranking, geen RRF, hybrid staat uit |
| Context Assembly | ⚠️ bruikbaar | 6/10 | Stopt bij de eerste chunk die niet past; geen deduplicatie |
| MCP Tools | ✅ werkt | 7/10 | Alleen excerpts; geen `get_chunk`; nooit end-to-end gedraaid |

---

## 1. Scraper

**Bestand:** [1_scrap_docs.py](./1_scrap_docs.py) · **Status:** ✅ draait, in CI elke 3 dagen

### Wat het doet

`crawl4ai` met Playwright (headless Chromium) haalt `/pine-script-docs/welcome/` op,
loopt de navigatielinks af, en schrijft per pagina een markdown-bestand naar
`pinescript_docs/unprocessed/` plus een gecombineerd `all_docs_{timestamp}.md`. De
orkestrator [3_scrap_and_process.py](./3_scrap_and_process.py) draait crawler en
processor in één CI-stap.

Resultaat op de huidige corpus: **73 pagina's**, `failed_urls_*.txt` is leeg (0 regels),
dus de crawl zelf faalt niet.

### Volwassenheid: 6/10

Werkt betrouwbaar voor wat het doet, maar de dekking is een aanname geworden. De
crawler loopt uitsluitend `nav`- en `sidebar`-links af vanaf de welkomstpagina.

### Technische schuld

| Schuld | Gevolg |
|---|---|
| **De Language Reference wordt nooit gecrawld** | De corpus bestaat volledig uit de User Manual. `reference/functions/*` en `reference/variables/*` zitten achter een aparte, JS-gedreven boom die de navigatie-wandeling niet bereikt. Er zijn dus **geen functiesignaturen** beschikbaar, terwijl de corpus 338 functienamen wél in prose noemt |
| Selectors zijn DOM-afhankelijk (`main > div`, `nav`, `sidebar`) | Een herontwerp van de TradingView-documentatiesite breekt de crawl stil: er komen minder bestanden uit, zonder foutmelding |
| Geen robuuste foutafhandeling per pagina | `failed_urls` wordt geschreven, maar een gedeeltelijke crawl is niet te onderscheiden van een volledige |
| `content_schema` en `structure_schema` worden gedefinieerd maar de LLM-extractiestrategie wordt niet gebruikt | Dode configuratie die suggereert dat er meer extractie plaatsvindt dan in werkelijkheid |
| Geen content-hash of skip-logica | Elke ronde crawlt alles opnieuw; het onderscheiden van "gewijzigd" en "ongewijzigd" gebeurt pas in de ingest-stap, op bestandsnaam — die per ronde verandert |

### Aanbevolen verbeteringen

1. **De Reference-sectie toevoegen** als expliciete crawl-set (of via de API/JSON die de
   site zelf gebruikt). Dit is een voorwaarde voor `explain_function`.
2. Selectors valideren met een kanarie-assertie: verwacht minimaal N pagina's en
   minimaal M codeblokken, en laat CI falen als dat niet gehaald wordt.
3. Een stabiele paginacrawl-id (slug uit de URL) vastleggen, los van de bestandsnaam.
4. Per pagina een content-hash opslaan zodat een ronde alleen gewijzigde pagina's hoeft
   te verwerken.

---

## 2. Processing

**Bestand:** [2_process_docs.py](./2_process_docs.py) · **Status:** ⚠️ werkt, fragiel

### Wat het doet

Regex-gebaseerde opschoning van de ruwe markdown: navigatie verwijderen, markdown-links
platslaan tot hun weergavetekst, secties extraheren via `##`-koppen, en codeblokken
normaliseren.

### Volwassenheid: 4/10

Dit is de zwakste schakel. De transformatie is destructief: er wordt informatie
weggegooid die later niet meer te reconstrueren is.

### Technische schuld

| Schuld | Gevolg |
|---|---|
| **`remove_markdown_links()` verwijdert álle links** | De bron-URL van elke pagina is onherstelbaar verdwenen. Er is dus **geen provenance naar de originele TradingView-pagina** in de database, en geen enkele manier om een citaat te verifiëren. Dit is een ontwerpfout in de ETL, niet alleen een ontbrekend veld |
| **Codeblokken worden twee keer opgenomen** | Ze blijven in de sectietekst staan én worden nog eens toegevoegd onder `## Code Examples`. Dit verklaart mede waarom 89% van de chunks de code-vlag heeft, en het verspilt embeddingcapaciteit aan duplicaten |
| Secties worden gefilterd op keywords (`pine`, `script`, `function`, `indicator`, `value`, `parameter`) | Een sectie zonder die woorden verdwijnt, ook als de inhoud relevant is. Er is een fallback naar het hele document, dus het faalt zacht in plaats van hard |
| Verse timestamp in elke bestandsnaam | Dit is de oorzaak van PROD-01: de ingest ziet elk bestand als *nieuw*, en verdwenen bestanden worden nooit opgeruimd. De index vervuilt daardoor per scrape-cyclus |
| Geen tests | Er bestaat geen test voor `2_process_docs.py`. Regressies in de regexes zijn onzichtbaar |

### Aanbevolen verbeteringen

1. **Bron-URL behouden** in de front matter of als metadata-veld, en die in
   `documents.metadata` opnemen zodat retrieval echte provenance kan tonen.
2. **Codeblokken niet dupliceren**: of in de sectie laten, of apart opnemen — niet beide.
3. **Stabiele bestandsnamen** (slug zonder timestamp) zodat de manifest-diff werkt.
4. Tests toevoegen met een vaste invoerpagina en een verwachte uitvoer.

---

## 3. Chunking

**Bestanden:** [server/ingest.py](./server/ingest.py) (`chunk_document`,
`extract_token_overlap`, `split_text_by_token_limit`) · **Status:** ⚠️ rudimentair

### Wat het doet

Documenten boven 1500 tokens worden gesplitst op H1/H2-koppen, met een overlap van 150
tokens die als `... <vorige inhoud>` aan de volgende chunk wordt voorgezet. Chunks die
daarna nog boven de embeddingmodellimiet (8192) uitkomen, worden hard gesplitst met een
teken-heuristiek van 3 tekens per token.

### Gemeten resultaat op de huidige corpus

```
73 bestanden → 594 chunks (na kop-chunking)
18 chunks > 8192 tokens (max 17.832) → opnieuw gesplitst
531 van 594 chunks met code-vlag (89%)
584 van 594 chunks met een sectiekop
0 duplicate id's
```

### Volwassenheid: 4/10

Het verdeelt tekst, maar niet op een manier die retrieval ondersteunt.

### Technische schuld

| Schuld | Gevolg |
|---|---|
| **Alleen H1/H2** | H3-subsecties worden niet als grens gebruikt, dus thematisch samenhangende stukken worden willekeurig in tweeën geknipt |
| **Twee overlappende overlap-mechanismen** | Kop-chunking zet een `... `-prefix met de vorige chunk, en de harde splitter doet daarnaast tekengebaseerde overlap. De prefix vervuilt de embedding: de chunk begint dan niet meer met zijn eigen inhoud |
| **Teken-heuristieken** (`4` en `3` tekens per token) | De werkelijke tokengrenzen worden niet gerespecteerd; vandaar 18 chunks boven de limiet die een tweede, agressievere correctie nodig hadden |
| **`code_snippet` heeft geen onderscheidend vermogen** | 89% is `true`, omdat de vlag ook waar is bij één inline backtick. Filteren op die vlag levert bijna niets op, en dat was de oorspronkelijke bedoeling wel |
| Geen chunkkwaliteitsvalidatie | Er is geen controle dat een chunk niet midden in een codeblok eindigt, en geen meting van chunkgrootteverdeling |

### Aanbevolen verbeteringen

1. **Eén recursieve splitter** (H1 → H2 → H3 → alinea → zin) op werkelijke
   tokentellingen, met een doel van 600–900 tokens en 15% overlap.
2. **De `... `-prefix laten vallen.** Overlap hoort aan de randen te zitten, niet als
   voorvoegsel dat de embedding vertekent.
3. **Codeblokken als aparte chunks** met een eigen type, in plaats van een boolean op
   een tekstchunk. Dat maakt `find_examples` en toekomstige codegerichte tools beter en
   maakt de vlag pas echt bruikbaar.
4. Chunkstatistieken in CI rapporteren (grootteverdeling, percentage boven limiet).

---

## 4. Embeddings

**Bestand:** [server/embed_client.py](./server/embed_client.py) · **Status:** ✅ solide

### Wat het doet

`text-embedding-3-small` (1536 dimensies) via de OpenAI-client, in batches van 100, met
`tenacity` exponential backoff. Er is een singleton-client en een kostenraming.

### Volwassenheid: 7/10

Het werkt en is redelijk robuust voor het gelukkige pad.

### Technische schuld

| Schuld | Gevolg |
|---|---|
| **`retry_if_exception_type((Exception,))`** | Een 401 (verkeerde key) of een 400 (ongeldige input) wordt drie keer herhaald met backoff. Dat vertraagt falen en maskeert de echte oorzaak. Alleen `APIConnectionError`, `RateLimitError` en `InternalServerError` zijn herhaalbaar |
| **Geen timeout op de client** | Een hangende upstream houdt een worker vast tot de Gunicorn-timeout. Met vier workers is dat een eenvoudige denial of service |
| **De embeddingversie wordt nergens opgeslagen** | Wisselt iemand van model, dan staan er stil vectoren van twee modellen in dezelfde kolom en degradeert de retrieval zonder enige melding. Dit is de gevaarlijkste stille faalmodus in de pijplijn |
| `EMBEDDING_MODEL`-constante negeert `config.embedding_model` | De configuratie optie bestaat maar heeft geen effect op het daadwerkelijke pad; de config-documentatie is daardoor misleidend |
| `get_embedding_dimension()` en `estimate_embedding_cost()` bestaan, maar het eerste wordt in productie niet gebruikt | Dode code die suggereert dat dimensies dynamisch worden bepaald |

### Aanbevolen verbeteringen

1. **Timeouts** op de OpenAI-client en een retrybeleid dat alleen transiente fouten
   herhaalt.
2. **Embeddingmodel en -dimensie per chunk opslaan** in `documents.metadata`, en bij een
   modelwissel een volledige herindexering forceren. Dit maakt stille degradatie
   detecteerbaar.
3. `EMBEDDING_MODEL` laten respecteren of verwijderen zodat er één bron van waarheid is.

---

## 5. Supabase

**Bestanden:** [migrations/](./migrations),
[server/supabase_client.py](./server/supabase_client.py) · **Status:** ⚠️ deels

### Wat er staat

| Object | Toestand |
|---|---|
| `documents` | `id TEXT PK` (sha256), `content TEXT`, `source_filename`, `chunk_index`, `chunk_count`, `section_heading`, `token_count`, `code_snippet`, `metadata JSONB`, `embedding vector(1536)` |
| `file_manifest` | `filename PK`, `content_hash`, `last_indexed`, `doc_id` FK → `documents.id` ON DELETE CASCADE, `chunk_count` |
| Indexen | HNSW op `embedding` (`m=16, ef_construction=64`), B-tree op `source_filename` en `code_snippet` |
| `match_documents` (v1) | Bestaat, **negeert `match_threshold`**, heeft geen filters, retourneert de embeddingkolom |
| `match_documents_v2` | Nieuw: werkende drempel, `filter_source`, `filter_code_only`, geen embeddingkolom, `REVOKE`/`GRANT` |
| RLS | **Niet aangezet** |

### Volwassenheid: 5/10

Het schema is redelijk gekozen en v2 is de eerste correcte retrievalfunctie. Maar er
staan ontwerpfouten in die zich pas bij groei of bij een securityaudit laten zien.

### Technische schuld

| Schuld | Gevolg |
|---|---|
| **Geen RLS (SEC-01)** | Met de publieke anon-key is de hele index leesbaar én beschrijfbaar, en `match_documents` is publiek aanroepbaar. Dit is een deployment-blocker |
| **FK-ontwerpfout in `file_manifest`** | `doc_id` verwijst naar alleen chunk 0 van een bestand. Verdwijnt die chunk (re-chunking, delete-per-bestand), dan verdwijnt de manifestrij mee via `ON DELETE CASCADE` en raak je de change-tracking kwijt |
| `chunk_count` in de manifest wordt hardgecodeerd op 1 | Het veld liegt, ondanks het commentaar dat het "later bijgewerkt" wordt |
| **Upsert van álle chunks in één request** | 630 rijen × ~6 KB vectoren is ruim 4 MB JSON in één call. Bij groei geeft dat 413's, timeouts en geheugenpieken |
| **`fetch_manifest()` zonder paginering** | PostgREST kapt stil af op zijn rijlimiet; boven ~1000 bestanden breekt de incrementele indexering zonder foutmelding |
| `get_document_stats()` doet `count="exact"` per `/status`-aanroep | Een O(n)-count op een endpoint dat 30×/minuut mag worden aangeroepen |
| Geen `tsvector`/GIN-index | Er is geen performante tekstzoekroute, wat hybrid search blokkeert |
| HNSW-parameters op de standaardwaarden | `m=16, ef_construction=64` is wat pgvector zelf kiest; er is geen tuning en geen `ef_search`-instelling |
| `id TEXT` + `vector(1536)` | Bij 100k chunks is dat ~600 MB tabel plus ~1 GB index. `halfvec` of Matryoshka-truncatie naar 512 dimensies geeft een factor 3 |
| **Migratie 0002 is nergens toegepast** | De nieuwe code werkt niet tegen een database zonder 0002 |

### Aanbevolen verbeteringen

1. **0002 uitvoeren en verifiëren** (P1), daarna RLS aanzetten inclusief `REVOKE` op v1.
2. Upserts batchen (~200 rijen) en de manifest gefaseerd ophalen.
3. `tsvector`-kolom met GIN-index plus `ts_rank_cd` om hybrid search mogelijk te maken.
4. `halfvec` of dimensietruncatie overwegen vóór 100k chunks.
5. `match_documents` (v1) uitfaseren zodra `/chat` op v2 is overgezet; dan is er één
   retrievalfunctie in plaats van twee, waarvan één met een dode parameter.

---

## 6. Retrieval

**Bestand:** [pinescript_mcp/service.py](./pinescript_mcp/service.py) +
`server/supabase_client.search_documents_v2` · **Status:** ✅ correct

### Wat het doet

`_fetch_rows()` embedt de query en roept `match_documents_v2` aan met de drempel, het
sourcefilter en de code-vlag. De database past de predicaten toe vóór `LIMIT`, dus de
client vraagt exact `top_k` rijen en gooit niets weg. Er is geen over-fetch meer.

### Volwassenheid: 7/10

Dit is de stap die in deze ronde het meest is verbeterd. De filtering is nu correct en
de datastroom is 92,9% kleiner.

### Technische schuld

| Schuld | Gevolg |
|---|---|
| **Geen re-ranking** | De ordening is pure vectorafstand. Een cross-encoder of LLM-rerank zou de top-5 aanzienlijk verbeteren, zeker voor code-vragen |
| **Geen diversiteit (MMR)** | De context kan meerdere bijna-identieke chunks bevatten, wat contextbudget verspilt aan redundantie |
| **Hybrid search staat uit** | `bm25_search` leest een `rank`-veld dat de Supabase-client nooit teruggeeft, dus de score is altijd 0 en de gewichten doen niets. Puur vector is nu de eerlijke toestand |
| **HNSW past filters ná de indexwandeling toe** | Een zeer selectief `filter_source` kan minder dan `match_count` rijen geven. `hnsw.iterative_scan` lost dit op (pgvector ≥ 0.8) maar staat bewust niet in de functiebody |
| `strpos()` kan geen B-tree gebruiken | Bij een grote tabel is het sourcefilter een seq-scan-achtige post-filter. Op de huidige omvang onmeetbaar, bij groei relevant |
| Geen caching | Elke aanroep doet een nieuwe embedding-call, ook voor identieke queries |

### Aanbevolen verbeteringen

1. **Hybrid repareren** (`tsvector` + GIN + `ts_rank_cd`) of de BM25-code verwijderen.
   Dode code die een functie suggereert is erger dan geen code.
2. **RRF of genormaliseerde scorefusie** in plaats van de huidige gewogen som.
3. **Re-ranking** van top-50 naar top-8 met een cross-encoder.
4. **MMR** met λ ≈ 0.7 om redundantie in de context te beperken.
5. Embedding-cache op queryhash voor herhaalde vragen.

---

## 7. Context Assembly

**Bestanden:** `server/retriever.assemble_context`, `_trim_content_for_budget` ·
**Status:** ⚠️ bruikbaar

### Wat het doet

Selecteert chunks op aflopende similariteit totdat het tokenbudget bereikt is. Past een
chunk niet meer, dan stopt de selectie — of, als er nog niets geselecteerd is, wordt die
ene chunk afgekapt op een codemuur-bewuste manier.

### Volwassenheid: 6/10

De codemuur-bewuste afkapping is een goede, doordachte toevoeging. De selectielus heeft
een echt probleem.

### Technische schuld

| Schuld | Gevolg |
|---|---|
| **`assemble_context` stopt bij de eerste chunk die niet past** | Is chunk 3 te groot, dan worden chunk 4 tot en met 12 nooit bekeken, ook als ze klein zijn en prima zouden passen. De context wordt dus onnodig kleiner dan het budget toestaat |
| Geen deduplicatie | Overlappende chunks uit dezelfde pagina komen allebei in de context, wat budget verspilt aan herhaling |
| Trimming op tekens, niet op tokens | `_trim_content_for_budget` gebruikt 4 tekens per token; voor code klopt die verhouding niet, dus het budget kan worden overschreden of onderschreden |
| Geen ordeningstrategie | De meest relevante passage staat vooraan, wat bij lange contexten niet optimaal is ("lost in the middle") |

### Aanbevolen verbeteringen

1. **Doorgaan in plaats van stoppen** bij een chunk die niet past: probeer de volgende.
2. Dedupliceren op `source_filename` + overlappende `chunk_index`.
3. Werkelijke tokentelling in plaats van een teken-heuristiek.
4. Overwegen om de best scorende passage als laatste te plaatsen.

---

## 8. MCP Tools

**Bestand:** [pinescript_mcp/](./pinescript_mcp) · **Status:** ✅ werkt

### Wat er staat

| Onderdeel | Toestand |
|---|---|
| `app.py` | `MCPServer` + registratie van drie tools via `add_tool` |
| `tools.py` | Drie synchrone tools met `Annotated[..., Field(...)]`-parameters en `ToolError`-vertaling |
| `service.py` | Retrieval, delegeert filtering naar de database |
| `schemas.py` | `DocHit`, `ExampleHit`, `Passage`, `ContextPack` als outputschema's |
| `__main__.py` | stdio-entrypoint; stdout blijft schoon, logging naar stderr |
| Transport | stdio (bewezen met een subprocess-test); Streamable HTTP gedocumenteerd maar niet uitgerold |
| Tests | 66 tests met `pglast`, 65 + 1 skipped zonder; hermetisch, inclusief een contract-lock op alle schema's |

### Volwassenheid: 7/10

Compleet voor zijn scope: read-only retrieval over de bestaande index, met correcte
schema's, correcte foutafhandeling en een testbare structuur.

### Technische schuld

| Schuld | Gevolg |
|---|---|
| **`search_docs` levert alleen een excerpt** | Er is geen `get_chunk`. Wie de volledige tekst wil, moet `get_context` gebruiken, dat een tokenbudget opvraagt en dus duurder is dan nodig |
| **`EXAMPLE_SCAN_FACTOR = 2` is niet gekalibreerd** | De factor bestaat omdat `code_snippet` ook waar is bij inline code, maar het getal 2 is een aanname. Met echte data is te meten welk deel van de gevlagde chunks daadwerkelijk een fenced block oplevert |
| `language` staat hardgecodeerd op `"pine"` | `extract_code_blocks` legt de taalmuur niet vast, dus een niet-Pine-blok zou verkeerd gelabeld worden. In deze corpus is alles Pine, dus het is nu onschadelijk maar niet correct |
| Geen resources en geen prompts | De SDK ondersteunt `@mcp.resource` en `@mcp.prompt`; die zijn niet gebruikt, terwijl een resource als `pinescript://doc/{slug}` de excerpt-beperking zou opheffen |
| Geen caching | Identieke queries doen identieke embeddings en RPC's |
| **Nooit end-to-end gedraaid** | Alle tests stubs de retrieval. Er is geen bewijs dat de keten met een echte index werkt |
| Geen authenticatie of rate limiting (SEC-21) | Over stdio onschadelijk; over HTTP een open leesroute |

### Aanbevolen verbeteringen

1. **`get_chunk` toevoegen** plus resources, zodat excerpt en volledige tekst gescheiden
   blijven zonder `get_context` te misbruiken.
2. **De scanfactor kalibreren** met echte data.
3. De taalmuur wél vastleggen als de corpus ooit niet-Pine code gaat bevatten.
4. Een querycache toevoegen.
5. End-to-end verifiëren tegen een echte index — dit is de belangrijkste openstaande
   onbekende in het hele systeem.

---

## Samenvattende beoordeling

| Dimensie | Score | Toelichting |
|---|---|---|
| Scraper | 6/10 | Betrouwbaar voor de User Manual, blind voor de Reference |
| Processing | 4/10 | Destructief: bron-URL's weg, code gedupliceerd, testloos |
| Chunking | 4/10 | Alleen H1/H2, heuristische grenzen, nutteloze code-vlag |
| Embeddings | 7/10 | Solide, maar zonder modelversie per chunk en zonder timeout |
| Supabase | 5/10 | Correct schema voor v2, maar geen RLS en schaalproblemen in de client |
| Retrieval | 7/10 | Nu correct gefilterd; mist re-ranking, RRF en diversiteit |
| Context Assembly | 6/10 | Goede muren-afkapping, maar stopt te vroeg bij een grote chunk |
| MCP Tools | 7/10 | Compleet voor read-only; mist `get_chunk` en end-to-end bewijs |
| **Totaal** | **5,8/10** | Een werkende keten met een correcte retrievalstap, waarvan de ETL en de database-laag de kwaliteit nog beperken |

De zwakste schakels liggen **vóór** de retrieval: processing en chunking bepalen welke
tekst er te vinden is, en die stappen gooien informatie weg (bron-URL's) en versnipperen
slecht (H1/H2-only, dubbele codeblokken). Betere retrieval kan dat niet compenseren. De
sterkste schakels zijn de MCP-laag en de embeddingclient, die beide doen wat ze moeten
doen zonder verrassingen.

De eerstvolgende winst zit daarom niet in meer retrievaltrucs, maar in het **bewijzen van
de huidige keten** (migratie uitvoeren, end-to-end draaien) en daarna in het
**terugwinnen van provenance** in de processing-stap.
