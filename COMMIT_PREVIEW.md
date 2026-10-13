# Commit Preview — MCP MVP + migratie 0002

**Dit document is een voorstel. Er is niets gecommit.** De repository staat op
`49352fe` en de werkboom bevat de hieronder beschreven wijzigingen.

## Blokkade: de merge is nog open

```
MERGE_HEAD aanwezig: ja   (0aca412 "Automated: regenerate processed docs")
conflictpaden:       70   (69 × DU, 7 × AD, 1 × UU processed_all_docs.md)
```

`git commit` weigert met **`fatal: cannot do a partial commit during a merge`**. Elke
gewone commit is geblokkeerd tot de merge is afgerond. Zie
[Opties](#opties) hieronder voor twee werkwijzen.

## Gewijzigde bestanden

### Van deze wijziging

| Bestand | Wijziging | Toelichting |
|---|---|---|
| [server/supabase_client.py](./server/supabase_client.py) | **+123 / −0** | `git diff --numstat` bevestigt **nul verwijderingen**. Toegevoegd: `MigrationRequiredError`, `_is_missing_function_error()`, `search_documents_v2()`. `search_similar_documents` en daarmee `/chat` zijn niet aangeraakt |
| [docs/DEPLOYMENT.md](./docs/DEPLOYMENT.md) | **+223 / −2** | nieuwe sectie **MCP server deployment** (208 regels): verplichte uitrolvolgorde (migratie 0002 vóór de code), wat er faalt zonder, vier verificatiequeries met benoemde argumenten (directe call, source-filter, code-only-filter, threshold) en een rollbackprocedure. Daarnaast de migratietabel in *Running migrations and indexing* bijgewerkt |
| [SECURITY-REVIEW.md](./SECURITY-REVIEW.md) | **+102 / −0** | statusupdate: Opgelost / Openstaand / Uitgesteld, SEC-01 expliciet nog open, nieuwe risico's (SEC-21, uitrolkoppeling) en verdwenen risico's |

### Al aanwezig vóór deze wijziging — **niet meenemen zonder beoordeling**

| Bestand | Status |
|---|---|
| `.github/workflows/cleanup-processed.yml` | gewijzigd (unstaged) |
| `.github/workflows/generate-processed.yml` | gewijzigd (unstaged) |
| `README.md` | gewijzigd (unstaged) |

Deze drie horen bij eerder werk en hebben niets met de MCP-MVP te maken. Ze zijn
bewust buiten de voorgestelde commit gelaten.

## Nieuwe bestanden

### Van deze wijziging — code, tests en configuratie

| Bestand | Regels |
|---|---|
| `pinescript_mcp/__init__.py` | 17 |
| `pinescript_mcp/__main__.py` | 32 |
| `pinescript_mcp/app.py` | 63 |
| `pinescript_mcp/schemas.py` | 85 |
| `pinescript_mcp/service.py` | 329 |
| `pinescript_mcp/tools.py` | 183 |
| `tests/mcp/conftest.py` | 115 |
| `tests/mcp/test_mvp_tools.py` | 468 |
| `tests/mcp/test_migration_0002.py` | 210 |
| `tests/mcp/test_search_documents_v2.py` | 162 |
| `migrations/0002_match_documents_v2.sql` | 151 |
| `scripts/benchmark_filtering.py` | 388 |
| `requirements-mcp.txt` | 24 |
| `requirements-dev.txt` | 6 |
| `docs/MCP.md` | 330 |
| **Subtotaal** | **2.563** |

### Documentatie in deze ronde

| Bestand | Regels | Opmerking |
|---|---|---|
| `RELEASE_NOTES_MCP_MVP.md` | 238 | nieuw |
| `ARCHITECTURE_STATUS.md` | 374 | nieuw |
| `implementatieplan_mcp.md` | ±666 | inhoudelijk bijgewerkt |
| `verificatie_mcp.md` | ±541 | inhoudelijk bijgewerkt |
| `COMMIT_PREVIEW.md` | ±188 | dit bestand |
| **Subtotaal** | **±2.007** | |

**Totaal nieuwe bestanden: ±4.570 regels.**
Plus 448 toegevoegde regels in bestaande tracked bestanden (+123 supabase_client,
+223 docs/DEPLOYMENT, +102 SECURITY-REVIEW) =
**±5.018 toegevoegde regels in deze wijziging.**

> Twee voorbehouden bij deze aantallen.
> 1. Regelaantallen zijn **fysieke regels, inclusief lege regels**
>    (`@(Get-Content <bestand>).Count`). `Measure-Object -Line` telt lege regels niet
>    mee en geeft een 10 tot 40 procent lager getal; eerdere versies van dit document
>    gebruikten die methode en noemden 3.682 regels. Dat was te laag.
> 2. De aantallen van `implementatieplan_mcp.md`, `verificatie_mcp.md` en
>    `COMMIT_PREVIEW.md` zijn indicatief (±), omdat het bewerken van die bestanden hun
>    eigen regelaantal verandert. De code- en testaantallen zijn exact.

## Voorgestelde commit message

```
feat(mcp): add read-only MCP server with database-side filtering

Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>
```

Alternatief als de migratie als aparte commit wordt uitgebracht (aanbevolen, zie
[Risicoanalyse](#risicoanalyse)):

```
feat(mcp): add read-only MCP server (search_docs, get_context, find_examples)
feat(db): add match_documents_v2 with working threshold and filters
docs(mcp): add release notes, architecture status and updated reports
```

## Voorgestelde commit description

```
Adds a read-only MCP server over the existing Pine Script v6 documentation
index, plus the database function that makes its filtering correct.

MCP server (new package pinescript_mcp/, 709 lines):
- search_docs: ranked chunks with provenance (source, section, score)
- get_context: full passages within a token budget
- find_examples: extracted Pine Script code samples
- stdio entrypoint (python -m pinescript_mcp); HTTP mounting documented
- Tool contracts locked by tests; output schemas derive from return types

Database (migrations/0002_match_documents_v2.sql):
- match_documents_v2 applies match_threshold as a SIMILARITY floor
  (v1 accepted the argument and ignored it)
- adds filter_source (literal, case-insensitive substring) and
  filter_code_only
- stops returning the embedding column, 71.5% of a row's bytes
- REVOKE EXECUTE FROM PUBLIC + GRANT TO service_role, so the new function
  is not publicly callable the way PostgreSQL defaults to

Retrieval:
- filtering now happens in the database, before LIMIT
- removes the 4x/2x over-fetch that existed only to compensate for
  client-side filtering
- measured: 32 -> 8 rows per query at top_k=8, 92.9% fewer bytes

Client:
- server/supabase_client.py gains search_documents_v2 (additive, +123/-0)
- search_similar_documents is untouched so POST /chat is unaffected
- missing migration is reported as MigrationRequiredError after one
  attempt instead of three, with an actionable message

Tests (66 passed, 0 failed, hermetic; 65 passed + 1 skipped without pglast):
- 37 tool tests: registration, contract lock, behaviour, error handling,
  stdio subprocess
- 16 migration contract tests: SQL parameter names must match the client
  payload, the threshold must be used in the body, no wildcard construction
  in the source filter, v1 must stay untouched
- 13 client tests: parameter semantics, retry policy, migration detection
- an autouse guard fails any test that forgets to stub an outbound call
- the SQL parser validation needs the optional pglast dependency from
  requirements-dev.txt; without it that one test skips and the other 65 pass
- an autouse guard fails any test that forgets to stub an outbound call

Docs: docs/MCP.md (install, migrations, four client configs) and
docs/DEPLOYMENT.md (mandatory migration order, four verification
queries, rollback), plus RELEASE_NOTES_MCP_MVP.md, ARCHITECTURE_STATUS.md
and updated implementatieplan_mcp.md, verificatie_mcp.md and
SECURITY-REVIEW.md.

Dependencies: requirements-mcp.txt declares the MCP runtime dependencies
and raises the pydantic floor to >=2.12 as the SDK requires;
requirements-dev.txt declares the optional pglast for the SQL parser test.

Not included on purpose:
- migration 0002 has never been executed against a real database; validate
  on a test project before deploying (see RELEASE_NOTES_MCP_MVP.md)
- RLS remains off (SEC-01); 0002 only avoids repeating that mistake for the
  new function
- review_pinescript and explain_function are postponed

Existing tests: server/tests is unchanged at 8 failed / 68 passed /
2 skipped, identical to the baseline before this change.
```

## Risicoanalyse

| # | Risico | Ernst | Toelichting en mitigatie |
|---|---|---|---|
| 1 | **De merge is open; committen is geblokkeerd** | 🔴 Hoog | 70 conflictpaden plus `UU processed_all_docs.md`. Zolang dit open staat is al het werk untracked en dus niet herstelbaar via git. **Mitigatie:** eerst de merge afronden, of een gerichte commit maken via een tijdelijk index (`read-tree` → `write-tree` → `commit-tree` → `update-ref`), wat de merge-state niet aanraakt |
| 2 | **De code vereist migratie 0002 die nergens is toegepast** | 🔴 Hoog | Wordt de code uitgerold vóór de migratie, dan mislukken **alle drie** de MCP-tools met `MigrationRequiredError`. Dit is de eerste wijziging waarbij code en schema samen moeten uitrollen. **Mitigatie:** 0002 eerst toepassen en verifiëren, uitrolvolgorde vastleggen in `docs/DEPLOYMENT.md`, en overwegen de migratie als aparte commit uit te brengen zodat de volgorde in de history zichtbaar is |
| 3 | **Delen van bestaande bestanden zonder beoordeling** | 🟠 Middel | `.github/workflows/*.yml` en `README.md` zijn al gewijzigd. Een `git add -A` zou die ongevraagd meesturen. **Mitigatie:** expliciet per pad stagen, nooit `-A`. De voorgestelde commit bevat alleen de MCP- en migratiepaden |
| 4 | **Merge-conflicten in `pinescript_docs/`** | 🟠 Middel | De 69 `DU`-paden en de `UU` in `processed_all_docs.md` horen bij de docs-generatie en niet bij deze wijziging. Ze meenemen in een commit over de MCP-MVP maakt de history onleesbaar en kan de corpus in een inconsistente staat vastleggen. **Mitigatie:** buiten de commit laten |
| 5 | **Testverwachtingen wijzigen mee** | 🟡 Laag | `tests/mcp/` is nieuw, dus er is geen bestaande test aangepast. De server-suite is aantoonbaar ongewijzigd (identieke 8 failures). **Mitigatie:** geen; het is een controlepunt, geen probleem |
| 6 | ~~`pglast` is geïnstalleerd maar niet gedeclareerd~~ **opgelost** | — | Vastgelegd in `requirements-dev.txt` als optionele ontwikkeldependency. Zonder het pakket skipt de parsertest (`importorskip`) en is het resultaat 65 passed, 1 skipped, 0 failed; dat is geverifieerd door het pakket te simuleren als afwezig. Wie de repo kloont en alleen `requirements-mcp.txt` installeert, ziet dus één skip — het bedoelde gedrag, nu gedocumenteerd |
| 7 | **`server.zip` en `structure.txt` zijn verdwenen** | 🟡 Laag | Beide stonden eerder als untracked in de werkboom en zijn nu **niet meer aanwezig op schijf**. Ze waren niet tracked, dus ze zijn **niet via git te herstellen**. Niet door mij verwijderd. **Mitigatie:** als dit onbedoeld was, zijn ze alleen terug te halen uit een bestandssysteem-backup; als het bewust was, is er niets aan de hand — het waren beide artefacten die als opruimkandidaat stonden |
| 8 | **De migratie is niet uitgevoerd** | 🟠 Middel | De SQL is alleen statisch gevalideerd (echte PostgreSQL-parser, 4 statements plus de functiebody) en via contracttests. Semantiek, planner-gedrag en latency zijn onbewezen. **Mitigatie:** de commit-description zegt dit expliciet; de release notes zetten het als P1 |

## Opties om te committen

| Optie | Werkwijze | Voor- en nadelen |
|---|---|---|
| **A. Merge eerst afronden** | De 70 conflictpaden oplossen, dan normaal committen | Schoonste history. Kost tijd en vraagt beslissingen over de docs-generatie (welke generatie blijft?) |
| **B. Gerichte commit via tijdelijk index** | `GIT_INDEX_FILE=<tmp>` → `git read-tree HEAD` → `git add <paden>` → `git write-tree` → `git commit-tree` → `git update-ref refs/heads/main <commit> HEAD`, daarna het tijdelijke index verwijderen | Alleen de bedoelde bestanden komen erin, de merge blijft intact. Nadeel: de commit staat op een branch die midden in een merge zit, dus zodra de merge wordt afgerond of afgebroken moet er opnieuw worden gekeken. **`git merge --abort` zou de commit van de branch-tip halen** (herstelbaar via `git reflog`) |
| **C. Niet committen** | Laten staan tot de merge is afgerond | Geen risico op een verkeerde commit, maar al het werk blijft untracked en dus onbeschermd |

Aanbevolen: **A**, en **B** alleen als de merge niet binnen afzienbare tijd afgerond
kan worden.
