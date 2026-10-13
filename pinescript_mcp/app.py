"""MCPServer instance and tool registration.

Deliberately does not call `run()`: `mcp dev`, `mcp run`, the tests and
`python -m pinescript_mcp` all import this module, and an import must never
start a server.

Serves the read-only MVP:

    search_docs    ranked documentation chunks
    get_context    full passages, token-budgeted
    find_examples  extracted Pine Script code samples

Postponed: `review_pinescript` (no rule engine exists) and `explain_function`
(the Language Reference has never been crawled -- see verificatie_mcp.md).
"""

from __future__ import annotations

from mcp.server import MCPServer

from pinescript_mcp import tools

INSTRUCTIONS = """\
Pine Script v6 documentation assistant.

This server searches the index of the Pine Script v6 user manual (73 pages) that
the pinescript-docs-scraper project maintains. It is read-only: it quotes and
retrieves documentation, it does not execute or validate code.

Tools:
- search_docs(query): start here for documentation questions. Returns ranked
  chunks with the source page, section heading and similarity score.
- get_context(query, token_budget): full passages when the search excerpts are
  not enough. Costs more context; use it deliberately.
- find_examples(topic): copy-ready code samples from the documentation.

Guidance:
- Always cite the `source` (and `section` when present) of anything you quote.
- A `score` near 1.0 means the chunk matches the query closely; a low score
  below about 0.2 is usually noise. Use `min_score` to filter rather than
  discarding hits by hand.
- The index covers the user manual, not the Language Reference: it explains how
  components work, but it does not contain per-function signatures. Do not
  present a retrieved example as an authoritative function signature.
- Code from the documentation is illustrative. Adapt inputs, risk management and
  version annotations before suggesting it for live use.
"""

mcp = MCPServer(
    name="pinescript-docs",
    title="Pine Script v6 documentation",
    description="Read-only retrieval over the Pine Script v6 user manual.",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)

# Registered by reference rather than by decorator so that the tool definitions
# stay importable on their own (tests import pinescript_mcp.tools directly).
mcp.add_tool(tools.search_docs)
mcp.add_tool(tools.get_context)
mcp.add_tool(tools.find_examples)

__all__ = ["mcp", "INSTRUCTIONS"]
