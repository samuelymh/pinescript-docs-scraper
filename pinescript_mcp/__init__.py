"""PineScript MCP server (read-only MVP).

Exposes the existing Pine Script v6 documentation index over the Model Context
Protocol. The retrieval logic lives in `pinescript_mcp.service`, which reuses
the modules under `server/` unchanged.

Entry points:
    python -m pinescript_mcp          # stdio (default, for editor clients)
    mcp dev pinescript_mcp/app.py     # MCP Inspector

Importing this package does not read configuration or open any connection;
`server.config.get_config()` is only called when a tool actually runs.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
