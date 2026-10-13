"""stdio entry point.

    python -m pinescript_mcp

stdio is the default transport: the client launches this process as a
subprocess and speaks MCP over stdin/stdout. Nothing may be written to stdout
while serving, since stdout *is* the wire -- logging goes to stderr.

For a network transport, run `mcp run pinescript_mcp/app.py --transport
streamable-http` instead, or mount the ASGI app as described in docs/MCP.md.
"""

from __future__ import annotations

import logging
import sys

from pinescript_mcp.app import mcp


def main() -> None:
    # Diagnostics must go to stderr; stdout is the protocol channel.
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    mcp.run()


if __name__ == "__main__":
    main()
