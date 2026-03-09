# MCP Tools — Architecture & Usage

## Two-Layer Design

```
src/tools/               ← Plain Python implementations (shared by app and MCP server)
    data_loader.py
    document_parser.py
    calculator.py

mcp_server/              ← Standalone MCP server (separate process / Docker)
    server.py                wraps src/tools/* as @mcp.tool() endpoints
    Dockerfile
    requirements.txt

src/mcp_tools/           ← Application-side only
    client.py                connects to mcp_server, returns list[BaseTool]
    mcp_tools_settings.py    transport + host + port config
```

**`src/tools/`** — ordinary Python functions. No protocol, no network. Unit-testable in isolation.

**`mcp_server/server.py`** — a separate service that wraps those functions behind the
[Model Context Protocol](https://modelcontextprotocol.io/). It never runs inside the
application process.

**`src/mcp_tools/client.py`** — the application's only connection point to the MCP server.
It translates MCP tool schemas into `list[BaseTool]` for `llm.bind_tools()`.

---

## Transport Modes

Configured via `src/config/mcp_tools_settings.py` (env vars or `.env`):

| Env var | Default | Description |
|---|---|---|
| `MCP_TRANSPORT` | `stdio` | `stdio` = subprocess, `sse` = HTTP |
| `MCP_SERVER_HOST` | `localhost` | SSE mode only |
| `MCP_SERVER_PORT` | `8000` | SSE mode only |

### stdio — local development

The client spawns `mcp_server/server.py` as a subprocess automatically.
No separate server startup needed.

```
Application process
    └── spawns → mcp_server/server.py (subprocess, stdin/stdout)
```

```ini
# .env (or omit — stdio is the default)
MCP_TRANSPORT=stdio
```

### sse — separate process or Docker

The server runs independently and exposes an HTTP/SSE endpoint.
The client connects by URL.

```
mcp_server container  :8000/sse
         ↑
Application process  ──── HTTP/SSE ────
```

```ini
# .env
MCP_TRANSPORT=sse
MCP_SERVER_HOST=localhost   # or Docker service name, e.g. "mcp-server"
MCP_SERVER_PORT=8000
```

---

## Running the Server

### Locally (stdio — auto-started by client, no manual step needed)

```bash
# Only needed for manual testing / MCP Inspector
python -m mcp_server.server
```

### Locally (SSE mode)

```bash
MCP_TRANSPORT=sse python -m mcp_server.server
```

### Docker

```bash
# Build from project root
docker build -f mcp_server/Dockerfile -t financial-mcp-server .

# Run (mount data/ as volume to keep container stateless)
docker run -p 8000:8000 -v $(pwd)/data:/app/data financial-mcp-server
```

Then set in `.env`:
```ini
MCP_TRANSPORT=sse
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=8000
```

### Inspect tools (MCP Inspector)

```bash
npx @modelcontextprotocol/inspector python -m mcp_server.server
```

---

## Using the Client in LangGraph

```python
from src.mcp_tools.client import get_mcp_tools

async with get_mcp_tools() as tools:
    # tools is list[BaseTool] — standard LangChain interface
    llm_with_tools = llm.bind_tools(tools)
```

Override settings programmatically:

```python
from src.config.mcp_tools_settings import MCPToolsSettings
from src.mcp_tools.client import get_mcp_tools

settings = MCPToolsSettings(mcp_transport="sse", mcp_server_host="mcp-server", mcp_server_port=8000)
async with get_mcp_tools(settings=settings) as tools:
    ...
```

---

## Available Tools (as seen by the LLM)

| Tool | Description |
|---|---|
| `list_available_documents` | List documents, optional ticker filter |
| `get_document` | Full text by `doc_id` |
| `get_document_info` | Metadata by `doc_id` |
| `find_document_by_ticker_and_type` | Look up `doc_id` by ticker + doc type |
| `preview_document` | First N chars of a document |
| `calc_liquidity_ratios` | Current ratio, quick ratio, cash ratio |
| `calc_profitability_ratios` | Gross/operating/net margin, ROA, ROE |
| `calc_leverage_ratios` | D/E, debt-to-assets, net-debt/EBITDA, coverage |
