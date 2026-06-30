# MCP Server

FastAPI application exposing MCP tools via FastMCP over HTTP/SSE.

## Service root
`src/` — all commands run from here.

## Run
```bash
cd src
uv sync
uv run python -m mcp_server.main
# → http://localhost:8000
# Health: /health
# MCP:    /mcp (streamable HTTP / SSE)
```

## Quality gate (run before any handback)
```bash
cd src
uv run ruff format .
uv run ruff check .
uv run mypy src/ tests/
uv run pytest
```

## Layout
See `.claude/rules/python-mcp.md` — authoritative structure, naming, telemetry, and testing conventions.

## Docker
```bash
cd src
docker build -t mcp-server .
docker run -p 8000:8000 mcp-server
```
