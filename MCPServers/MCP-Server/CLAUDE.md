# MCP Server

FastAPI application exposing MCP tools via FastMCP over HTTP/SSE.

## Service root
`src/` — all commands run from here.

## Run locally
```bash
cd src
uv sync --group dev
source .venv/bin/activate
python -m mcp_server.main
# → http://localhost:8000
# Health: /health
# MCP:    /mcp  (streamable HTTP / SSE)
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

## Docker / Compose
```bash
cd src
cp .env.example .env          # fill in values
docker compose up --build -d
curl localhost:8000/health
docker compose down
```

Image: `python:3.13-slim-bookworm`, platform `linux/arm64`, non-root user `app` (uid/gid 999).
Runtime: venv Python called directly (`/app/.venv/bin/python`) — no uv cache needed at runtime.

## What's done
- FastAPI + FastMCP foundation booting and verified in Docker
- `/health` endpoint, MCP streamable HTTP/SSE at `/mcp`, no tools yet
- Gated telemetry stub (`OTEL_TELEMETRY_ENABLED=false` → no-op tracer)
- Full package tree: `config/`, `routers/`, `schemas/`, `services/`, `telemetry/`, `tools/`
- ruff (py313) + mypy (strict) + pytest (asyncio_mode=auto) passing clean

## Up next
1. **OpenTelemetry → Arize Phoenix** — wire full tracer provider + OTLP exporter in `telemetry/setup.py` when `OTEL_TELEMETRY_ENABLED=true`; add Phoenix env vars to `.env.example`
2. **First `@mcp.tool()`** — structured output via pydantic schemas in `schemas/`; thin tool in `tools/`, logic in `services/`; OTel span per tool following the span convention in the rules file
