# Changelog

## [Unreleased]

### Up next
- OpenTelemetry → Arize Phoenix: full tracer provider + OTLP exporter in `telemetry/setup.py`
- First `@mcp.tool()` with structured pydantic output, OTel span convention

### Added
- Minimal FastAPI + FastMCP foundation: `create_app()`, `/health`, MCP mount at `/mcp`
- `app_context.py`: lifespan-scoped `AppContext` (settings, tracer), no module globals
- `config/settings.py`: pydantic-settings for `OTEL_TELEMETRY_ENABLED`, `MCP_SERVER_PHOENIX_PROJECT_NAME`, `MCP_SERVER_LOG_LEVEL`
- `telemetry/setup.py`: gated tracer stub — no-op when disabled, TracerProvider skeleton when enabled
- `routers/health.py`, `routers/mcp.py`: FastAPI routers
- Placeholder packages: `schemas/`, `services/`, `tools/`
- `pyproject.toml`: uv + hatchling, ruff (py313), mypy strict, pytest asyncio_mode=auto
- `Dockerfile`: python:3.13-slim-bookworm, linux/arm64, non-root user app (uid/gid 999), split dep/source layers, venv Python at runtime
- `docker-compose.yml`: env loaded from `src/.env` via `env_file`
- `README.md`: dev commands, Docker/Compose workflow, env var reference
- `CLAUDE.md`, `AGENTS.md`: bootstrap docs
- ADR `adr/260630-mcp-foundation.md`: FastMCP-on-FastAPI via `http_app` + `combine_lifespans`

### Verified
- Docker build and run clean on `linux/arm64` (Apple Silicon)
- `curl localhost:8000/health` → `{"status":"ok"}`
- `uv run pytest` → 3/3 passed (health + MCP handshake + empty tool list)
- `ruff format/check` + `mypy --strict` → no issues
