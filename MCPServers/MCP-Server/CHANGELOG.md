# Changelog

## [Unreleased]

### Added
- Minimal FastAPI + FastMCP foundation: `create_app()`, `/health`, MCP mount at `/mcp`
- `app_context.py`: lifespan-scoped `AppContext` (settings, tracer), no module globals
- `config/settings.py`: pydantic-settings with `OTEL_TELEMETRY_ENABLED`, `MCP_SERVER_PHOENIX_PROJECT_NAME`, `MCP_SERVER_LOG_LEVEL`
- `telemetry/setup.py`: gated tracer stub (no-op when disabled, ready for Phoenix when enabled)
- `routers/health.py`, `routers/mcp.py`: FastAPI routers
- Placeholder packages: `schemas/`, `services/`, `tools/`
- `pyproject.toml`: uv + hatchling, ruff (py313), mypy strict, pytest asyncio_mode=auto
- `Dockerfile`: python:3.13-slim + uv, production build
- Bootstrap `CLAUDE.md`, `AGENTS.md`
- ADR `adr/260630-mcp-foundation.md`: records FastMCP-on-FastAPI transport/layout choice
