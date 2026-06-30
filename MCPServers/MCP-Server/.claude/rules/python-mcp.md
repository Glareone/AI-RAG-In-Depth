---
paths:
    - `src/src/mcp_server`
---

# Agent MCP Server - Python and MCP conventions

Applied to `src/src/mcp_server` only. For LangGraph, Langchain, and other aspects, see `python-langchain.md`.

# Purpose

FastAPI application exposing MCP tools via **FastMCP** over HTTP/SSE. Supports human-in-the-loop flows using MCP `_meta` (elicitation and approval). Instrumented with OpenTelemetry + OpenInference semantic conventions; traces exported to Arize Phoenix. This rule governs Python and MCP conventions for this package only.

# Toolchain

- uv + hatchling, Python 3.13+, ruff, mypy (strict), pytest
- Before hand-back: `uv run ruff format .`, `uv run ruff check .`, `uv run mypy src/ tests/`, `uv run pytest` clean; update `CHANGELOG.md` `[Unreleased]` and `AGENTS.md` if behavior or surface changed.

## FastAPI + MCP SDK

- **FastMCP instance** built and configured in `mcp_app.py`; mounted onto the FastAPI app (not run standalone).
- **Tools** defined in `tools/` with `@mcp.tool()` async defs. Tools are thin: validate input, delegate to `services/`, return typed output. No business logic in tool definitions.
- **`_meta` passthrough** — always propagate the incoming request's `_meta` field to the response. Dropping it silently breaks client-side human-in-the-loop (elicitation/approval payloads never reach the caller). Verify `_meta` round-trip in any new tool.
- **Shared singletons** (config, tracer, HTTP clients) held in `app_context.py` via FastAPI lifespan. No module-level globals — all state acquired at startup and injected via FastAPI `Depends`.
- **I/O validated** through pydantic models in `schemas/`. Tool params and return types reference these models.
- **Config** via pydantic-settings in `config/`, fully env-driven. See SECRETS section for required vars.
- **Routers** in `routers/` handle HTTP routes (health, MCP mount, any non-tool HTTP endpoints). MCP transport is mounted once in `routers/mcp.py`.

## Tool span convention

- Every `@mcp.tool()` opens a span via `tracer.start_as_current_span("mcp.tool.<name>", ...)` and uses **OpenInference** semantic conventions (dependency).
- `get_span_kind_attributes("tool")` — emits `openinference.span.kind = "TOOL"` so Arize Phoenix classifies the row correctly (without it Phoenix shows `kind: unknown`).
- `get_tool_attributes(name=..., description=..., parameters={...})` — populates the Tool details panel.
- `get_input_attributes({...})` — `input.value` JSON-serialized.
- On success: `span.set_attributes(get_output_attributes(<text>))` then `span.set_status(Status(StatusCode.OK))`.
- Do **not** set the legacy `tool.outcome` attribute — `Status` is the canonical OTel signal and `tool.outcome` was leaking literal `"ok"` into Phoenix's output column.
- Truncate `output.value` for unbounded responses (e.g. `$metadata` summaries) — keeps Phoenix span exports small.
- Tool errors return a JSON error object.

## SECRETS in env

- **`OTEL_TELEMETRY_ENABLED`** — enables Phoenix telemetry config.
- **`MCP_SERVER_PHOENIX_PROJECT_NAME`** — Phoenix project name, default `mcp-server`; emitted as the `openinference.project.name` resource attribute so cross-service traces (LangGraph + MCP-server) group under one Phoenix project.
- **`MCP_SERVER_LOG_LEVEL`** — logging level (default `INFO`).

## Testing

- `tests/conftest.py` MUST set all necessary secrets, e.g. `OTEL_TELEMETRY_ENABLED=false`. Add others as new env vars are introduced.
- Test files use `*_test.py` format, **not** `*.test.py`; `asyncio_mode = "auto"`; monkeypatch: `pytest.MonkeyPatch`; shared fixtures in `tests/helpers.py`.

## Project Structure

```text
src/                         # deploy root — Docker build context
  .env                       # local secrets (gitignored)
  .env.example               # template listing all required env vars
  Dockerfile                 # container build for the MCP server
  src/                       # Python source root (import root on PYTHONPATH)
    mcp_server/              # importable package (note underscore, not hyphen)
      main.py                # entrypoint — assembles FastAPI app, runs uvicorn
      mcp_app.py             # FastMCP instance; registers tools; mounted onto FastAPI
      app_context.py         # lifespan-scoped shared state: config, tracer, HTTP clients
      config/                # pydantic-settings: env loading, typed settings dataclasses
      routers/               # FastAPI routers (health check, MCP transport mount, etc.)
      schemas/               # pydantic models for tool I/O, request/response shapes
      services/              # business logic and external integrations (tools delegate here)
      telemetry/             # OTel setup, OpenInference helpers, Phoenix exporter config
      tools/                 # MCP tool definitions (@mcp.tool()); thin wrappers over services
  tests/                     # pytest suite (sits outside the package, beside src/)
    conftest.py              # session-scoped fixtures; sets env vars (OTEL off, secrets)
    helpers.py               # shared fixtures and test utilities
    *_test.py                # individual test modules (asyncio_mode=auto)
```
