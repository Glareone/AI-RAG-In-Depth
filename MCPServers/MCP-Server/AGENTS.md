# MCP Server — Agent Surface

Current MCP surface: **no tools registered** (foundation only).

## Endpoints
- `GET /health` → `{"status": "ok"}`
- `/mcp` — FastMCP streamable HTTP / SSE transport (MCP protocol)

## Status
Foundation verified: boots locally and in Docker, quality gate (ruff/mypy/pytest) clean.

## Up next
- **OpenTelemetry + Arize Phoenix** — activate full tracing when `OTEL_TELEMETRY_ENABLED=true`
- **First `@mcp.tool()`** — structured pydantic output, thin tool wrapper, OTel span per tool

## Conventions & footguns
See `.claude/rules/python-mcp.md` for the full list. Summary:
- `_meta` passthrough — #1 footgun. Must propagate on every tool when tools land.
- `tool.outcome` banned — use `Status(StatusCode.OK/ERROR)`.
- Tools thin — logic in `services/`, not `tools/`.
- No module-level globals — `app_context.py` lifespan + `Depends`.
- OTel span per tool: `mcp.tool.<name>`, `get_span_kind_attributes("tool")`, `get_tool_attributes`, `get_input_attributes`, `get_output_attributes`.
