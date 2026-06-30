# MCP Server — Agent Surface

Current MCP surface: **no tools registered** (foundation only).

## Endpoints
- `GET /health` → `{"status": "ok"}`
- `/mcp` — FastMCP streamable HTTP / SSE transport (MCP protocol)

## Conventions & footguns
See `.claude/rules/python-mcp.md` for the full list. Summary:
- `_meta` passthrough — #1 footgun. Must propagate on every tool (not needed yet — no tools).
- `tool.outcome` banned — use `Status(StatusCode.OK/ERROR)`.
- Tools thin — logic in `services/`, not `tools/`.
- No module-level globals — `app_context.py` lifespan + `Depends`.
