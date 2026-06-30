---
status: accepted
date: 2026-06-30
author: Aleksei Kolesnikov
---

# Use FastMCP mounted on FastAPI via http_app + combine_lifespans

## Context
Need to expose MCP tools over HTTP/SSE inside a FastAPI application that also serves REST
endpoints (health, future admin routes). Two options: run FastMCP standalone, or mount it
onto FastAPI. Choice affects lifespan management, test ergonomics, and future HITL routing.

## Decision
Mount FastMCP onto FastAPI:
- `mcp_asgi_app = mcp.http_app(path="/")` (FastMCP 2.x API)
- `FastAPI(lifespan=combine_lifespans(app_lifespan, mcp_asgi_app.lifespan))`
- Mount at `/mcp` via `app.mount("/mcp", mcp_asgi_app)`

MCP endpoint: `http://host:8000/mcp` (streamable HTTP / SSE transport).

Package layout: `src/src/mcp_server/` (double-nested src; deploy root is `src/`, import root
is `src/src/`).

## Consequences
- REST and MCP share one process and port — simpler Docker/deployment.
- `combine_lifespans` required — omitting it leaves MCP session manager uninitialized.
- Standalone `FastMCP` stdio or SSE-only transport not available without refactor.
- Test ergonomics: in-memory `Client(mcp)` works without a running server for unit tests;
  integration tests use httpx ASGI transport against `create_app()`.
