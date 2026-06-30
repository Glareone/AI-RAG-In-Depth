---
name: explore-python
description: Research Python/FastAPI MCP server project code - tools, structure, patterns, FastAPI organization, asyncio, _meta, human-in-the-loop, OTel, Auth
context: fork
agent: Explore
allowed-tools: Read Grep Glob Bash
---

# Explore MCP Server (Python)

Explore this project's Python/FastAPI MCP server code.

## Orientation — read these first

Before exploring code, check project state:
- `CLAUDE.md` — conventions, architecture decisions, setup notes
- `CHANGELOG.md` — what's been implemented, recent changes, what's in progress

Both may be empty early in the project. Note that and proceed to code.

## Service Context

- Python >= 3.13, FastAPI >= 0.115
- MCP protocol via **FastMCP** (`fastmcp`); tool registration with `@mcp.tool()` async defs
- `_meta` field used for human-in-the-loop elicitation / approval flows
- Package manager **uv** (Astral); build backend **hatchling**, lockfile: `uv.lock`
- Linter **ruff** (target `py313`); type-check **mypy** >= 1.14
- External MCP servers in use: context7, GitHub MCP, Serena MCP, rtl (rust token killer)

## What to look for

- Tool definitions (`@mcp.tool()`) — inputs, outputs, `_meta` handling
- Human-in-the-loop patterns — elicitation requests, approval flows, `_meta` passthrough
- FastAPI route organization — async handlers, dependency injection, middleware
- Auth patterns — API keys, OAuth, token validation
- Observability — OTel spans, logging, structured output
- Error handling — MCP error codes, HTTP status mapping
- Project structure — how modules are split, where config lives

## Task

$ARGUMENTS
