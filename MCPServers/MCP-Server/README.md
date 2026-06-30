# MCP Server

FastAPI application exposing MCP tools via FastMCP over HTTP/SSE.

- **Health:** `GET /health`
- **MCP endpoint:** `/mcp` (streamable HTTP / SSE transport)

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) — `brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Docker + Docker Compose (for containerized run)

## Setup

```bash
cd src
cp .env.example .env      # fill in any values you need
uv sync --group dev       # installs app deps + dev tools into src/.venv
source .venv/bin/activate
```

## Development commands

All commands run from `src/`.

### Format

```bash
uv run ruff format .
```

### Lint

```bash
uv run ruff check .
```

### Type check

```bash
uv run mypy src/ tests/
```

### Tests

```bash
uv run pytest
```

### Full quality gate (run before committing)

```bash
uv run ruff format . && uv run ruff check . && uv run mypy src/ tests/ && uv run pytest
```

## Run locally

```bash
cd src
uv run python -m mcp_server.main
# → http://localhost:8000
```

## Docker

Build and run the container with env vars loaded from `src/.env`:

```bash
cd src
cp .env.example .env      # if not done already

# Build and start
docker compose up --build

# Background
docker compose up --build -d

# Stop
docker compose down
```

Env vars are sourced from `src/.env` via the `env_file` directive in `docker-compose.yml`.
Add or change values in `.env` and restart — no image rebuild needed for env-only changes.

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OTEL_TELEMETRY_ENABLED` | `false` | Enable Phoenix/OTel tracing |
| `MCP_SERVER_PHOENIX_PROJECT_NAME` | `mcp-server` | Phoenix project name for traces |
| `MCP_SERVER_LOG_LEVEL` | `INFO` | Logging level (`DEBUG`/`INFO`/`WARNING`/`ERROR`) |

See `src/.env.example` for the full template.
