FROM --platform=linux/arm64 python:3.13-slim-bookworm

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Non-root runtime user
RUN groupadd --gid 999 app \
    && useradd --uid 999 --gid 999 --no-create-home --shell /sbin/nologin app

WORKDIR /app

# --- Dependency layer (cache bust only when pyproject.toml or uv.lock changes) ---
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# --- Project source ---
COPY src/ ./src/
RUN uv sync --frozen --no-dev \
    && chown -R app:app /app

ENV PYTHONPATH=/app/src
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

USER app

EXPOSE 8000

CMD ["/app/.venv/bin/python", "-m", "mcp_server.main"]
