FROM python:3.12-slim

WORKDIR /app

# Install uv
RUN pip install --no-cache-dir uv

# Copy workspace manifests first (cached dependency layer)
COPY pyproject.toml uv.lock* ./
COPY mcp_server/pyproject.toml mcp_server/pyproject.toml

# Install only the mcp_server package and its dependencies
RUN uv sync --package financial-mcp-server --no-dev

# Copy only what the server needs from the project
COPY src/__init__.py          src/__init__.py
COPY src/tools/__init__.py    src/tools/__init__.py
COPY src/tools/data_loader.py    src/tools/data_loader.py
COPY src/tools/document_parser.py src/tools/document_parser.py
COPY src/tools/calculator.py     src/tools/calculator.py
COPY mcp_server/__init__.py   mcp_server/__init__.py
COPY mcp_server/server.py     mcp_server/server.py

# Document data — mount as a volume in production to keep the container stateless
# docker run -v $(pwd)/data:/app/data ...
COPY data/ data/

ENV MCP_TRANSPORT=sse
ENV MCP_SERVER_PORT=8000
ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000

CMD ["python", "-m", "mcp_server.server"]