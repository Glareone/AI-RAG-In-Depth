"""MCP client — connects the LangGraph application to the MCP tool server.

Transport is controlled by MCPToolsSettings (src/config/mcp_tools_settings.py):

  stdio (default)
    The client spawns mcp_server/server.py as a subprocess.
    No separate server process needed — good for local development.

  sse
    The client connects to a running HTTP/SSE server.
    Use this when the server runs as a separate process or in Docker.
    Set MCP_SERVER_HOST and MCP_SERVER_PORT (or MCP_TRANSPORT=sse in .env).

Usage:

    async with get_mcp_tools() as tools:
        llm_with_tools = llm.bind_tools(tools)
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from src.config.mcp_tools_settings import MCPToolsSettings


def _build_server_config(settings: MCPToolsSettings) -> dict:
    """Return MultiServerMCPClient config for the configured transport."""
    if settings.mcp_transport == "sse":
        return {
            "financial-documents": {
                "url": settings.mcp_server_url,
                "transport": "sse",
            }
        }

    # stdio — spawn the server as a subprocess
    return {
        "financial-documents": {
            "command": sys.executable,
            "args": ["-m", "mcp_server.server"],
            "transport": "stdio",
        }
    }


@asynccontextmanager
async def get_mcp_tools(
    settings: MCPToolsSettings | None = None,
) -> AsyncGenerator[list[BaseTool], None]:
    """Async context manager that yields LangChain tools from the MCP server.

    Args:
        settings: MCPToolsSettings instance. Loads from env/.env if not provided.

    Yields:
        list[BaseTool] ready for llm.bind_tools(tools).

    Example::

        async with get_mcp_tools() as tools:
            llm_with_tools = llm.bind_tools(tools)
    """
    if settings is None:
        settings = MCPToolsSettings()

    config = _build_server_config(settings)
    async with MultiServerMCPClient(config) as client:
        yield client.get_tools()
