"""MCP server connection settings.

Controls how the LangGraph application connects to the MCP tool server.

Env vars (override in .env):
    MCP_TRANSPORT   — "stdio" (default, local dev) | "sse" (separate process / Docker)
    MCP_SERVER_HOST — hostname/IP of the SSE server (default: localhost)
    MCP_SERVER_PORT — port of the SSE server (default: 8000)
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class MCPToolsSettings(BaseSettings):
    # depending on how we run the mcp_server (separate folder), in Docker or in the separate process - we need to choose the separate configuration
    # stdio - process
    # sse - docker
    mcp_transport: Literal["stdio", "sse"] = "stdio"
    mcp_server_host: str = "localhost"
    mcp_server_port: int = 8000

    @property
    def mcp_server_url(self) -> str:
        """SSE endpoint URL (only used when mcp_transport='sse')."""
        return f"http://{self.mcp_server_host}:{self.mcp_server_port}/sse"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
