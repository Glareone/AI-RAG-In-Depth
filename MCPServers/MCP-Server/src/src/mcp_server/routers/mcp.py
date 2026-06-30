from fastapi import FastAPI

MCP_MOUNT_PATH = "/mcp"


def mount_mcp(app: FastAPI, mcp_asgi_app: object) -> None:
    """Mount FastMCP ASGI app at MCP_MOUNT_PATH."""
    app.mount(MCP_MOUNT_PATH, mcp_asgi_app)  # type: ignore[arg-type]
