from fastmcp import FastMCP

mcp = FastMCP("MCP-Server")


def build_mcp_app() -> object:
    """Return FastMCP ASGI app mounted at root. Mount on FastAPI at /mcp."""
    return mcp.http_app(path="/")
