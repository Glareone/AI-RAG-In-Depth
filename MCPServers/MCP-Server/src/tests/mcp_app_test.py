from fastmcp import Client

from mcp_server.mcp_app import mcp


async def test_mcp_ping() -> None:
    async with Client(mcp) as client:
        result = await client.ping()
    assert result is not None


async def test_mcp_no_tools() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert tools == []
