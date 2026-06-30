import uvicorn
from fastapi import FastAPI
from fastmcp.utilities.lifespan import combine_lifespans

from mcp_server.app_context import app_lifespan
from mcp_server.mcp_app import build_mcp_app
from mcp_server.routers.health import router as health_router
from mcp_server.routers.mcp import mount_mcp


def create_app() -> FastAPI:
    mcp_asgi_app = build_mcp_app()
    app = FastAPI(
        title="MCP Server",
        lifespan=combine_lifespans(app_lifespan, mcp_asgi_app.lifespan),  # type: ignore[attr-defined]
    )
    app.include_router(health_router)
    mount_mcp(app, mcp_asgi_app)
    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run("mcp_server.main:app", host="0.0.0.0", port=8000, reload=False)
