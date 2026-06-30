import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from opentelemetry.trace import Tracer

from mcp_server.config.settings import Settings
from mcp_server.telemetry.setup import setup_telemetry


@dataclass
class AppContext:
    settings: Settings
    tracer: Tracer


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = Settings()
    logging.basicConfig(level=settings.mcp_server_log_level.upper())
    tracer = setup_telemetry(settings)
    app.state.context = AppContext(settings=settings, tracer=tracer)
    yield
    # teardown here when needed


def get_app_context(request: Request) -> AppContext:
    """FastAPI dependency — injects AppContext from lifespan state."""
    return request.app.state.context  # type: ignore[no-any-return]
