import os

import pytest

from tests.helpers import client  # noqa: F401


@pytest.fixture(autouse=True, scope="session")
def _env_defaults() -> None:
    """Disable telemetry and set required env vars for all tests."""
    os.environ.setdefault("OTEL_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("MCP_SERVER_PHOENIX_PROJECT_NAME", "mcp-server-test")
    os.environ.setdefault("MCP_SERVER_LOG_LEVEL", "WARNING")
