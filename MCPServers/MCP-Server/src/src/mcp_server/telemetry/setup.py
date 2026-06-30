import logging

from opentelemetry import trace
from opentelemetry.trace import Tracer

from mcp_server.config.settings import Settings

logger = logging.getLogger(__name__)


def setup_telemetry(settings: Settings) -> Tracer:
    """Configure OTel tracer. No-op when OTEL_TELEMETRY_ENABLED is false."""
    if not settings.otel_telemetry_enabled:
        logger.info("Telemetry disabled (OTEL_TELEMETRY_ENABLED=false) — using no-op tracer")
        return trace.get_tracer(__name__)

    # Lazy import: telemetry extras must be installed
    try:
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError as exc:
        raise RuntimeError(
            "Install [telemetry] extra to enable Phoenix tracing: uv sync --extra telemetry"
        ) from exc

    resource = Resource.create(
        {
            SERVICE_NAME: "mcp-server",
            "openinference.project.name": settings.mcp_server_phoenix_project_name,
        }
    )
    provider = TracerProvider(resource=resource)
    trace.set_tracer_provider(provider)
    logger.info(
        "Telemetry enabled — Phoenix project: %s",
        settings.mcp_server_phoenix_project_name,
    )
    return trace.get_tracer(__name__)
