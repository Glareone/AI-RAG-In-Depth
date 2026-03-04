from src.config import Settings


def init_telemetry(settings: Settings) -> None:
    """Initialize Arize Phoenix OTel tracing and LangChain auto-instrumentation.

    Args:
        settings: Application settings containing Phoenix endpoint and project name.
    """
    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor

        tracer_provider = register(
            project_name=settings.phoenix_project_name,
            endpoint=settings.phoenix_endpoint + "v1/traces",
        )

        LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
        print(
            f"[telemetry] Phoenix tracing enabled → {settings.phoenix_endpoint} "
            f"(project: {settings.phoenix_project_name})"
        )
    except Exception as exc:
        print(f"[telemetry] Warning: could not init Phoenix OTel tracing: {exc}")
        print("[telemetry] Continuing without tracing.")
