from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # env var: OTEL_TELEMETRY_ENABLED
    otel_telemetry_enabled: bool = False

    # env var: MCP_SERVER_PHOENIX_PROJECT_NAME
    mcp_server_phoenix_project_name: str = "mcp-server"

    # env var: MCP_SERVER_LOG_LEVEL
    mcp_server_log_level: str = "INFO"
