from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # App
    log_level: str = "INFO"
    data_dir: str = "data/"
    # AWS Bedrock
    aws_profile: str = ""
    aws_region: str = "eu-central-1"
    bedrock_model_id: str = "arn:aws:bedrock:eu-central-1:119796001828:inference-profile/eu.anthropic.claude-sonnet-4-6"
    # Arize Phoenix
    phoenix_endpoint: str = "http://localhost:6006/"
    phoenix_project_name: str = "aml-react-agent"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
