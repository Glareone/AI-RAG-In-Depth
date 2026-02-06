import os
from dotenv import load_dotenv
from dataclasses import dataclass

# Load environment variables from .env file
load_dotenv()

@dataclass
class AzureOpenAIConfig:
    """Configuration for Azure OpenAI"""
    api_key: str
    api_version: str
    deployment: str
    azure_endpoint: str
    model: str

def get_azure_openai_configurations() -> AzureOpenAIConfig:
    """Get Azure OpenAI configuration from environment variables"""
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    model = os.getenv("AZURE_OPENAI_MODEL")

    if not api_key or not api_version or not deployment or not endpoint or not model:
        raise ValueError("one of environment variables is not found in .env file")

    return AzureOpenAIConfig(
        api_key=api_key,
        api_version=api_version,
        deployment=deployment,
        azure_endpoint=endpoint,
        model=model
    )


def get_phoenix_endpoint() -> str:
    arize_phoenix_endpoint = os.getenv("ARIZE_PHOENIX_ENDPOINT")
    if not arize_phoenix_endpoint:
        raise ValueError("ARIZE_PHOENIX_ENDPOINT environment variable is not set")
    return arize_phoenix_endpoint