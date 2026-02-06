"""
Configuration module for the LangGraph document analysis application.
Contains environment setup, LLM initialization, and application constants.
"""

import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

# Load environment variables
load_dotenv()

# Application Constants
MAX_ITERATIONS = 3
CONFIDENCE_THRESHOLD = 0.7
VALIDATION_THRESHOLD = 0.5
EXPECTED_RULES_PER_DOCUMENT = 3

# Azure OpenAI Model Options
# - gpt-5: Larger model, slower, more expensive, best for complex tasks
# - gpt-5-mini: Smaller, faster, less expensive
# - gpt-5-nano: Smallest, fastest, least expensive
# - gpt-5.1-chat: Latest reasoning model

# LLM Configuration
def get_llm(model: str = None):
    """
    Initialize and return the Azure OpenAI LLM instance.

    Args:
        model: Model deployment name (if None, reads from AZURE_OPENAI_MODEL env var)

    Returns:
        AzureChatOpenAI instance configured for reasoning models
    """
    # Use model from environment variable if not specified
    if model is None:
        model = os.getenv("AZURE_OPENAI_MODEL", "gpt-5-mini")

    return AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
        azure_deployment=model,
        temperature=0,
    )

# Initialize default LLM instance
llm = get_llm()