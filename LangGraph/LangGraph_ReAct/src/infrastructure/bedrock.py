import boto3
from botocore.client import BaseClient

from src.config import Settings


def make_bedrock_client(settings: Settings) -> BaseClient:
    session = boto3.Session(
        profile_name=settings.aws_profile or None,
        region_name=settings.aws_region,
    )
    return session.client("bedrock-runtime")
