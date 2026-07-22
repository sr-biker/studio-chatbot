import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MIGRATIONS_DIR = BASE_DIR / "db_migrations"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_profile: Literal["local", "prod"] = "local"

    openai_api_key: str = ""
    # OpenAI direct for chat/agents in both profiles -- local defaults to gpt-4o-mini, prod
    # overrides CHAT_MODEL_NAME via values-prod.yaml. OpenAI direct also used for prod embeddings.
    chat_model_name: str = "gpt-4o-mini"
    # Cheap/fast tier for the summarize agent -- no tool-calling loop, short outputs, doesn't
    # need chat_model_name's quality; gpt-5-nano is OpenAI's Haiku-equivalent tier.
    summarize_model_name: str = "gpt-5-nano"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    openai_embedding_model_name: str = "text-embedding-3-small"

    db_host: str = "localhost"
    db_port: str = "5432"
    db_name: str = "studio"
    db_user: str = "studio"
    db_password: str = "studio"
    # prod pulls DB credentials from AWS Secrets Manager via db_secret_name instead.
    db_secret_name: str | None = None
    aws_region: str = "us-east-1"

    # FAQ source: local profile reads a checked-in copy; prod pulls the live object from S3.
    faq_s3_bucket: str = "senthil-studio-faq"
    faq_s3_key: str = "faq.md"

    @property
    def vector_dimension(self) -> int:
        # Each env re-embeds its own store with its own model — a PGVector column is
        # fixed-dimension, so local (MiniLM, 384) and prod (OpenAI text-embedding-3-small,
        # 1536) can never share one physical store. "Similar embedding models" means same
        # family/quality tier, not identical vectors or a shared table.
        return 384 if self.app_profile == "local" else 1536

    @property
    def faq_local_path(self) -> Path:
        return DATA_DIR / "faq.md"

    def resolve_db_credentials(self) -> tuple[str, str]:
        """Returns (username, password). In prod, pulled from AWS Secrets Manager."""
        if self.app_profile == "prod" and self.db_secret_name:
            import boto3

            client = boto3.client("secretsmanager", region_name=self.aws_region)
            secret = json.loads(client.get_secret_value(SecretId=self.db_secret_name)["SecretString"])
            return secret["username"], secret["password"]
        return self.db_user, self.db_password


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
