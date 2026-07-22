import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

PROFILE = os.environ.get("APP_PROFILE", "local")  # "local" or "prod"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
# OpenAI direct for chat/agents in both profiles -- local defaults to gpt-4o-mini, prod
# overrides CHAT_MODEL_NAME via values-prod.yaml. OpenAI direct also used for prod embeddings.
CHAT_MODEL_NAME = os.environ.get("CHAT_MODEL_NAME", "gpt-4o-mini")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OPENAI_EMBEDDING_MODEL_NAME = "text-embedding-3-small"

DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = os.environ.get("DB_PORT", "5432")
DB_NAME = os.environ.get("DB_NAME", "studio")
DB_USER = os.environ.get("DB_USER", "studio")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "studio")

# prod pulls DB credentials from AWS Secrets Manager via DB_SECRET_NAME instead.
DB_SECRET_NAME = os.environ.get("DB_SECRET_NAME")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# Each env re-embeds its own store with its own model — a PGVector column is fixed-dimension,
# so local (MiniLM, 384) and prod (OpenAI text-embedding-3-small, 1536) can never share one
# physical store. "Similar embedding models" means same family/quality tier, not identical
# vectors or a shared table.
VECTOR_DIMENSION = 384 if PROFILE == "local" else 1536

# FAQ source: local profile reads a checked-in copy; prod pulls the live object from S3.
FAQ_S3_BUCKET = os.environ.get("FAQ_S3_BUCKET", "senthil-studio-faq")
FAQ_S3_KEY = os.environ.get("FAQ_S3_KEY", "faq.md")
FAQ_LOCAL_PATH = BASE_DIR / "data" / "faq.md"

DATA_DIR = BASE_DIR / "data"
MIGRATIONS_DIR = BASE_DIR / "db_migrations"


def resolve_db_credentials() -> tuple[str, str]:
    """Returns (username, password). In prod, pulled from AWS Secrets Manager."""
    if PROFILE == "prod" and DB_SECRET_NAME:
        import json

        import boto3

        client = boto3.client("secretsmanager", region_name=AWS_REGION)
        secret = json.loads(client.get_secret_value(SecretId=DB_SECRET_NAME)["SecretString"])
        return secret["username"], secret["password"]
    return DB_USER, DB_PASSWORD
