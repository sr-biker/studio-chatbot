import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


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
    openai_embedding_model_name: str = "text-embedding-3-small"

    # Refs (commit hash, or "latest") pinning which LangSmith Hub prompt commit each named
    # agent runs -- see app/prompts.py. Default "latest" is right for local dev (immediately
    # see Hub edits); prod overrides these via values-prod.yaml to a specific commit hash
    # that scripts/promote_prompt.py has verified passes the eval suite, so a CSR's in-
    # progress Hub edit never reaches prod traffic on its own.
    support_prompt_ref: str = "latest"
    general_prompt_ref: str = "latest"
    membership_prompt_ref: str = "latest"

    db_host: str = "localhost"
    db_port: str = "5432"
    db_name: str = "studio"
    db_user: str = "studio"
    db_password: str = "studio"
    # prod pulls DB credentials from AWS Secrets Manager via db_secret_name instead.
    db_secret_name: str | None = None
    aws_region: str = "us-east-1"

    # FAQ source: local profile reads a checked-in copy; prod exports the live Google Doc
    # Auth is via GOOGLE_APPLICATION_CREDENTIALS (a service-account key file mounted from
    # a k8s Secret), picked up automatically by google-auth's application-default flow.
    faq_google_doc_id: str = ""

    # membership is a sibling Spring Boot service (~/projects/membership) -- class
    # type/schedule, linked to a contact (~/projects/contacts-micro-service, PII) only by
    # email. The chatbot only ever calls membership's GET /api/memberships/lookup
    # (email-or-phone, sandboxed -- see app/tools/studio_api.py); membership itself resolves
    # a phone to an email internally via its own ContactsClient, so this app never talks to
    # contacts-micro-service directly. Local default assumes membership's docker-compose is
    # run with its default port remapped off the shared 8080 -- see that repo's README.
    membership_api_base_url: str = "http://localhost:8082"

    @property
    def vector_dimension(self) -> int:
        """Embedding vector width for the active profile's PGVector column.

        Both profiles use OpenAI text-embedding-3-small now, so this is the same in local
        and prod -- kept as a property (rather than inlined at call sites) since the two
        profiles still never share a physical store (separate DBs).

        Returns:
            1536.
        """
        return 1536

    @property
    def faq_local_path(self) -> Path:
        """Filesystem path to the local-profile FAQ markdown copy (data/faq.md).

        Returns:
            Absolute Path to data/faq.md under the project root.
        """
        return DATA_DIR / "faq.md"

    def resolve_db_credentials(self) -> tuple[str, str]:
        """Resolves DB credentials for the active profile.

        In prod with db_secret_name set, credentials are pulled live from AWS Secrets
        Manager (JSON secret with "username"/"password" keys); otherwise falls back to
        the static db_user/db_password settings fields (used for local dev).

        Returns:
            A (username, password) tuple.
        """
        if self.app_profile == "prod" and self.db_secret_name:
            import boto3

            client = boto3.client("secretsmanager", region_name=self.aws_region)
            secret = json.loads(client.get_secret_value(SecretId=self.db_secret_name)["SecretString"])
            return secret["username"], secret["password"]
        return self.db_user, self.db_password


@lru_cache
def get_settings() -> Settings:
    """Builds (and memoizes) the process-wide Settings instance.

    Returns:
        A cached Settings, loaded from environment variables and .env on first call.
    """
    return Settings()


settings = get_settings()
