"""LangChain model/store construction.

Chat model is profile-gated: local = OpenAI (gpt-4o-mini, no AWS creds needed for a dev loop),
prod = AWS Bedrock (Claude 3.5 Sonnet). Embedding model is also profile-gated: local = in-process
sentence-transformers all-MiniLM-L6-v2 (384-dim), prod = OpenAI text-embedding-3-small (1536-dim).
The two profiles never share a physical vector store — see config.VECTOR_DIMENSION."""

from functools import lru_cache

from langchain_postgres import PGVector

from app import config


@lru_cache
def chat_model():
    if config.PROFILE == "prod":
        from langchain_aws import ChatBedrock

        return ChatBedrock(model_id=config.BEDROCK_CHAT_MODEL_ID, region_name=config.AWS_REGION)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.CHAT_MODEL_NAME, api_key=config.OPENAI_API_KEY)


@lru_cache
def router_chat_model():
    """A separate, deterministic (temperature 0) model instance for Router's classifier."""
    if config.PROFILE == "prod":
        from langchain_aws import ChatBedrock

        return ChatBedrock(
            model_id=config.BEDROCK_CHAT_MODEL_ID,
            region_name=config.AWS_REGION,
            model_kwargs={"temperature": 0.0},
        )
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=config.CHAT_MODEL_NAME, api_key=config.OPENAI_API_KEY, temperature=0.0)


@lru_cache
def embedding_model():
    if config.PROFILE == "prod":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=config.OPENAI_EMBEDDING_MODEL_NAME, api_key=config.OPENAI_API_KEY)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)


@lru_cache
def vector_store() -> PGVector:
    _username, _password = config.resolve_db_credentials()
    connection = (
        f"postgresql+psycopg://{_username}:{_password}@{config.DB_HOST}:{config.DB_PORT}/{config.DB_NAME}"
    )
    return PGVector(
        embeddings=embedding_model(),
        collection_name="faq_store",
        connection=connection,
        use_jsonb=True,
        create_extension=False,
    )
