"""LangChain model/store construction.

Chat model is OpenAI direct for both profiles (local = gpt-4o-mini, prod = whatever
settings.chat_model_name is set to via values-prod.yaml). Embedding model is
profile-gated: local = in-process sentence-transformers all-MiniLM-L6-v2 (384-dim), prod =
OpenAI text-embedding-3-small (1536-dim). The two profiles never share a physical vector store
— see settings.vector_dimension."""

from functools import lru_cache

from langchain_openai import ChatOpenAI
from langchain_postgres import PGVector

from app.config import settings


@lru_cache
def chat_model():
    return ChatOpenAI(model=settings.chat_model_name, api_key=settings.openai_api_key)


@lru_cache
def router_chat_model():
    """A separate, deterministic (temperature 0) model instance for Router's classifier."""
    return ChatOpenAI(model=settings.chat_model_name, api_key=settings.openai_api_key, temperature=0.0)


@lru_cache
def embedding_model():
    if settings.app_profile == "prod":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model_name, api_key=settings.openai_api_key)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


@lru_cache
def vector_store() -> PGVector:
    _username, _password = settings.resolve_db_credentials()
    connection = (
        f"postgresql+psycopg://{_username}:{_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}"
    )
    return PGVector(
        embeddings=embedding_model(),
        collection_name="faq_store",
        connection=connection,
        use_jsonb=True,
        create_extension=False,
    )
