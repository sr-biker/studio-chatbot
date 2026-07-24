"""LangChain model/store construction.

Chat model is OpenAI direct for both profiles (local = gpt-4o-mini, prod = whatever
settings.chat_model_name is set to via values-prod.yaml). Embedding model is OpenAI
text-embedding-3-small (1536-dim) for both profiles too -- see settings.vector_dimension."""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres import PGVector

from app.config import settings


@lru_cache
def chat_model():
    """Builds (and memoizes) the primary chat model used by the named agents.

    Returns:
        A ChatOpenAI instance configured from settings.chat_model_name /
        settings.openai_api_key, shared by all Assistant instances in main.py's lifespan.
    """
    return ChatOpenAI(model=settings.chat_model_name, api_key=settings.openai_api_key)


@lru_cache
def router_chat_model():
    """Builds (and memoizes) the model used by Router's LLM classifier.

    A separate, deterministic (temperature 0) model instance from chat_model() so the
    router's route selection doesn't inherit chat_model's generation temperature.

    Returns:
        A ChatOpenAI instance with temperature=0.0.
    """
    return ChatOpenAI(model=settings.chat_model_name, api_key=settings.openai_api_key, temperature=0.0)


@lru_cache
def summarize_model():
    """Builds (and memoizes) the model used by the SUMMARIZE agent.

    Cheap/fast tier, distinct from chat_model() -- see settings.summarize_model_name.

    Returns:
        A ChatOpenAI instance configured from settings.summarize_model_name.
    """
    return ChatOpenAI(model=settings.summarize_model_name, api_key=settings.openai_api_key)


@lru_cache
def embedding_model():
    """Builds (and memoizes) the embedding model, shared by both profiles.

    Returns:
        An OpenAIEmbeddings instance configured from settings.openai_embedding_model_name.
    """
    return OpenAIEmbeddings(model=settings.openai_embedding_model_name, api_key=settings.openai_api_key)


@lru_cache
def vector_store() -> PGVector:
    """Builds (and memoizes) the PGVector store backing the FAQ collection.

    Returns:
        A PGVector instance bound to the "faq_store" collection, using embedding_model()
        and a connection string built from settings/resolve_db_credentials().
    """
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
