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
    """Builds (and memoizes) the embedding model for the active profile.

    Profile-gated: local uses an in-process HuggingFace sentence-transformers model (no
    API key needed); prod uses OpenAI embeddings. See settings.vector_dimension -- the two
    profiles' embeddings are never dimension-compatible, so this must stay profile-gated.

    Returns:
        An OpenAIEmbeddings instance in prod, a HuggingFaceEmbeddings instance in local.
    """
    if settings.app_profile == "prod":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model=settings.openai_embedding_model_name, api_key=settings.openai_api_key)
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)


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
