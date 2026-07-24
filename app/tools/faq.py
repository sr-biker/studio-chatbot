"""LangChain tool that grounds studio questions in the FAQ pgvector store via RAG."""

from functools import lru_cache

from langchain_core.tools import tool

from app.ai_config import vector_store

DEFAULT_TOP_K = 4
MAX_TOP_K = 10
RETRIEVAL_CACHE_SIZE = 256


def search_faq_raw(query: str, top_k: int | None = None) -> list[dict]:
    """Runs a similarity search against the FAQ vector store.

    Shared implementation used by both the LangChain tool (search_studio_faq) and
    GET /internal/faq/search, so both paths return the same hit shape.

    Args:
        query: The search text.
        top_k: Requested number of hits; clamped to MAX_TOP_K, defaults to
            DEFAULT_TOP_K if None, zero, or negative.

    Returns:
        A list of dicts, each with "section", "source", and "text" keys, one per
        matched chunk, ordered by similarity.
    """
    k = min(top_k, MAX_TOP_K) if (top_k and top_k > 0) else DEFAULT_TOP_K
    return list(_cached_search(query, k))


# Cached separately from search_faq_raw() so the cache key is the resolved k, not the raw
# (possibly None) top_k -- same query at different effective k must be distinct entries.
# Safe for the process lifetime: FAQ content only changes via a fresh ingest + redeploy (see
# app.faq_loader), never mid-process, so there's no in-process path that changes what a
# given (query, k) should return. Returns a
# tuple, not a list, so the result itself stays hashable-safe to cache (search_faq_raw wraps
# it back into a list per call so callers can't mutate the cached entry).
@lru_cache(maxsize=RETRIEVAL_CACHE_SIZE)
def _cached_search(query: str, k: int) -> tuple[dict, ...]:
    hits = vector_store().similarity_search(query, k=k)
    return tuple(
        {"section": doc.metadata.get("section"), "source": doc.metadata.get("source"), "text": doc.page_content}
        for doc in hits
    )


@tool
def search_studio_faq(query: str, top_k: int | None = None) -> list[dict]:
    """Retrieve relevant excerpts from the studio's FAQ knowledge base — membership,
    class/event registration, cancellations, and class-specific policies (yoga, pilates,
    strength training, birthday parties, happy hour, etc). Pass the user's question as
    the query. Base your answer on what this returns; if it doesn't contain the answer,
    say so rather than guessing."""
    return search_faq_raw(query, top_k)


TOOLS = [search_studio_faq]
