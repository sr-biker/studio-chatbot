"""LangChain tool that grounds studio questions in the FAQ pgvector store via RAG."""

from typing import Optional

from langchain_core.tools import tool

from app.ai_config import vector_store

DEFAULT_TOP_K = 4
MAX_TOP_K = 10


def search_faq_raw(query: str, top_k: Optional[int] = None) -> list[dict]:
    """Shared implementation used by both the LangChain tool and GET /faq/search."""
    k = min(top_k, MAX_TOP_K) if (top_k and top_k > 0) else DEFAULT_TOP_K
    hits = vector_store().similarity_search(query, k=k)
    return [
        {"section": doc.metadata.get("section"), "source": doc.metadata.get("source"), "text": doc.page_content}
        for doc in hits
    ]


@tool
def search_studio_faq(query: str, top_k: Optional[int] = None) -> list[dict]:
    """Retrieve relevant excerpts from the studio's FAQ knowledge base — membership,
    class/event registration, cancellations, and class-specific policies (yoga, pilates,
    strength training, birthday parties, happy hour, etc). Pass the user's question as
    the query. Base your answer on what this returns; if it doesn't contain the answer,
    say so rather than guessing."""
    return search_faq_raw(query, top_k)


TOOLS = [search_studio_faq]
