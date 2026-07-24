"""Mocked unit test for app.tools.faq's retrieval cache -- mocks vector_store directly (no
live DB, no OpenAI embedding call)."""

from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from app.tools.faq import _cached_search, search_faq_raw


def _mock_vector_store(docs):
    store = MagicMock()
    store.return_value.similarity_search.return_value = docs
    return store


# The whole point of the cache: an identical (query, k) must not re-hit the vector store a
# second time -- this is what makes it worth adding at all (saves an embedding + similarity
# search round trip on repeat FAQ questions).
def test_repeated_query_hits_vector_store_only_once():
    _cached_search.cache_clear()
    docs = [Document(page_content="Arrive early.", metadata={"section": "Classes", "source": "faq.md"})]
    store = _mock_vector_store(docs)

    with patch("app.tools.faq.vector_store", store):
        search_faq_raw("How early should I arrive?")
        search_faq_raw("How early should I arrive?")

    store.return_value.similarity_search.assert_called_once()


# Same question text but a different effective k must be treated as a distinct cache entry --
# otherwise a caller asking for top_k=10 could silently get back a stale top_k=4 result (or
# vice versa) from an earlier call.
def test_same_query_different_top_k_are_cached_separately():
    _cached_search.cache_clear()
    docs = [Document(page_content="Arrive early.", metadata={"section": "Classes", "source": "faq.md"})]
    store = _mock_vector_store(docs)

    with patch("app.tools.faq.vector_store", store):
        search_faq_raw("How early should I arrive?", top_k=2)
        search_faq_raw("How early should I arrive?", top_k=8)

    assert store.return_value.similarity_search.call_count == 2


# Callers must get back a fresh list each time, not a shared reference into the cached tuple --
# otherwise one caller mutating its result list would corrupt what every other caller sees.
def test_result_is_a_fresh_list_not_a_shared_cached_reference():
    _cached_search.cache_clear()
    docs = [Document(page_content="Arrive early.", metadata={"section": "Classes", "source": "faq.md"})]
    store = _mock_vector_store(docs)

    with patch("app.tools.faq.vector_store", store):
        first = search_faq_raw("How early should I arrive?")
        first.append({"section": "injected", "source": "bad", "text": "bad"})
        second = search_faq_raw("How early should I arrive?")

    assert len(second) == 1
