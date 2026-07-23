"""Mocked integration test for session-based follow-ups over the real /chat endpoint --
verifies main.py's Route.SUMMARIZE branch hands the SUMMARIZE agent the session's stored
transcript (app.session_store), not just the literal trigger message, and that the
session_id is threaded correctly across turns. Mocks the chat models directly (no API
key, no live DB read on the hot path) so it runs in the default, mocked pytest suite
alongside test_router.py / test_moderation.py."""

from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


class _FakeChatModel:
    """A LangChain-model-shaped stub: replies are computed from the message list it's
    actually invoked with, so a test can assert on what context flowed into a turn."""

    def __init__(self, reply_fn):
        self._reply_fn = reply_fn

    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        return AIMessage(content=self._reply_fn(messages))


def _registration_reply(messages):
    return "You're booked for the Friday yoga class -- see you there!"


def _summarize_reply(messages):
    transcript = messages[-1].content
    return f"Summary: {transcript}"


def test_summarize_followup_sees_prior_turn_not_just_trigger_word():
    """"resolved" is a SUMMARIZE-route trigger word (app.router._SUMMARIZE_KW) with no
    content of its own -- the reply must reflect the earlier turn's content, proving
    main.py passed sessions.transcript(session_id) rather than the literal word."""
    with (
        patch("app.main.chat_model", return_value=_FakeChatModel(_registration_reply)),
        patch("app.main.summarize_model", return_value=_FakeChatModel(_summarize_reply)),
        patch("app.main.router_chat_model", return_value=MagicMock()),
        patch("app.main.is_flagged", return_value=False),
    ):
        from app.main import app

        with TestClient(app) as client:
            first = client.post("/chat", json={"message": "I'd like to register for the Friday yoga class"})
            assert first.status_code == 200
            assert first.json()["agent"] == "membership_registration"
            session_id = first.json()["session_id"]

            second = client.post("/chat", json={"message": "resolved", "session_id": session_id})

    assert second.status_code == 200
    body = second.json()
    assert body["agent"] == "summarize"
    assert body["session_id"] == session_id
    assert "yoga" in body["reply"].lower()
    assert "resolved" not in body["reply"].lower()


def test_second_turn_without_session_id_starts_a_new_session():
    """Omitting session_id (as a fresh client would on its first message) must not
    accidentally continue someone else's history -- each call gets its own session_id
    unless one is explicitly passed back."""
    with (
        patch("app.main.chat_model", return_value=_FakeChatModel(_registration_reply)),
        patch("app.main.summarize_model", return_value=MagicMock()),
        patch("app.main.router_chat_model", return_value=MagicMock()),
        patch("app.main.is_flagged", return_value=False),
    ):
        from app.main import app

        with TestClient(app) as client:
            first = client.post("/chat", json={"message": "I'd like to register for the Friday yoga class"})
            second = client.post("/chat", json={"message": "I'd like to register for the Saturday pilates class"})

    assert first.json()["session_id"] != second.json()["session_id"]


def test_summarize_with_unknown_session_id_falls_back_to_trigger_message():
    """A SUMMARIZE request whose session_id was never seen before (no prior /chat call)
    must not error -- SessionStore.transcript() returns "" for unknown ids, and main.py
    falls back to the raw trigger message in that case (see main.py's chat())."""
    with (
        patch("app.main.chat_model", return_value=MagicMock()),
        patch("app.main.summarize_model", return_value=_FakeChatModel(_summarize_reply)),
        patch("app.main.router_chat_model", return_value=MagicMock()),
        patch("app.main.is_flagged", return_value=False),
    ):
        from app.main import app

        with TestClient(app) as client:
            response = client.post("/chat", json={"message": "tldr", "session_id": "never-seen-before"})

    assert response.status_code == 200
    body = response.json()
    assert body["agent"] == "summarize"
    assert "tldr" in body["reply"].lower()
