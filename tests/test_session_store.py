"""Mocked unit test for app.session_store — pure in-memory logic, no API key, no DB."""

from app.session_store import MAX_TURNS, SessionStore


def test_transcript_is_empty_for_unknown_session():
    store = SessionStore()
    assert store.transcript("never-seen-before") == ""


def test_append_turn_then_transcript_renders_role_prefixed_lines():
    store = SessionStore()
    store.append_turn("s1", "hello", "hi there")
    assert store.transcript("s1") == "user: hello\nassistant: hi there"


def test_multiple_turns_accumulate_in_order():
    store = SessionStore()
    store.append_turn("s1", "what are your hours?", "9 to 5")
    store.append_turn("s1", "on weekends too?", "yes, same hours")
    assert store.transcript("s1") == (
        "user: what are your hours?\n"
        "assistant: 9 to 5\n"
        "user: on weekends too?\n"
        "assistant: yes, same hours"
    )


def test_sessions_are_isolated_by_session_id():
    store = SessionStore()
    store.append_turn("s1", "hello from s1", "hi s1")
    store.append_turn("s2", "hello from s2", "hi s2")
    assert store.transcript("s1") == "user: hello from s1\nassistant: hi s1"
    assert store.transcript("s2") == "user: hello from s2\nassistant: hi s2"


def test_history_is_capped_at_max_turns():
    store = SessionStore()
    # One turn past the cap -- the oldest turn's pair of lines must be evicted.
    for i in range(MAX_TURNS + 1):
        store.append_turn("s1", f"message {i}", f"reply {i}")

    transcript = store.transcript("s1")
    lines = transcript.split("\n")

    assert len(lines) == 2 * MAX_TURNS
    assert "message 0" not in transcript
    assert "reply 0" not in transcript
    assert f"message {MAX_TURNS}" in transcript
    assert f"reply {MAX_TURNS}" in transcript
