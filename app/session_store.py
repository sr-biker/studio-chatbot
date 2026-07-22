"""In-memory, per-process conversation history, keyed by session_id.

Single pod today, so in-memory is enough -- history is lost on restart/across replicas, which
is fine for "summarize this conversation" (a live-session feature) but would need a shared
store (Redis, etc.) if session affinity across pods/restarts ever mattered."""

import threading

MAX_TURNS = 50  # cap so a very long-lived session_id can't grow the transcript unbounded


class SessionStore:
    def __init__(self):
        """Builds an empty, thread-safe, in-process session store."""
        self._lock = threading.Lock()
        self._sessions: dict[str, list[tuple[str, str]]] = {}

    def append_turn(self, session_id: str, user_message: str, reply: str) -> None:
        """Appends one (user message, assistant reply) turn to a session's history.

        Args:
            session_id: The session to append to; created on first use if unseen.
            user_message: The user's message for this turn.
            reply: The assistant's reply for this turn.
        """
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append(("user", user_message))
            turns.append(("assistant", reply))
            del turns[: -2 * MAX_TURNS]

    def transcript(self, session_id: str) -> str:
        """Renders a session's stored history as a single "role: text" transcript.

        Args:
            session_id: The session to read.

        Returns:
            Newline-joined "role: text" lines in chronological order, or "" if the
            session_id is unknown or has no turns yet.
        """
        with self._lock:
            turns = list(self._sessions.get(session_id, []))
        return "\n".join(f"{role}: {text}" for role, text in turns)
