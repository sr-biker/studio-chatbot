"""In-memory, per-process conversation history, keyed by session_id.

Single pod today, so in-memory is enough -- history is lost on restart/across replicas, which
is fine for "summarize this conversation" (a live-session feature) but would need a shared
store (Redis, etc.) if session affinity across pods/restarts ever mattered."""

import threading

MAX_TURNS = 50  # cap so a very long-lived session_id can't grow the transcript unbounded


class SessionStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._sessions: dict[str, list[tuple[str, str]]] = {}

    def append_turn(self, session_id: str, user_message: str, reply: str) -> None:
        with self._lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append(("user", user_message))
            turns.append(("assistant", reply))
            del turns[: -2 * MAX_TURNS]

    def transcript(self, session_id: str) -> str:
        with self._lock:
            turns = list(self._sessions.get(session_id, []))
        return "\n".join(f"{role}: {text}" for role, text in turns)
