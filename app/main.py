"""FastAPI app.

Endpoints:
  POST /chat        -> {"reply": "...", "session_id": "..."} — the incoming message is
                        checked against OpenAI's Moderation API first (app.moderation);
                        flagged messages get a 400 without reaching any agent. Otherwise
                        routed to a named agent (membership_registration, support, general,
                        or summarize) via app.router.Router. Pass back the returned
                        session_id on subsequent
                        calls to keep them in the same conversation (needed so SUMMARIZE has
                        history to summarize -- e.g. saying "resolved" summarizes the session
                        so far, not the literal word "resolved"). Every turn is logged
                        (session_id, route, message, reply) for debugging/observability,
                        instead of exposing retrieval internals over the network.
  GET  /internal/faq/search -> raw vector-search hits against the FAQ store, no LLM.
                        /internal prefix marks it as not meant for public/end-user traffic
                        (widgets, evals, debugging) -- no auth enforced on it yet, see
                        "Next steps" in README.
"""

import logging
import uuid
from contextlib import asynccontextmanager
from typing import TypedDict

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from app.agents import (
    GENERAL_SYSTEM_PROMPT,
    MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    SUPPORT_SYSTEM_PROMPT,
)
from app.ai_config import chat_model, router_chat_model, summarize_model
from app.assistant import Assistant
from app.config import settings
from app.faq_loader import load_faq_knowledge
from app.moderation import is_flagged
from app.router import Route, Router
from app.session_store import SessionStore
from app.tools.faq import TOOLS as FAQ_TOOLS, search_faq_raw

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class AppState(TypedDict):
    assistants: dict[Route, Assistant]
    router: Router
    sessions: SessionStore


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan: runs startup work once, then shares built state via request.state.

    Runs FAQ ingestion, builds the shared chat model and one Assistant per Route, and
    constructs the Router and SessionStore -- all yielded as a dict so each field is
    reachable as request.state.<key> in endpoint handlers.

    Args:
        app: The FastAPI app instance (required by the lifespan protocol, unused here).

    Yields:
        An AppState dict: {"assistants": ..., "router": ..., "sessions": ...}.
    """
    load_faq_knowledge()

    model = chat_model()
    assistants = {
        Route.MEMBERSHIP_REGISTRATION: Assistant(model, MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT, FAQ_TOOLS),
        Route.SUPPORT: Assistant(model, SUPPORT_SYSTEM_PROMPT, FAQ_TOOLS),
        Route.GENERAL: Assistant(model, GENERAL_SYSTEM_PROMPT, FAQ_TOOLS),
        # No FAQ tool -- summarizing the user's own text shouldn't be grounded in studio FAQ.
        Route.SUMMARIZE: Assistant(summarize_model(), SUMMARIZE_SYSTEM_PROMPT),
    }
    router = Router(router_chat_model())
    sessions = SessionStore()

    log.info("studio-chatbot started (profile=%s)", settings.app_profile)
    yield {"assistants": assistants, "router": router, "sessions": sessions}


app = FastAPI(title="studio-chatbot", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    agent: str
    reply: str
    session_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
    """Routes a chat message to a named agent and returns its reply.

    Args:
        chat_request: The request body -- message text plus an optional session_id to
            continue an existing conversation (a new one is generated if omitted).
        request: The FastAPI request, used to reach the shared router/assistants/sessions
            built once in lifespan() (request.state).

    Returns:
        A ChatResponse with which agent handled it, its reply, and the session_id to
        pass back on the next call in this conversation.

    Raises:
        HTTPException: 400 if the message is flagged by OpenAI's Moderation API.
    """
    if is_flagged(chat_request.message):
        log.warning("chat message flagged by moderation, session=%s", chat_request.session_id)
        raise HTTPException(status_code=400, detail="Message flagged by moderation.")

    session_id = chat_request.session_id or str(uuid.uuid4())
    sessions = request.state.sessions
    route = request.state.router.route(chat_request.message)

    if route == Route.SUMMARIZE:
        # Summarize the session's history so far, not the literal trigger message
        # ("resolved", "tldr", ...) -- that message carries no content of its own.
        transcript = sessions.transcript(session_id)
        reply = request.state.assistants[route].chat(transcript or chat_request.message)
    else:
        reply = request.state.assistants[route].chat(chat_request.message)

    sessions.append_turn(session_id, chat_request.message, reply)
    log.info(
        "chat session=%s route=%s message=%r reply=%r",
        session_id,
        route.value,
        chat_request.message,
        reply,
    )
    return ChatResponse(agent=route.value.lower(), reply=reply, session_id=session_id)


class FaqSnippet(BaseModel):
    section: str | None = None
    source: str | None = None
    text: str


@app.get("/internal/faq/search", response_model=list[FaqSnippet])
def faq_search(q: str, topK: int | None = None) -> list[FaqSnippet]:
    """Runs a raw FAQ similarity search, bypassing the LLM/router entirely.

    Args:
        q: The search query string.
        topK: Requested number of results; see app.tools.faq.search_faq_raw for
            default/clamping behavior.

    Returns:
        Matched FAQ snippets ordered by similarity.
    """
    return search_faq_raw(q, topK)
