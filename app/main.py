"""FastAPI app.

Endpoints:
  POST /chat        -> {"reply": "..."} — routed to a named agent (membership_registration,
                        support, or general) via app.router.Router
  GET  /faq/search   -> raw vector-search hits against the FAQ store, no LLM
"""

import logging
from contextlib import asynccontextmanager
from typing import TypedDict

from fastapi import FastAPI, Request
from pydantic import BaseModel

from app.agents import GENERAL_SYSTEM_PROMPT, MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT, SUPPORT_SYSTEM_PROMPT
from app.ai_config import chat_model, router_chat_model
from app.assistant import Assistant
from app.config import settings
from app.faq_loader import load_faq_knowledge
from app.migrate import run_migrations
from app.router import Route, Router
from app.tools.faq import TOOLS as FAQ_TOOLS, search_faq_raw

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class AppState(TypedDict):
    assistants: dict[Route, Assistant]
    router: Router


@asynccontextmanager
async def lifespan(app: FastAPI):
    run_migrations()
    load_faq_knowledge()

    model = chat_model()
    assistants = {
        Route.MEMBERSHIP_REGISTRATION: Assistant(model, MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT, FAQ_TOOLS),
        Route.SUPPORT: Assistant(model, SUPPORT_SYSTEM_PROMPT, FAQ_TOOLS),
        Route.GENERAL: Assistant(model, GENERAL_SYSTEM_PROMPT, FAQ_TOOLS),
    }
    router = Router(router_chat_model())

    log.info("studio-chatbot started (profile=%s)", settings.app_profile)
    yield {"assistants": assistants, "router": router}


app = FastAPI(title="studio-chatbot", lifespan=lifespan)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    agent: str
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest, request: Request) -> ChatResponse:
    route = request.state.router.route(chat_request.message)
    reply = request.state.assistants[route].chat(chat_request.message)
    return ChatResponse(agent=route.value.lower(), reply=reply)


class FaqSnippet(BaseModel):
    section: str | None = None
    source: str | None = None
    text: str


@app.get("/faq/search", response_model=list[FaqSnippet])
def faq_search(q: str, topK: int | None = None) -> list[FaqSnippet]:
    return search_faq_raw(q, topK)
