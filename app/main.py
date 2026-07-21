"""FastAPI app.

Endpoints:
  POST /chat        -> {"reply": "..."} — routed to a named agent (membership_registration,
                        support, or general) via app.router.Router
  GET  /faq/search   -> raw vector-search hits against the FAQ store, no LLM
"""

import logging

from fastapi import FastAPI
from pydantic import BaseModel

from app import config
from app.agents import GENERAL_SYSTEM_PROMPT, MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT, SUPPORT_SYSTEM_PROMPT
from app.ai_config import chat_model, router_chat_model
from app.assistant import Assistant
from app.faq_loader import load_faq_knowledge
from app.migrate import run_migrations
from app.router import Route, Router
from app.tools.faq import TOOLS as FAQ_TOOLS, search_faq_raw

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="studio-chatbot")

_assistants: dict[Route, Assistant] = {}
_router: Router | None = None


@app.on_event("startup")
def startup() -> None:
    global _router

    run_migrations()
    load_faq_knowledge()

    model = chat_model()
    _assistants[Route.MEMBERSHIP_REGISTRATION] = Assistant(model, MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT, FAQ_TOOLS)
    _assistants[Route.SUPPORT] = Assistant(model, SUPPORT_SYSTEM_PROMPT, FAQ_TOOLS)
    _assistants[Route.GENERAL] = Assistant(model, GENERAL_SYSTEM_PROMPT, FAQ_TOOLS)

    _router = Router(router_chat_model())

    log.info("studio-chatbot started (profile=%s)", config.PROFILE)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    agent: str
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    route = _router.route(request.message)
    reply = _assistants[route].chat(request.message)
    return ChatResponse(agent=route.value.lower(), reply=reply)


class FaqSnippet(BaseModel):
    section: str | None = None
    source: str | None = None
    text: str


@app.get("/faq/search", response_model=list[FaqSnippet])
def faq_search(q: str, topK: int | None = None) -> list[FaqSnippet]:
    return search_faq_raw(q, topK)
