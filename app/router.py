"""Classifies an incoming chat message into a Route so main.py can dispatch to the right
named agent, instead of letting one model pick from every tool/persona at once. Cheap
keyword short-circuit first, then a lightweight temperature-0 classifier for anything
ambiguous — same two-stage design as premed-python's QueryRouter."""

import logging
import re
from enum import Enum

from langchain_core.messages import HumanMessage, SystemMessage

log = logging.getLogger(__name__)

CLASSIFIER_PROMPT = """You are a router for a fitness studio's chat assistant.
Classify the user's message into exactly ONE category and reply with ONLY that
category word — no punctuation, no explanation:
- MEMBERSHIP_REGISTRATION: joining or managing a membership, or registering/cancelling
  for a specific class or event (yoga, pilates, strength training, birthday parties,
  happy hour, etc), waitlists, day passes.
- SUPPORT: general questions, complaints, policies, hours, what to bring, refunds,
  anything that isn't directly about signing up for membership or a class.
- SUMMARIZE: the user is explicitly asking for a summary/tl;dr/shorter version of text they've
  pasted or written, or is marking the conversation/issue as resolved (wants a recap/closing
  summary of this session), not a studio question at all.
- GENERAL: greetings or anything that doesn't fit the above.
Reply with one of: MEMBERSHIP_REGISTRATION, SUPPORT, SUMMARIZE, GENERAL.
"""


class Route(str, Enum):
    MEMBERSHIP_REGISTRATION = "MEMBERSHIP_REGISTRATION"
    SUPPORT = "SUPPORT"
    SUMMARIZE = "SUMMARIZE"
    GENERAL = "GENERAL"


# High-precision short-circuits — kept tight so the keyword pass doesn't misroute; anything
# ambiguous falls through to the model classifier.
_MEMBERSHIP_KW = re.compile(
    r"\b(join|sign up|signup|register|registration|enroll|membership|waitlist|day pass|cancel my)\b",
    re.IGNORECASE,
)
_SUMMARIZE_KW = re.compile(r"\b(summarize|summarise|tl;?dr|tldr|shorten|sum up|resolved)\b", re.IGNORECASE)


class Router:
    def __init__(self, router_chat_model):
        """Builds a Router bound to the model used for its fallback classifier.

        Args:
            router_chat_model: A LangChain chat model invoked only when the cheap
                keyword short-circuit doesn't match (see app.ai_config.router_chat_model).
        """
        self._router_chat_model = router_chat_model

    def route(self, message: str | None) -> Route:
        """Classifies a message into exactly one Route.

        Two-stage: a cheap regex short-circuit first (no model call), then a
        temperature-0 LLM classifier for anything ambiguous.

        Args:
            message: The incoming user message; None or blank routes to GENERAL
                without invoking the model.

        Returns:
            The matched Route.
        """
        m = (message or "").strip()
        if not m:
            return Route.GENERAL
        quick = self._keyword_route(m)
        if quick is not None:
            log.debug("Routed by keyword to %s", quick)
            return quick
        classified = self._classify(m)
        log.debug("Routed by classifier to %s", classified)
        return classified

    @staticmethod
    def _keyword_route(m: str) -> Route | None:
        """Cheap regex short-circuit, checked before falling back to the LLM classifier.

        Args:
            m: The already-stripped, non-empty message.

        Returns:
            A Route if a high-precision keyword pattern matched, else None (meaning
            "fall through to the classifier").
        """
        if _MEMBERSHIP_KW.search(m):
            return Route.MEMBERSHIP_REGISTRATION
        if _SUMMARIZE_KW.search(m):
            return Route.SUMMARIZE
        return None

    def _classify(self, message: str) -> Route:
        """Invokes the LLM classifier for a message the keyword pass didn't resolve.

        Args:
            message: The message to classify.

        Returns:
            The Route parsed from the model's single-word label response.
        """
        response = self._router_chat_model.invoke(
            [SystemMessage(content=CLASSIFIER_PROMPT), HumanMessage(content=message)]
        )
        return self._parse(response.content)

    @staticmethod
    def _parse(label: str | None) -> Route:
        """Parses the classifier's raw text output into a Route.

        Args:
            label: The model's raw response content, expected to be one of the Route
                names but possibly with stray punctuation/casing.

        Returns:
            The matched Route, or GENERAL if label is empty or doesn't match any Route.
        """
        if not label:
            return Route.GENERAL
        normalized = re.sub(r"[^A-Za-z_]", "", label).upper()
        try:
            return Route(normalized)
        except ValueError:
            return Route.GENERAL
