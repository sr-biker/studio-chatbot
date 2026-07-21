"""Named agents the Router dispatches to — each is an Assistant bound to a
domain-specific system prompt and the FAQ retrieval tool."""

from app.agents.membership import MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT
from app.agents.support import SUPPORT_SYSTEM_PROMPT
from app.agents.general import GENERAL_SYSTEM_PROMPT

__all__ = [
    "MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT",
    "SUPPORT_SYSTEM_PROMPT",
    "GENERAL_SYSTEM_PROMPT",
]
