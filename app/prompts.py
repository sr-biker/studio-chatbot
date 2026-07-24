"""Resolves each named agent's system prompt from the LangSmith Prompt Hub, so CSR/support
staff can edit prompt wording there without an engineer touching code or shipping a deploy.

Each prompt is pulled by a *pinned* ref (settings.<route>_prompt_ref, default "latest") --
not always "latest" -- so a CSR edit doesn't reach prod until scripts/promote_prompt.py has
run the eval suite against it and an operator bumps the pinned ref. Local dev defaults to
"latest" so iterating on a prompt in the Hub Playground is immediately visible without a
promotion step.

Falls back to the hardcoded constants in app/agents/*.py (passed in as `fallback`) if the
Hub is unreachable at startup (network blip, LANGCHAIN_API_KEY unset, prompt not yet pushed)
-- a Hub outage or first-run-before-push shouldn't take the whole chat product down."""

import logging
from functools import lru_cache

from langsmith import Client

log = logging.getLogger(__name__)


@lru_cache
def _client() -> Client:
    return Client()


def resolve_system_prompt(prompt_name: str, ref: str, fallback: str) -> str:
    """Pulls prompt_name's system message text at the given ref from the Hub.

    Args:
        prompt_name: The Hub repo name (e.g. "studio-chatbot-support").
        ref: "latest", a named tag, or a commit hash -- passed straight through as
            "<prompt_name>:<ref>" to Client.pull_prompt.
        fallback: The hardcoded prompt text to use if the pull fails for any reason.

    Returns:
        The system message content as a plain string.
    """
    try:
        template = _client().pull_prompt(f"{prompt_name}:{ref}")
        # ChatPromptTemplate with a single system message -- messages[0].prompt.template is
        # the raw string, same shape as the hardcoded SUPPORT_SYSTEM_PROMPT etc. constants.
        return template.messages[0].prompt.template
    except Exception:
        log.warning("Could not pull prompt %r (ref=%s) from LangSmith Hub; using local fallback.", prompt_name, ref)
        return fallback
