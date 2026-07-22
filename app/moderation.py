"""Runtime safety guardrail via OpenAI's Moderation API -- checked on the incoming chat
message before it's routed/answered. Distinct from evals/ (offline, CI-gated quality
checks against the FAQ, not a safety filter in the request path -- see README)."""

from functools import lru_cache

from openai import OpenAI

from app.config import settings


@lru_cache
def moderation_client() -> OpenAI:
    """Builds (and memoizes) the OpenAI client used for moderation checks.

    Returns:
        An OpenAI client configured from settings.openai_api_key.
    """
    return OpenAI(api_key=settings.openai_api_key)


def is_flagged(text: str) -> bool:
    """Checks text against OpenAI's Moderation API.

    Args:
        text: The text to check (e.g. an incoming chat message).

    Returns:
        True if the moderation API flagged the text as violating policy.
    """
    result = moderation_client().moderations.create(input=text)
    return result.results[0].flagged
