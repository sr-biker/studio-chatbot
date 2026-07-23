"""LangChain tool that looks up a member's real registration status from the membership
service (~/projects/membership, a sibling Spring Boot app outside this repo).

Deliberately does NOT search by name. Earlier this tool fetched contacts-micro-service's
full contact list and fuzzy-matched by name -- since /chat has no authenticated caller
identity, that let anyone ask "does <any name> have a membership" and get back another
member's PII with no ownership check at all. membership's GET /api/memberships/lookup
(email or phone only, see membership's MembershipController) sandboxes this: the tool can
only resolve the exact identifier the user themselves supplies, not browse/search by name.
This narrows but doesn't fully close the gap -- there's still no proof the caller *is* the
email/phone they typed -- see app/agents/membership.py's system prompt for how that's
messaged to the user.
"""

import logging

import httpx
from langchain_core.tools import tool

from app.config import settings

log = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 5.0


def _get_json(url: str, params: dict) -> list[dict] | None:
    """GETs a JSON list from the membership service, tolerating it being unreachable.

    Args:
        url: Full URL to GET.
        params: Query parameters to send.

    Returns:
        The parsed JSON list, or None if the service errored or couldn't be reached --
        callers treat None the same as "no data available" rather than crashing the chat
        turn over an internal service being down.
    """
    try:
        response = httpx.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError:
        log.exception("Request to %s failed", url)
        return None


def _format_schedule(schedule: dict) -> str:
    """Renders one Schedule dict as a short human-readable string.

    Args:
        schedule: A schedule dict with frequency/dayOfWeek/time keys.

    Returns:
        "WEEKLY on MONDAY at 09:00" for weekly schedules, "DAILY at 09:00" for daily ones.
    """
    frequency = schedule.get("frequency", "")
    time = schedule.get("time", "")
    if frequency == "WEEKLY" and schedule.get("dayOfWeek"):
        return f"{frequency} on {schedule['dayOfWeek']} at {time}"
    return f"{frequency} at {time}"


@tool
def lookup_membership_status(email: str | None = None, phone: str | None = None) -> str:
    """Look up membership status (class type + schedule) by the exact email or phone
    number the user themselves provides -- exactly one of the two, never both, never a
    name. This tool cannot search by name; if the user hasn't given their email or phone
    yet, ask them for it first rather than guessing or passing a name in either field."""
    if (email is None) == (phone is None):
        return "Provide exactly one of email or phone, not both or neither."

    params = {"email": email} if email else {"phone": phone}
    memberships = _get_json(f"{settings.membership_api_base_url}/api/memberships/lookup", params)
    if memberships is None:
        return "Membership lookup is temporarily unavailable -- please try again shortly."
    if not memberships:
        identifier = email or phone
        return f"No membership found for '{identifier}'."

    lines = []
    for membership in memberships:
        schedules = ", ".join(_format_schedule(s) for s in membership.get("schedules", [])) or "no schedule set"
        lines.append(
            f"{membership['memberName']} ({membership['email']}) has a {membership['classType']} "
            f"membership -- schedule: {schedules}."
        )
    return "\n".join(lines)


TOOLS = [lookup_membership_status]
