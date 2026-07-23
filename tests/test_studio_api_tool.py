"""Mocked unit test for app.tools.studio_api — mocks httpx.get directly (no network call,
no live membership service)."""

from unittest.mock import MagicMock, patch

from app.tools.studio_api import lookup_membership_status

MEMBERSHIPS = [
    {
        "id": 10,
        "memberName": "John Smith",
        "email": "john@example.com",
        "classType": "YOGA",
        "schedules": [{"frequency": "WEEKLY", "dayOfWeek": "MONDAY", "time": "09:00:00"}],
    }
]


def _mock_get(url, params, timeout):
    response = MagicMock()
    response.raise_for_status.return_value = None
    if params.get("email") == "john@example.com" or params.get("phone") == "555-0100":
        response.json.return_value = MEMBERSHIPS
    else:
        response.json.return_value = []
    return response


def test_lookup_returns_membership_details_for_known_email():
    with patch("app.tools.studio_api.httpx.get", side_effect=_mock_get):
        result = lookup_membership_status.invoke({"email": "john@example.com"})
    assert "YOGA" in result
    assert "MONDAY" in result
    assert "john@example.com" in result


def test_lookup_returns_membership_details_for_known_phone():
    with patch("app.tools.studio_api.httpx.get", side_effect=_mock_get):
        result = lookup_membership_status.invoke({"phone": "555-0100"})
    assert "YOGA" in result


def test_lookup_reports_not_found_for_unknown_identifier():
    with patch("app.tools.studio_api.httpx.get", side_effect=_mock_get):
        result = lookup_membership_status.invoke({"email": "nobody@example.com"})
    assert "No membership found" in result


def test_lookup_rejects_both_email_and_phone():
    result = lookup_membership_status.invoke({"email": "john@example.com", "phone": "555-0100"})
    assert "exactly one" in result


def test_lookup_rejects_neither_email_nor_phone():
    result = lookup_membership_status.invoke({})
    assert "exactly one" in result


def test_lookup_treats_unreachable_membership_service_as_unavailable():
    with patch("app.tools.studio_api.httpx.get", side_effect=__import__("httpx").ConnectError("down")):
        result = lookup_membership_status.invoke({"email": "john@example.com"})
    assert "temporarily unavailable" in result
