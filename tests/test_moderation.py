"""Mocked unit test for app.moderation — mocks the OpenAI client directly (no API key,
no network call)."""

from unittest.mock import MagicMock, patch

from app.moderation import is_flagged


def _mock_response(flagged: bool):
    result = MagicMock()
    result.flagged = flagged
    response = MagicMock()
    response.results = [result]
    return response


def test_is_flagged_true_when_moderation_flags_text():
    with patch("app.moderation.moderation_client") as mock_client_fn:
        mock_client_fn.return_value.moderations.create.return_value = _mock_response(True)
        assert is_flagged("some text") is True


def test_is_flagged_false_when_moderation_clears_text():
    with patch("app.moderation.moderation_client") as mock_client_fn:
        mock_client_fn.return_value.moderations.create.return_value = _mock_response(False)
        assert is_flagged("some text") is False
