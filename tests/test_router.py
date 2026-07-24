"""Mocked unit test for Router — mocks the LangChain chat model directly (no API key,
no DB), asserting the keyword short-circuit never calls the model."""

from unittest.mock import MagicMock

from app.router import Route, Router


def _router_with_mock():
    mock_model = MagicMock()
    return Router(mock_model), mock_model


# "join"/"register" are keyword short-circuits (app.router's keyword lists) -- these must
# route without ever invoking the LLM classifier, both for cost and for determinism.
def test_join_keyword_routes_to_membership_registration_without_calling_model():
    router, mock_model = _router_with_mock()
    assert router.route("I want to join the studio") == Route.MEMBERSHIP_REGISTRATION
    mock_model.invoke.assert_not_called()


def test_register_keyword_routes_to_membership_registration():
    router, mock_model = _router_with_mock()
    assert router.route("How do I register for the yoga class?") == Route.MEMBERSHIP_REGISTRATION
    mock_model.invoke.assert_not_called()


# "tldr"/"resolved" are the SUMMARIZE-route trigger words (see test_chat_session_followup.py
# for what happens to the actual conversation content once this route is picked).
def test_summarize_keyword_routes_to_summarize_without_calling_model():
    router, mock_model = _router_with_mock()
    assert router.route("Can you tldr this for me?") == Route.SUMMARIZE
    mock_model.invoke.assert_not_called()


def test_resolved_keyword_routes_to_summarize_without_calling_model():
    router, mock_model = _router_with_mock()
    assert router.route("This is resolved, thanks!") == Route.SUMMARIZE
    mock_model.invoke.assert_not_called()


# No keyword matches -- must fall through to the LLM classifier, and its output must be
# trusted (SUPPORT here) rather than silently overridden.
def test_ambiguous_message_falls_back_to_classifier():
    router, mock_model = _router_with_mock()
    mock_model.invoke.return_value = MagicMock(content="SUPPORT")
    assert router.route("Do I need to bring my own mat?") == Route.SUPPORT
    mock_model.invoke.assert_called_once()


# Guards against a bad/garbled LLM response crashing the request -- must degrade to the
# safest default route (GENERAL) rather than raising or routing somewhere more specialized.
def test_unparseable_classifier_output_defaults_to_general():
    router, mock_model = _router_with_mock()
    mock_model.invoke.return_value = MagicMock(content="not a real label")
    assert router.route("tell me something") == Route.GENERAL


def test_empty_message_is_general_without_calling_model():
    router, mock_model = _router_with_mock()
    assert router.route("") == Route.GENERAL
    mock_model.invoke.assert_not_called()
