"""openai/evals-style suite: model-graded routing + answer-groundedness checks over the
full /chat path (Router -> named agent -> FAQ RAG tool). Structured the way an
openai/evals YAML eval would be (id, input, ideal, grading), just expressed as pytest
instead of registering against the `oaieval` CLI -- avoids pulling in the full openai/evals
package (and its own eval-registry conventions) for a single small suite, while keeping the
same input/ideal/grading shape so it could be lifted into a real openai/evals registry
later if this suite grows.

Gated behind RUN_ROUTER_EVALS (needs a live DB + OPENAI_API_KEY), same reasoning as
evals/test_ragas_faq.py -- these hit a real chat model and real DB, not mocked unit tests.

Run:
    RUN_ROUTER_EVALS=1 OPENAI_API_KEY=... pytest evals/test_router_evals.py -q
"""

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_ROUTER_EVALS"),
    reason="Router evals need a live DB + OPENAI_API_KEY; set RUN_ROUTER_EVALS=1 to run",
)

# (input, ideal_route) -- "match" grading, same shape as an openai/evals basic-eval.
ROUTING_CASES = [
    {"input": "I want to sign up for the yoga class on Saturday", "ideal": "MEMBERSHIP_REGISTRATION"},
    {"input": "How do I join as a member?", "ideal": "MEMBERSHIP_REGISTRATION"},
    {"input": "What should I wear to a pilates class?", "ideal": "SUPPORT"},
    {"input": "Are refunds available if I cancel?", "ideal": "SUPPORT"},
    {"input": "Hi there!", "ideal": "GENERAL"},
]

# (input, must_include) -- "includes" grading over the final reply text, same shape as an
# openai/evals includes-eval. Phrases are drawn straight from data/faq.md so a passing
# result means the agent actually retrieved and used the right FAQ section.
GROUNDEDNESS_CASES = [
    {"input": "Do I need a membership to attend classes?", "must_include_any": ["day pass", "drop-in"]},
    {"input": "How early should I arrive for class?", "must_include_any": ["10", "15", "minutes"]},
]


# Unlike tests/test_router.py (mocked model, pure keyword-shortcut logic), this calls the
# real LLM classifier for every case here -- these inputs are deliberately ambiguous enough
# to miss the keyword short-circuits and actually exercise router_chat_model()'s judgment.
@pytest.mark.parametrize("case", ROUTING_CASES, ids=[c["input"] for c in ROUTING_CASES])
def test_router_matches_ideal_route(case):
    from app.ai_config import router_chat_model
    from app.router import Router

    router = Router(router_chat_model())
    assert router.route(case["input"]).value == case["ideal"]


# Deliberately a crude substring check, not a semantic one (that's RAGAS's job, see
# test_ragas_faq.py's faithfulness) -- the point is a cheap, fast tripwire: if the reply
# doesn't even contain a fact straight out of the source FAQ text, retrieval or the prompt
# broke badly enough that a subtler semantic score isn't needed to notice.
@pytest.mark.parametrize("case", GROUNDEDNESS_CASES, ids=[c["input"] for c in GROUNDEDNESS_CASES])
def test_reply_is_grounded_in_faq(case):
    from app.agents.support import SUPPORT_SYSTEM_PROMPT
    from app.ai_config import chat_model
    from app.assistant import Assistant
    from app.faq_loader import load_faq_knowledge
    from app.tools.faq import TOOLS as FAQ_TOOLS

    load_faq_knowledge()

    assistant = Assistant(chat_model(), SUPPORT_SYSTEM_PROMPT, FAQ_TOOLS)
    reply = assistant.chat(case["input"]).lower()

    assert any(phrase in reply for phrase in case["must_include_any"]), reply
