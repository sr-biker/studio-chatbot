"""RAGAS offline eval: faithfulness, answer-relevancy, and context-precision for the FAQ
RAG path (the retrieval + generation loop backing the SUPPORT / MEMBERSHIP_REGISTRATION
agents). Not part of the default `pytest` run -- gated behind the RUN_RAGAS_EVALS env var
(and requires a live DB + OPENAI_API_KEY) so CI runs it as an explicit, separate job rather
than on every mocked test invocation.

Run:
    RUN_RAGAS_EVALS=1 OPENAI_API_KEY=... pytest evals/eval_ragas_faq.py -q
"""

import os
from functools import lru_cache

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_RAGAS_EVALS"),
    reason="RAGAS evals need a live DB + OPENAI_API_KEY; set RUN_RAGAS_EVALS=1 to run",
)

# Minimum scores below which a regression should fail CI. Thresholds are loose (RAG over a
# 35-item FAQ with a small local model is not going to hit 0.95+ reliably) -- the point is
# catching a broken retrieval path or a badly regressed prompt, not chasing a leaderboard.
MIN_FAITHFULNESS = 0.8
MIN_ANSWER_RELEVANCY = 0.8
MIN_CONTEXT_PRECISION = 0.8


@lru_cache
def _build_ragas_dataset():
    from datasets import Dataset

    from app.agents.support import SUPPORT_SYSTEM_PROMPT
    from app.ai_config import chat_model
    from app.assistant import Assistant
    from app.faq_loader import load_faq_knowledge
    from app.tools.faq import TOOLS as FAQ_TOOLS, search_faq_raw
    from evals.faq_eval_dataset import FAQ_EVAL_CASES

    load_faq_knowledge()

    assistant = Assistant(chat_model(), SUPPORT_SYSTEM_PROMPT, FAQ_TOOLS)

    questions, answers, contexts, ground_truths = [], [], [], []
    for case in FAQ_EVAL_CASES:
        answers.append(assistant.chat(case["question"]))
        questions.append(case["question"])
        ground_truths.append(case["ground_truth"])
        contexts.append([hit["text"] for hit in search_faq_raw(case["question"])])

    return Dataset.from_dict(
        {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}
    )


def test_faq_rag_meets_ragas_thresholds():
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    dataset = _build_ragas_dataset()
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    scores = result.to_pandas().mean(numeric_only=True)

    assert scores["faithfulness"] >= MIN_FAITHFULNESS, scores
    assert scores["answer_relevancy"] >= MIN_ANSWER_RELEVANCY, scores
    assert scores["context_precision"] >= MIN_CONTEXT_PRECISION, scores
