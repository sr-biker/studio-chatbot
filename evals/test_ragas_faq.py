"""RAGAS offline eval: faithfulness, answer-relevancy, and context-precision for the FAQ
RAG path (the retrieval + generation loop backing the SUPPORT / MEMBERSHIP_REGISTRATION
agents). Not part of the default `pytest` run -- gated behind the RUN_RAGAS_EVALS env var
(and requires a live DB + OPENAI_API_KEY) so CI runs it as an explicit, separate job rather
than on every mocked test invocation.

Run:
    RUN_RAGAS_EVALS=1 OPENAI_API_KEY=... pytest evals/test_ragas_faq.py -q
"""

import os
from functools import lru_cache

import pandas as pd
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

# A control case's answer_relevancy is *expected* to score near 0 (see
# evals/faq_eval_dataset.py's expect_low_relevancy) -- anything above this means the judge
# failed to notice a genuine non-answer, which is the opposite failure mode from the main
# threshold above and gets its own assertion rather than being averaged into it.
MAX_CONTROL_CASE_RELEVANCY = 0.3

# Deliberately a different, stronger model than app.ai_config.chat_model() (gpt-4o-mini) --
# judge and generator sharing a model risks self-preference bias (a model rating its own
# phrasing/reasoning more favorably), and empirically gpt-4o-mini as judge was misfiring:
# RAGAS's answer_relevancy has a noncommittal-answer detector that zeroed out a perfectly
# good, on-topic reply just because it ended with "contact the front desk for more info" --
# gpt-4o as judge correctly told that apart from an actual non-answer (see the "swimming
# pool" control case above) without changing what gpt-4o-mini actually generates.
JUDGE_MODEL = "gpt-4o"


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

    questions, answers, contexts, ground_truths, control_flags = [], [], [], [], []
    for case in FAQ_EVAL_CASES:
        answers.append(assistant.chat(case["question"]))
        questions.append(case["question"])
        ground_truths.append(case["ground_truth"])
        contexts.append([hit["text"] for hit in search_faq_raw(case["question"])])
        control_flags.append(case.get("expect_low_relevancy", False))

    dataset = Dataset.from_dict(
        {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}
    )
    return dataset, control_flags


def test_faq_rag_meets_ragas_thresholds():
    from langchain_openai import ChatOpenAI
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    from app.config import settings

    dataset, control_flags = _build_ragas_dataset()
    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, api_key=settings.openai_api_key))
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision], llm=judge)
    df = result.to_pandas()

    control_mask = pd.Series(control_flags, index=df.index)
    main_scores = df[~control_mask].mean(numeric_only=True)
    control_scores = df[control_mask]

    assert main_scores["faithfulness"] >= MIN_FAITHFULNESS, main_scores
    assert main_scores["answer_relevancy"] >= MIN_ANSWER_RELEVANCY, main_scores
    assert main_scores["context_precision"] >= MIN_CONTEXT_PRECISION, main_scores

    for _, row in control_scores.iterrows():
        assert row["answer_relevancy"] <= MAX_CONTROL_CASE_RELEVANCY, row
