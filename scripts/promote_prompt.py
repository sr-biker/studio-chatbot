"""Gates a CSR's in-progress LangSmith Hub prompt edit before it can reach prod: pulls the
prompt's "latest" commit, runs it through the same dataset/evaluator
scripts/langsmith_baseline.py uses (filtered to the one route this prompt belongs to), plus
a RAGAS faithfulness/answer-relevancy/context-precision check (evals/test_ragas_faq.py's
thresholds), and reports the commit hash to pin as settings.<route>_prompt_ref (via
values-prod.yaml) if both look good. Promotion itself is a manual config change, not
automated here -- this script's job is only to tell you whether the pending edit is safe to
promote.

The judge-based pass rate alone isn't enough: it only checks "does the reply still convey
the right facts," so a prompt that answers correctly but in a wildly off-tone or padded way
(verified with a pirate-slang prompt: judge scored it 10/10) can still tank RAGAS's
answer_relevancy, which measures how on-topic the actual reply text is. Run both.

Needs a live DB (pgvector) + OPENAI_API_KEY + LANGCHAIN_API_KEY (see .env). The RAGAS check
needs the .venv-py313 interpreter (ragas/nest_asyncio don't work on 3.14) -- run this whole
script with that interpreter:

Run:
    .venv-py313/bin/python -m scripts.promote_prompt studio-chatbot-support
    .venv-py313/bin/python -m scripts.promote_prompt studio-chatbot-general
    .venv-py313/bin/python -m scripts.promote_prompt studio-chatbot-membership-registration
"""

import sys

from langsmith import Client
from langsmith.evaluation import evaluate

from app.ai_config import chat_model
from app.assistant import Assistant
from app.tools.faq import TOOLS as FAQ_TOOLS, search_faq_raw
from app.tools.studio_api import TOOLS as MEMBER_LOOKUP_TOOLS
from evals.faq_eval_dataset import FAQ_EVAL_CASES
from scripts.langsmith_baseline import DATASET_NAME, _judge_pass

PROMPT_TO_ROUTE = {
    "studio-chatbot-support": "SUPPORT",
    "studio-chatbot-general": "GENERAL",
    "studio-chatbot-membership-registration": "MEMBERSHIP_REGISTRATION",
}

ROUTE_TOOLS = {
    "SUPPORT": FAQ_TOOLS,
    "GENERAL": FAQ_TOOLS,
    "MEMBERSHIP_REGISTRATION": FAQ_TOOLS + MEMBER_LOOKUP_TOOLS,
}

MIN_PASS_RATE = 0.9

# Same bar as evals/test_ragas_faq.py -- kept in sync manually since that file is pytest-only.
MIN_FAITHFULNESS = 0.8
MIN_ANSWER_RELEVANCY = 0.8
MIN_CONTEXT_PRECISION = 0.8

client = Client()


def _ragas_scores(prompt_text: str, route: str) -> dict:
    """Runs FAQ_EVAL_CASES through prompt_text and scores the replies with RAGAS.

    Args:
        prompt_text: The candidate system prompt text (the Hub's "latest" commit).
        route: The route this prompt is for -- picks which tools the assistant gets.

    Returns:
        {"faithfulness": ..., "answer_relevancy": ..., "context_precision": ...}
    """
    from datasets import Dataset
    from ragas import evaluate as ragas_evaluate
    from ragas.metrics import answer_relevancy, context_precision, faithfulness

    assistant = Assistant(chat_model(), prompt_text, ROUTE_TOOLS[route])

    questions, answers, contexts, ground_truths = [], [], [], []
    for case in FAQ_EVAL_CASES:
        answers.append(assistant.chat(case["question"]))
        questions.append(case["question"])
        ground_truths.append(case["ground_truth"])
        contexts.append([hit["text"] for hit in search_faq_raw(case["question"])])

    dataset = Dataset.from_dict(
        {"question": questions, "answer": answers, "contexts": contexts, "ground_truth": ground_truths}
    )
    result = ragas_evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])
    return result.to_pandas().mean(numeric_only=True).to_dict()


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in PROMPT_TO_ROUTE:
        print(f"Usage: python -m scripts.promote_prompt <{'|'.join(PROMPT_TO_ROUTE)}>")
        sys.exit(1)

    prompt_name = sys.argv[1]
    route = PROMPT_TO_ROUTE[prompt_name]

    commit = client.pull_prompt_commit(f"{prompt_name}:latest")
    prompt_text = client.pull_prompt(f"{prompt_name}:latest").messages[0].prompt.template
    print(f"Evaluating {prompt_name}:latest (commit {commit.commit_hash}) against route={route}...")

    def target(inputs: dict) -> dict:
        if inputs["route"] != route:
            return {"reply": None}  # filtered out below, never scored
        assistant = Assistant(chat_model(), prompt_text, ROUTE_TOOLS[route])
        return {"reply": assistant.chat(inputs["question"])}

    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[_judge_pass],
        experiment_prefix=f"promote-{prompt_name}",
        metadata={"prompt_commit": commit.commit_hash},
    )

    scores = [r["evaluation_results"]["results"][0].score for r in results if r["example"].inputs["route"] == route]
    pass_rate = sum(scores) / len(scores) if scores else 0.0
    print(f"Judge pass rate for route={route}: {pass_rate:.2f} ({sum(scores)}/{len(scores)})")

    print("\nRunning RAGAS (faithfulness / answer_relevancy / context_precision)...")
    ragas_scores = _ragas_scores(prompt_text, route)
    for key, value in ragas_scores.items():
        print(f"  {key}: {value:.3f}")

    judge_ok = pass_rate >= MIN_PASS_RATE
    ragas_ok = (
        ragas_scores["faithfulness"] >= MIN_FAITHFULNESS
        and ragas_scores["answer_relevancy"] >= MIN_ANSWER_RELEVANCY
        and ragas_scores["context_precision"] >= MIN_CONTEXT_PRECISION
    )

    if judge_ok and ragas_ok:
        print(
            f"\nOK to promote. Pin this in values-prod.yaml:\n"
            f"  {route.lower()}PromptRef: {commit.commit_hash}"
        )
    else:
        failed = []
        if not judge_ok:
            failed.append(f"judge pass rate below {MIN_PASS_RATE:.0%}")
        if not ragas_ok:
            failed.append("RAGAS below threshold")
        print(f"\nDo not promote yet -- {', '.join(failed)}.")


if __name__ == "__main__":
    main()
