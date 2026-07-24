"""Runs evals/faq_eval_dataset.py's FAQ RAGAS cases as a LangSmith experiment -- same
target (SUPPORT agent) and metrics (faithfulness, answer_relevancy, context_precision) as
evals/test_ragas_faq.py, but logged to LangSmith per-example so repeat runs can be compared
as experiments in the UI (checking RAGAS score consistency run-to-run, not just a single
pass/fail).

Needs a live DB (pgvector) + OPENAI_API_KEY + LANGCHAIN_API_KEY (same as
evals/test_ragas_faq.py; see .env). Must run under Python 3.13 (ragas/nest_asyncio is
incompatible with 3.14's asyncio.timeout()) -- use .venv-py313.

Run:
    .venv-py313/bin/python -m scripts.langsmith_ragas
"""

from datasets import Dataset
from langsmith import Client
from langsmith.evaluation import evaluate
from ragas import evaluate as ragas_evaluate
from ragas.metrics import answer_relevancy, context_precision, faithfulness

from app.agents.support import SUPPORT_SYSTEM_PROMPT
from app.ai_config import chat_model
from app.assistant import Assistant
from app.tools.faq import TOOLS as FAQ_TOOLS, search_faq_raw
from evals.faq_eval_dataset import FAQ_EVAL_CASES

DATASET_NAME = "studio-chatbot-faq-ragas"

client = Client()


def _build_dataset():
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            DATASET_NAME,
            description=(
                "evals/faq_eval_dataset.py's FAQ_EVAL_CASES, run against the SUPPORT agent "
                "and scored with RAGAS (faithfulness/answer_relevancy/context_precision) -- "
                "logged as a LangSmith experiment per run to compare score consistency."
            ),
        )

    existing = {ex.inputs.get("question"): ex for ex in client.list_examples(dataset_id=dataset.id)}
    for case in FAQ_EVAL_CASES:
        inputs = {"question": case["question"]}
        outputs = {"ground_truth": case["ground_truth"]}
        if case["question"] not in existing:
            client.create_example(dataset_id=dataset.id, inputs=inputs, outputs=outputs)
        elif existing[case["question"]].outputs != outputs:
            client.update_example(existing[case["question"]].id, inputs=inputs, outputs=outputs)

    return dataset


def _target(inputs: dict) -> dict:
    assistant = Assistant(chat_model(), SUPPORT_SYSTEM_PROMPT, FAQ_TOOLS)
    reply = assistant.chat(inputs["question"])
    contexts = [hit["text"] for hit in search_faq_raw(inputs["question"])]
    return {"reply": reply, "contexts": contexts}


def _ragas_scores(run, example) -> list[dict]:
    """Scores one example with real RAGAS metrics -- a single-row Dataset through the same
    ragas.evaluate() call evals/test_ragas_faq.py uses, so this is the identical scoring
    path, just logged to LangSmith instead of asserted against a threshold.

    Tracing is disabled around the nested ragas_evaluate() call -- this evaluator itself
    runs inside our own outer evaluate()'s traced context, and ragas' internal callback
    tracer (ragas/callbacks.py's parse_run_traces) crashes with an IndexError if it picks up
    that outer trace instead of starting its own clean one."""
    import os

    prev = os.environ.pop("LANGCHAIN_TRACING_V2", None)
    try:
        row = Dataset.from_dict(
            {
                "question": [example.inputs["question"]],
                "answer": [run.outputs["reply"]],
                "contexts": [run.outputs["contexts"]],
                "ground_truth": [example.outputs["ground_truth"]],
            }
        )
        scores = ragas_evaluate(row, metrics=[faithfulness, answer_relevancy, context_precision]).to_pandas().iloc[0]
    finally:
        if prev is not None:
            os.environ["LANGCHAIN_TRACING_V2"] = prev
    return [
        {"key": "faithfulness", "score": float(scores["faithfulness"])},
        {"key": "answer_relevancy", "score": float(scores["answer_relevancy"])},
        {"key": "context_precision", "score": float(scores["context_precision"])},
    ]


def main():
    dataset = _build_dataset()
    results = evaluate(
        _target,
        data=dataset.name,
        evaluators=[_ragas_scores],
        experiment_prefix="faq-ragas",
    )
    print(results)


if __name__ == "__main__":
    main()
