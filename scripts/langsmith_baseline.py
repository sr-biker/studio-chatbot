"""Uploads the existing eval cases (evals/faq_eval_dataset.py + evals/test_llm_judge_evals.py)
as a LangSmith dataset, then runs the current system prompts against it as a baseline
experiment. Future system-prompt edits can be compared against this baseline as new
experiments in the LangSmith UI, instead of only a pass/fail CI signal.

Needs a live DB (pgvector) + OPENAI_API_KEY + LANGCHAIN_API_KEY (all already required by
evals/test_llm_judge_evals.py; see .env).

Run:
    python -m scripts.langsmith_baseline
"""

from langsmith import Client
from langsmith.evaluation import evaluate

from app.agents import (
    GENERAL_SYSTEM_PROMPT,
    MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT,
    SUPPORT_SYSTEM_PROMPT,
)
from app.ai_config import chat_model
from app.assistant import Assistant
from app.tools.faq import TOOLS as FAQ_TOOLS
from app.tools.studio_api import TOOLS as MEMBER_LOOKUP_TOOLS
from evals.faq_eval_dataset import FAQ_EVAL_CASES
from evals.test_llm_judge_evals import JUDGE_CASES, JUDGE_PROMPT

DATASET_NAME = "studio-chatbot-system-prompt-eval"

ROUTE_PROMPTS = {
    "SUPPORT": (SUPPORT_SYSTEM_PROMPT, FAQ_TOOLS),
    "GENERAL": (GENERAL_SYSTEM_PROMPT, FAQ_TOOLS),
    "MEMBERSHIP_REGISTRATION": (MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT, FAQ_TOOLS + MEMBER_LOOKUP_TOOLS),
}

client = Client()


def _desired_examples():
    """The full set of (inputs, outputs, metadata) this script wants in the dataset --
    same shape client.create_example/update_example expect, keyed below by (question,
    route) so a rerun can diff against what's already there."""
    examples = []
    for case in FAQ_EVAL_CASES:
        for route in ROUTE_PROMPTS:
            examples.append(
                {
                    "inputs": {"question": case["question"], "route": route},
                    "outputs": {"eval_type": "groundedness", "ground_truth": case["ground_truth"]},
                    "metadata": {"system_prompt": ROUTE_PROMPTS[route][0]},
                }
            )
    for case in JUDGE_CASES:
        for route in case["routes"]:
            examples.append(
                {
                    "inputs": {"question": case["input"], "route": route},
                    "outputs": {"eval_type": "judge", "rubric": case["rubric"]},
                    "metadata": {"system_prompt": ROUTE_PROMPTS[route][0]},
                }
            )
    return examples


def _key(inputs: dict) -> tuple:
    return (inputs.get("question"), inputs.get("route"))


def _build_dataset():
    """Upserts the dataset in place -- same dataset_id across runs, so past experiments
    stay comparable in the LangSmith UI (a delete+recreate would orphan every prior
    experiment's project, since experiments are tied to the dataset they ran against).
    Existing examples get their outputs/metadata updated if a case's ground_truth/rubric or
    a route's system_prompt changed; examples no longer in evals/ are deleted; new cases are
    added."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            DATASET_NAME,
            description=(
                "FAQ groundedness cases (evals/faq_eval_dataset.py) and LLM-judge rubric cases "
                "(evals/test_llm_judge_evals.py), expanded per route -- the same cases evals/ "
                "runs in CI, uploaded here so system-prompt edits can be compared as LangSmith "
                "experiments instead of only pass/fail. Each example's metadata.system_prompt is "
                "the exact prompt text run for its route, so it's visible in the dataset/"
                "experiment view without opening a trace."
            ),
        )

    existing = {_key(ex.inputs): ex for ex in client.list_examples(dataset_id=dataset.id)}
    desired = {_key(ex["inputs"]): ex for ex in _desired_examples()}

    for key, ex in desired.items():
        if key not in existing:
            client.create_example(dataset_id=dataset.id, **ex)
        else:
            current = existing[key]
            if current.outputs != ex["outputs"] or (current.metadata or {}).get("system_prompt") != ex["metadata"][
                "system_prompt"
            ]:
                # inputs is passed even though it's unchanged -- update_example does not
                # treat an omitted field as "leave as-is", it clears it, which previously
                # wiped an example's inputs to {} and broke every later run against it.
                client.update_example(current.id, inputs=ex["inputs"], outputs=ex["outputs"], metadata=ex["metadata"])

    for key, ex in existing.items():
        if key not in desired:
            client.delete_example(ex.id)

    return dataset


def _target(inputs: dict) -> dict:
    prompt, tools = ROUTE_PROMPTS[inputs["route"]]
    assistant = Assistant(chat_model(), prompt, tools)
    return {"reply": assistant.chat(inputs["question"])}


def _judge_pass(run, example) -> dict:
    """Single evaluator dispatching on eval_type -- groundedness cases get a fact-match
    judge prompt, judge cases reuse evals/test_llm_judge_evals.py's own JUDGE_PROMPT, so the
    LangSmith baseline grades identically to what CI already asserts."""
    outputs = example.outputs
    if not run.outputs or "reply" not in run.outputs:
        # The target call itself errored (timeout, rate limit, etc.) -- report it as a
        # scored failure rather than raising here and losing the row from the results.
        return {"key": "pass", "score": False, "comment": f"target error: {run.error or 'no output'}"}
    reply = run.outputs["reply"]
    judge = chat_model()

    if outputs["eval_type"] == "groundedness":
        prompt = (
            f"Ground truth: {outputs['ground_truth']}\n"
            f"Agent reply: {reply}\n\n"
            'Does the reply convey the same facts as the ground truth (paraphrasing is fine)? '
            'Answer with exactly one line: "PASS: <reason>" or "FAIL: <reason>".'
        )
    else:
        prompt = JUDGE_PROMPT.format(input=example.inputs["question"], reply=reply, rubric=outputs["rubric"])

    verdict = judge.invoke(prompt).content
    return {"key": "pass", "score": verdict.strip().upper().startswith("PASS"), "comment": verdict}


def main():
    dataset = _build_dataset()
    results = evaluate(
        _target,
        data=dataset.name,
        evaluators=[_judge_pass],
        experiment_prefix="baseline-system-prompts",
    )
    print(results)


if __name__ == "__main__":
    main()
