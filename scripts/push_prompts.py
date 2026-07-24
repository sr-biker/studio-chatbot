"""One-time (or as-needed) seed: pushes the current hardcoded system prompts to the
LangSmith Prompt Hub, so app/prompts.py's resolve_system_prompt("...", "latest", ...) has
something to pull. After this, CSR/support staff edit prompt wording directly in the Hub
(Playground or repo page) instead of asking an engineer to change app/agents/*.py.

Needs LANGCHAIN_API_KEY (see .env).

Run:
    python -m scripts.push_prompts
"""

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client

from app.agents.general import _GENERAL_SYSTEM_PROMPT_FALLBACK
from app.agents.membership import _MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT_FALLBACK
from app.agents.support import _SUPPORT_SYSTEM_PROMPT_FALLBACK

PROMPTS = {
    "studio-chatbot-support": _SUPPORT_SYSTEM_PROMPT_FALLBACK,
    "studio-chatbot-general": _GENERAL_SYSTEM_PROMPT_FALLBACK,
    "studio-chatbot-membership-registration": _MEMBERSHIP_REGISTRATION_SYSTEM_PROMPT_FALLBACK,
}


def main():
    client = Client()
    for name, text in PROMPTS.items():
        template = ChatPromptTemplate.from_messages([("system", text)])
        url = client.push_prompt(name, object=template)
        print(f"{name}: {url}")


if __name__ == "__main__":
    main()
