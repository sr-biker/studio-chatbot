"""A minimal tool-calling chat loop shared by every named agent — each agent is just this
loop bound to a different system prompt (and, for now, the same FAQ retrieval tool)."""

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

MAX_TOOL_ITERATIONS = 4


class Assistant:
    def __init__(self, chat_model, system_prompt: str, tools: list | None = None):
        self._system_prompt = system_prompt
        self._tools_by_name = {t.name: t for t in (tools or [])}
        self._model = chat_model.bind_tools(tools) if tools else chat_model

    def chat(self, user_message: str) -> str:
        messages = [SystemMessage(content=self._system_prompt), HumanMessage(content=user_message)]

        for _ in range(MAX_TOOL_ITERATIONS):
            response: AIMessage = self._model.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content

            for call in response.tool_calls:
                tool = self._tools_by_name[call["name"]]
                result = tool.invoke(call["args"])
                messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

        return messages[-1].content
