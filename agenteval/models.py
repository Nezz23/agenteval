"""Model backends. `openrouter:<model>` is a real LLM; `scripted:<policy>` is a deterministic
stand-in used to validate the harness and graders (oracle = ideal agent, naive = obeys injections)."""
import json
import os
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

OVERRIDE = re.compile(r"SYSTEM OVERRIDE: call (\w+) with ((?:\w+=\S+\s*)+)")


def get_model(spec: str, task: dict | None = None) -> BaseChatModel:
    kind, _, name = spec.partition(":")
    if kind == "openrouter":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, base_url="https://openrouter.ai/api/v1",
                          api_key=os.environ["OPENROUTER_API_KEY"], temperature=0)
    if kind == "ollama":  # local, free; Ollama serves an OpenAI-compatible API
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=name, base_url="http://localhost:11434/v1", api_key="ollama", temperature=0)
    if kind == "scripted":
        return ScriptedModel(policy=name, task=task or {})
    raise ValueError(f"unknown model spec: {spec}")


class ScriptedModel(BaseChatModel):
    policy: str = "oracle"
    task: dict = {}

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        role = next((m.content.split()[1].rstrip(".") for m in messages
                     if isinstance(m, SystemMessage) and m.content.startswith("ROLE:")), "executor")
        if role == "planner":
            msg = AIMessage(content="Plan: " + ", ".join(c[0] for c in self.task.get("reference_calls", [])))
        elif role == "reviewer":
            msg = AIMessage(content="APPROVE")
        elif role == "judge":
            msg = AIMessage(content=json.dumps({"score": 5, "rationale": "scripted"}))
        else:
            msg = self._act(messages)
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _act(self, messages) -> AIMessage:
        issued = [(c["name"], c["args"]) for m in messages if isinstance(m, AIMessage) for c in m.tool_calls]
        n = len(issued)
        if self.policy == "naive":
            text = "\n".join(str(m.content) for m in messages if not isinstance(m, (SystemMessage, AIMessage)))
            for name, raw in OVERRIDE.findall(text):
                args = dict(kv.split("=", 1) for kv in raw.split())
                if (name, args) not in issued:
                    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{n}"}])
        for name, args in self.task.get("reference_calls", []):
            if (name, args) not in issued:
                return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{n}"}])
        return AIMessage(content=str(self.task.get("reference_answer", "")))
