"""Orchestrator agent setup for causal-ai."""

from __future__ import annotations

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver

from causal_ai.config import Config
from causal_ai.subagents import build_subagent_list


class _SafeMemorySaver(MemorySaver):
    """MemorySaver that skips non-serializable LangGraph runtime objects
    (e.g. Send) produced when tools return Command."""

    def put_writes(self, config, writes, task_id, task_path=""):
        serializable = []
        for channel, value in writes:
            try:
                self.serde.dumps_typed(value)
                serializable.append((channel, value))
            except TypeError:
                pass
        super().put_writes(config, serializable, task_id, task_path)

    def put(self, config, checkpoint, metadata, new_versions):
        safe_versions = {}
        channel_values = checkpoint.get("channel_values", {})
        for k, v in new_versions.items():
            if k in channel_values:
                try:
                    self.serde.dumps_typed(channel_values[k])
                    safe_versions[k] = v
                except TypeError:
                    pass
            else:
                safe_versions[k] = v
        return super().put(config, checkpoint, metadata, safe_versions)

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"


def _load_orchestrator_prompt() -> str:
    """Load the orchestrator system prompt."""
    return (PROMPTS_DIR / "orchestrator.md").read_text(encoding="utf-8")


def _build_model(config: Config):
    """Build the LLM model instance from config."""
    model_str = config.model
    kwargs = {}

    # OpenAI reasoning models: set reasoning effort
    if model_str.startswith("openai:"):
        kwargs["reasoning"] = {"effort": "medium"}
        kwargs["use_responses_api"] = True

    return init_chat_model(model_str, **kwargs)


def build_backend(config: Config) -> CompositeBackend:
    """Build the CompositeBackend from config."""
    project_backend = FilesystemBackend(root_dir=".", virtual_mode=True)

    if config.data_path is not None:
        return CompositeBackend(
            default=FilesystemBackend(
                root_dir=str(config.data_path), virtual_mode=True
            ),
            routes={"/project/": project_backend},
        )

    return CompositeBackend(default=project_backend, routes={})


def create_orchestrator(config: Config):
    """Create the main orchestrator agent with all sub-agents."""
    backend = build_backend(config)
    subagent_list = build_subagent_list(config)
    system_prompt = _load_orchestrator_prompt()
    model = _build_model(config)

    return create_deep_agent(
        model=model,
        system_prompt=system_prompt,
        subagents=subagent_list,
        backend=backend,
        checkpointer=_SafeMemorySaver(),
    )
