"""Orchestrator agent setup for causal-ai."""

from __future__ import annotations

from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend
from langchain.chat_models import init_chat_model

from causal_ai.config import Config
from causal_ai.subagents import build_subagent_list

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
    )
