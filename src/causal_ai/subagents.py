"""Sub-agent definitions for causal-ai."""
from __future__ import annotations

import shutil
from pathlib import Path

from langchain.agents.middleware import HostExecutionPolicy, ShellToolMiddleware

from causal_ai.config import Config
from causal_ai.rag import build_search_tools

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
_VENV_BIN = Path(__file__).resolve().parent.parent.parent / ".venv" / "bin"

# Skill source paths are PARENT directories containing skill subdirectories.
# SkillsMiddleware lists the parent, discovers subdirs, and loads SKILL.md from each.
QA_SKILLS = ["/project/skills/qa/"]
CODER_SKILLS = ["/project/skills/coder/"]


def _load_prompt(name: str) -> str:
    """Load a prompt file from prompts/<name>.md."""
    return (PROMPTS_DIR / f"{name}.md").read_text()


def build_subagent_list(config: Config) -> list[dict]:
    """Return the list of sub-agent definition dicts."""
    return [
        {
            "name": "qa",
            "description": (
                "Answers causal inference questions — conceptual, "
                "methodological, or interpretive."
            ),
            "system_prompt": _load_prompt("qa"),
            "skills": QA_SKILLS,
        },
        {
            "name": "coder",
            "description": (
                "Writes, saves, and executes Python scripts for causal "
                "inference tasks using DoWhy, EconML, and causal-learn. "
                "Searches library documentation for correct API usage. "
                "Reviews own code with pylint before returning. "
                "Can run scripts via `uv run`."
            ),
            "system_prompt": _load_prompt("coder"),
            "skills": CODER_SKILLS,
            "tools": build_search_tools(config),
            "middleware": [
                ShellToolMiddleware(
                    workspace_root=str(config.data_path) if config.data_path else ".",
                    startup_commands=[
                        f"export PATH=\"{Path(shutil.which('uv')).parent}:$PATH\"",
                        f"export UV_PROJECT=\"{_VENV_BIN.parent.parent}\"",
                        "pip() { echo 'pip is disabled — all required packages are pre-installed.'; return 1; }",
                        "export -f pip",
                        "export PIP_NO_INSTALL=1",
                    ],
                    execution_policy=HostExecutionPolicy(
                        command_timeout=1800.0,
                    ),
                )
            ],
        },
    ]
