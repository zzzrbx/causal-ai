"""Sub-agent definitions for causal-ai."""
from __future__ import annotations

from pathlib import Path

from causal_ai.config import Config
from causal_ai.rag import build_search_code_tool

PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"

# Skill source paths are PARENT directories containing skill subdirectories.
# SkillsMiddleware lists the parent, discovers subdirs, and loads SKILL.md from each.
QA_SKILLS = ["/project/skills/"]
CODER_SKILLS = ["/project/skills/"]


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
            "tools": [build_search_code_tool(config)],
        },
    ]
