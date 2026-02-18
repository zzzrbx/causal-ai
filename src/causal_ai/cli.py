"""Rich-based CLI REPL for causal-ai."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.markdown import Markdown

from causal_ai.agent import create_orchestrator
from causal_ai.config import Config

console = Console()
WELCOME_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "welcome.md"


_THREAD_CONFIG = {"configurable": {"thread_id": "main"}}


def _run_repl(orchestrator) -> None:
    """Main REPL loop: read input, send to orchestrator, display response."""
    session: PromptSession[str] = PromptSession(history=InMemoryHistory())

    while True:
        try:
            user_input = session.prompt("> ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("Goodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("/quit", "/exit", "quit", "exit"):
            console.print("Goodbye!")
            break

        try:
            with console.status("[bold cyan]Thinking...[/bold cyan]"):
                result = orchestrator.invoke(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=_THREAD_CONFIG,
                )
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
            continue

        content = result["messages"][-1].content
        if isinstance(content, list):
            response = "\n".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        else:
            response = str(content)
        console.print()
        console.print(Markdown(response))
        console.print()


def main() -> None:
    """Entry point for the causal-ai CLI."""
    load_dotenv()
    config = Config.load()

    console.print(Markdown(WELCOME_PATH.read_text(encoding="utf-8")))
    console.print()

    with console.status("[bold cyan]Loading agent...[/bold cyan]"):
        orchestrator = create_orchestrator(config)

    _run_repl(orchestrator)
