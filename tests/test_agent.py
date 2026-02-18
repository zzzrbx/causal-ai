"""Integration test — orchestrator sends a prompt and receives a response.

Requires API keys to be set in the environment (e.g. OPENAI_API_KEY).
"""
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from causal_ai.agent import create_orchestrator
from causal_ai.config import Config

load_dotenv()


def test_orchestrator_returns_response():
    config = Config.load()
    orchestrator = create_orchestrator(config)

    result = orchestrator.invoke(
        {"messages": [HumanMessage(content="Say hello.")]},
        config={"configurable": {"thread_id": "test-thread"}},
    )

    assert result["messages"], "No messages returned"
    content = result["messages"][-1].content
    if isinstance(content, list):
        content = " ".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    assert content.strip(), "Response content is empty"
