# causal-ai

[![Tests](https://github.com/zzzrbx/causal-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/zzzrbx/causal-ai/actions/workflows/tests.yml)
[![Pylint](https://github.com/zzzrbx/causal-ai/actions/workflows/pylint.yml/badge.svg)](https://github.com/zzzrbx/causal-ai/actions/workflows/pylint.yml)

![causal-ai banner](.github/banner.png)

> **causal-ai is experimental and a learning tool.** It is designed for exploring causal inference concepts and running small, self-contained scripts — not for production pipelines or large-scale data processing. Expect rough edges.

## What it does

- **Answers causal questions** — identification strategies, adjustment sets, assumptions, interpretation of results
- **Writes and runs code** — DoWhy, EconML, and causal-learn scripts generated, linted, and executed on your machine

## Security

The coder agent can write and execute arbitrary shell commands on your machine. It is recommended to run causal-ai inside a **VM or Docker container**. Sandboxing is the user's responsibility.

## Prerequisites

- Python 3.12+
- [ripgrep](https://github.com/BurntSushi/ripgrep) — required for agent file search (`brew install ripgrep`)

## Installation

```bash
uv sync
```

## Build the RAG index

The coder agent uses a local ChromaDB index over DoWhy, EconML, and causal-learn source code. Build it once after install:

```bash
uv run scripts/install_rag.py
```

To wipe and rebuild from scratch:

```bash
uv run scripts/install_rag.py --force
```

## Configuration

Copy or create a `.config` file in the project root:

```
model=openai:gpt-5-mini
```

Available keys:

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `openai:gpt-5.2` | LangChain model string |
| `data_path` | — | Optional path to a data directory, exposed to the agent as `/userdata/` |
| `rag_index_path` | `/tmp/causal-ai-rag` | Where the ChromaDB index is stored |

## Supported providers

The `model` key accepts any [LangChain `init_chat_model`](https://python.langchain.com/docs/how_to/chat_models_universal_init/) string. The following providers are installed:

| Provider | Prefix | Env var | Example |
|----------|--------|---------|---------|
| OpenAI | `openai:` | `OPENAI_API_KEY` | `openai:gpt-5.2` |
| Anthropic | `anthropic:` | `ANTHROPIC_API_KEY` | `anthropic:claude-opus-4-6` |

## Usage

```bash
uv run causal-ai
```

Type your causal inference question at the `>` prompt. Type `exit` or `quit` to stop.
