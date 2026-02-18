# causal-ai

[![Tests](https://github.com/zzzrbx/causal-ai/actions/workflows/tests.yml/badge.svg)](https://github.com/zzzrbx/causal-ai/actions/workflows/tests.yml)
[![Pylint](https://github.com/zzzrbx/causal-ai/actions/workflows/pylint.yml/badge.svg)](https://github.com/zzzrbx/causal-ai/actions/workflows/pylint.yml)

![causal-ai banner](.github/banner.png)

## What it does

- **Answers causal questions** — identification strategies, adjustment sets, assumptions, interpretation of results
- **Writes and runs code** — DoWhy, EconML, and causal-learn scripts generated, linted, and executed on your machine
- **Searches library source** — a local RAG index over DoWhy, EconML, and causal-learn lets the coder agent look up correct API usage before writing code

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
model=anthropic:claude-opus-4-6
```

Available keys:

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `openai:gpt-5.2` | LangChain model string |
| `data_path` | — | Optional path to a data directory, exposed to the agent as `/userdata/` |
| `rag_index_path` | `/tmp/causal-ai-rag` | Where the ChromaDB index is stored |

## Usage

```bash
uv run causal-ai
```

Type your causal inference question at the `>` prompt. Type `exit` or `quit` to stop.
