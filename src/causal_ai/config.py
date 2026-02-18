"""Parse the .config file for causal-ai settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_MODEL = "anthropic:claude-opus-4-6"


@dataclass
class Config:
    """Application configuration loaded from .config file."""
    model: str = DEFAULT_MODEL
    data_path: Path | None = None
    rag_index_path: Path = Path("/tmp/causal-ai-rag")

    @classmethod
    def load(cls, path: Path | str = ".config") -> Config:
        """Load config from a key=value file.

        Missing file returns defaults. Unknown keys are ignored.
        """
        path = Path(path)
        cfg = cls()
        if not path.exists():
            return cfg

        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if key == "model":
                cfg.model = value
            elif key == "data_path":
                cfg.data_path = Path(value).expanduser()
            elif key == "rag_index_path":
                cfg.rag_index_path = Path(value).expanduser()

        return cfg
