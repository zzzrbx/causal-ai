"""Pytest configuration — add scripts/ to sys.path for install_rag imports."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
