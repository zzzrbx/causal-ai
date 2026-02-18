"""Unit tests for the AST chunker in scripts/install_rag.py.

No ChromaDB, no API keys, no network — pure AST parsing logic.
"""
import textwrap
from pathlib import Path

import pytest

from install_rag import _extract_chunks, _module_path_for


@pytest.fixture()
def pkg_root(tmp_path: Path) -> Path:
    """Create a fake package directory: tmp_path/mypkg/ mirroring site-packages layout."""
    return tmp_path / "mypkg"


@pytest.fixture()
def sample_py(pkg_root: Path) -> Path:
    """Write a small Python file inside the fake package."""
    code = textwrap.dedent("""\
        MY_CONSTANT = 42

        def top_level_function(x, y):
            '''Add two numbers.'''
            return x + y

        class MyEstimator:
            '''A simple estimator.'''

            def fit(self, data):
                pass

            def predict(self, data):
                return data
    """)
    pkg_root.mkdir()
    f = pkg_root / "mymodule.py"
    f.write_text(code)
    return f


def test_functions_are_chunked(sample_py: Path, pkg_root: Path):
    chunks = _extract_chunks(sample_py, pkg_root, "mypkg")
    module_paths = [c["metadata"]["module_path"] for c in chunks]
    assert "mypkg.mymodule.top_level_function" in module_paths


def test_class_is_chunked(sample_py: Path, pkg_root: Path):
    chunks = _extract_chunks(sample_py, pkg_root, "mypkg")
    module_paths = [c["metadata"]["module_path"] for c in chunks]
    assert "mypkg.mymodule.MyEstimator" in module_paths


def test_methods_are_chunked(sample_py: Path, pkg_root: Path):
    chunks = _extract_chunks(sample_py, pkg_root, "mypkg")
    module_paths = [c["metadata"]["module_path"] for c in chunks]
    assert "mypkg.mymodule.MyEstimator.fit" in module_paths
    assert "mypkg.mymodule.MyEstimator.predict" in module_paths


def test_constants_are_skipped(sample_py: Path, pkg_root: Path):
    chunks = _extract_chunks(sample_py, pkg_root, "mypkg")
    for chunk in chunks:
        assert "MY_CONSTANT" not in chunk["metadata"]["module_path"]


def test_file_path_metadata(sample_py: Path, pkg_root: Path):
    chunks = _extract_chunks(sample_py, pkg_root, "mypkg")
    for chunk in chunks:
        assert chunk["metadata"]["file_path"] == str(sample_py)


def test_chunk_text_contains_source(sample_py: Path, pkg_root: Path):
    chunks = _extract_chunks(sample_py, pkg_root, "mypkg")
    fn_chunk = next(
        c for c in chunks if c["metadata"]["module_path"] == "mypkg.mymodule.top_level_function"
    )
    assert "def top_level_function" in fn_chunk["text"]
    assert "Add two numbers" in fn_chunk["text"]
