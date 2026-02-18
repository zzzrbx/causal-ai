"""Build the ChromaDB RAG index from dowhy, econml, and causallearn sources.

Run once after clone:
    uv run scripts/install_rag.py
    uv run scripts/install_rag.py --force   # wipe and re-index
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import shutil
import warnings
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from causal_ai.config import Config

load_dotenv()

_ESC_PATTERN = ".*invalid escape sequence.*"
warnings.filterwarnings("ignore", category=SyntaxWarning, message=_ESC_PATTERN)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=_ESC_PATTERN)


def _find_package_root(package_name: str) -> Path | None:
    spec = importlib.util.find_spec(package_name)
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).parent


def _iter_py_files(root: Path):
    yield from sorted(root.rglob("*.py"))


def _module_path_for(file_path: Path, package_root: Path, package_name: str) -> str:
    try:
        rel = file_path.relative_to(package_root.parent)
    except ValueError:
        rel = file_path
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) if parts else package_name


def _extract_chunks(file_path: Path, package_root: Path, package_name: str) -> list[dict]:
    try:
        source = file_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    base_module = _module_path_for(file_path, package_root, package_name)
    lines = source.splitlines(keepends=True)
    chunks = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunk_text = "".join(lines[node.lineno - 1 : node.end_lineno])
            module_path = f"{base_module}.{node.name}"
            chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "module_path": module_path,
                        "file_path": str(file_path),
                    },
                }
            )
        elif isinstance(node, ast.ClassDef):
            class_lines = "".join(lines[node.lineno - 1 : node.end_lineno])
            class_module = f"{base_module}.{node.name}"
            chunks.append(
                {
                    "text": class_lines,
                    "metadata": {
                        "module_path": class_module,
                        "file_path": str(file_path),
                    },
                }
            )
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_text = "".join(lines[child.lineno - 1 : child.end_lineno])
                    method_path = f"{class_module}.{child.name}"
                    chunks.append(
                        {
                            "text": method_text,
                            "metadata": {
                                "module_path": method_path,
                                "file_path": str(file_path),
                            },
                        }
                    )

    return chunks


def _index_package(package_name: str, chroma_collection, batch_size: int = 100) -> int:
    """Index all functions and classes from a package into the ChromaDB collection."""
    root = _find_package_root(package_name)
    if root is None:
        print(f"  WARNING: could not locate package {package_name!r} — skipping")
        return 0

    print(f"  Indexing {package_name} from {root}")
    all_chunks = []
    for py_file in _iter_py_files(root):
        all_chunks.extend(_extract_chunks(py_file, root, package_name))

    total = 0
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        metadatas = [c["metadata"] for c in batch]
        ids = [f"{c['metadata']['module_path']}::{i + j}" for j, c in enumerate(batch)]
        chroma_collection.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        total += len(batch)

    print(f"  Indexed {total} chunks from {package_name}")
    return total


def main() -> None:
    """Parse args, build embedding function, and index all packages."""
    parser = argparse.ArgumentParser(
        description="Build ChromaDB RAG index from causal inference library sources."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Wipe existing index and re-index from scratch.",
    )
    args = parser.parse_args()

    config = Config.load()
    index_path = config.rag_index_path

    if args.force and index_path.exists():
        print(f"--force: deleting existing index at {index_path}")
        shutil.rmtree(index_path)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    packages = ["dowhy", "econml", "causallearn"]
    grand_total = 0

    for pkg in packages:
        chroma = Chroma(
            collection_name=pkg,
            embedding_function=embeddings,
            persist_directory=str(index_path),
        )

        if not args.force:
            existing = chroma.get(limit=1)
            if existing and existing.get("ids"):
                print(f"[{pkg}] already populated — skipping. Use --force to re-index.")
                continue

        print(f"\n[{pkg}]")
        grand_total += _index_package(pkg, chroma)

    print(f"\nDone. Total chunks indexed: {grand_total}")
    print(f"Index stored at: {index_path}")


if __name__ == "__main__":
    main()
