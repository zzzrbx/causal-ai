"""RAG search tools over per-library ChromaDB collections."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, StructuredTool
from langchain_openai import OpenAIEmbeddings

from causal_ai.config import Config


def _make_search_tool(collection_name: str, pkg_label: str, chroma: Chroma) -> BaseTool:
    """Factory that builds a search tool bound to a specific Chroma collection."""

    def _search(query: str) -> str:
        results = chroma.similarity_search(query, k=5)
        if not results:
            return "No results found."

        parts = []
        for i, doc in enumerate(results, start=1):
            module_path = doc.metadata.get("module_path", "unknown")
            file_path = doc.metadata.get("file_path", "unknown")
            parts.append(
                f"--- Result {i} ---\n"
                f"module_path: {module_path}\n"
                f"file_path: {file_path}\n\n"
                f"{doc.page_content}"
            )

        return "\n\n".join(parts)

    return StructuredTool.from_function(
        func=_search,
        name=f"search_{collection_name}",
        description=(
            f"Search the {pkg_label} source code index. "
            "Returns the top-5 most relevant code chunks with their module path and file path."
        ),
    )


def build_search_tools(config: Config) -> list[BaseTool]:
    """Build one search tool per library, each backed by its own ChromaDB collection."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    persist_dir = str(config.rag_index_path)

    libraries = [
        ("dowhy", "DoWhy"),
        ("econml", "EconML"),
        ("causallearn", "causal-learn"),
    ]

    tools = []
    for collection_name, pkg_label in libraries:
        chroma = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        tools.append(_make_search_tool(collection_name, pkg_label, chroma))

    return tools
