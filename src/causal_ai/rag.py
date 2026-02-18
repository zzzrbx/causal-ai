"""RAG search tool over the ChromaDB code index."""
from __future__ import annotations

from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings

from causal_ai.config import Config


def build_search_code_tool(config: Config) -> BaseTool:
    """Build a LangChain tool that searches the ChromaDB code index."""
    chroma = Chroma(
        collection_name="code",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small"),
        persist_directory=str(config.rag_index_path),
    )

    @tool
    def search_code(query: str) -> str:
        """Search the causal inference library source code index.

        Returns the top-5 most relevant code chunks with their module path
        and file path.
        """
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

    return search_code
