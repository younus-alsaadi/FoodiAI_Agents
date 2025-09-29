from __future__ import annotations
from typing import Optional, Dict, Any

from langchain_core.tools import StructuredTool
from langchain_tavily import TavilySearch


class TavilySearchTool:
    """Thin wrapper around TavilySearch with a StructuredTool export."""

    def __init__(
        self,
        name: str = "tavily_search_results_json",
        description: Optional[str] = None,
        max_results: int = 3,
        topic: str = "general",
        include_answer: bool = True,
        include_raw_content: bool = False,
    ):
        self.name = name
        self.description = (
            description
            or f'{name}(query="...") → JSON results from Tavily. topic="{topic}", max_results={max_results}.'
        )
        self._search = TavilySearch(
            max_results=max_results,
            topic=topic,
            include_answer=include_answer,
            include_raw_content=include_raw_content,
        )

    # Direct call API
    def run(self, query: str) -> Dict[str, Any]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        # Recent versions accept {"query": "..."}; this is forward-compatible.
        return self._search.invoke({"query": query})

    # LangChain tool export
    def tavily_as_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=self.run,
            name=self.name,
            description=self.description,
        )

