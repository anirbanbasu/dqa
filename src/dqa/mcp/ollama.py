# Based on https://github.com/ollama/ollama-python/blob/main/examples/web-search-mcp.py

from typing import Any, ClassVar, Dict
from dqa.mcp.mixin import MCPMixin

from importlib.metadata import version
from fastmcp import FastMCP
from ollama import Client


class OllamaMCP(MCPMixin):
    """
    A mixin-based MCP server for Ollama web search and fetch.

    Environment:
     - OLLAMA_API_KEY (required). Get it from https://ollama.com/settings/keys.
    """

    tools = [
        {
            "fn": "web_search",
            "name": "web_search",
            "description": "Search the web for relevant information.",
            "tags": ["ollama", "search", "web"],
            "annotations": {
                "readOnlyHint": True,
            },
        },
        {
            "fn": "web_fetch",
            "name": "web_fetch",
            "description": "Fetch the content of a given URL.",
            "tags": ["ollama", "fetch", "web"],
            "annotations": {
                "readOnlyHint": True,
            },
        },
    ]

    client: ClassVar[Client] = Client()

    def web_search(self, query: str, max_results: int = 3) -> Dict[str, Any]:
        response = self.client.web_search(query=query, max_results=max_results)
        return response.model_dump()

    def web_fetch(self, url: str) -> Dict[str, Any]:
        response = self.client.web_fetch(url=url)
        return response.model_dump()


def app() -> FastMCP:  # pragma: no cover
    app = FastMCP(
        name="dqa-ollama-search-fetch",
        version=version("ollama"),
        instructions="A MCP wrapper for Ollama search and fetch.",
        on_duplicate_prompts="error",
        on_duplicate_resources="error",
        on_duplicate_tools="error",
    )
    mcp_obj = OllamaMCP()
    app_with_features = mcp_obj.register_features(app)
    return app_with_features
