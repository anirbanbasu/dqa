import asyncio
from fastmcp import Client, FastMCP
from fastmcp.tools.tool import TextContent
import pytest
from dqa.mcp.ollama import OllamaMCP
from ollama import WebSearchResponse, WebFetchResponse
from ollama._types import WebSearchResult

from tests.mcp_test_mixin import MCPTestMixin


class TestOllamaMCP(MCPTestMixin):
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def mcp_server(cls):
        """
        Fixture to register features in an MCP server.
        """
        server = FastMCP()
        mcp_obj = OllamaMCP()
        server_with_features = mcp_obj.register_features(server)
        return server_with_features

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def mcp_client(cls, mcp_server):
        """
        Fixture to create a client for the MCP server.
        """
        mcp_client = Client(
            transport=mcp_server,
            timeout=60,
        )
        return mcp_client

    def test_web_search(self, mcp_client):
        query = "What is the official website of Ollama?"
        max_results = 3
        result = asyncio.run(
            self.call_tool(
                "web_search", mcp_client, query=query, max_results=max_results
            )
        )
        for content in result.content:
            assert isinstance(content, TextContent)
            validated_results = WebSearchResponse.model_validate_json(content.text)
            assert all(
                isinstance(item, WebSearchResult) for item in validated_results.results
            )
            assert any(
                item.url == "https://ollama.com/" for item in validated_results.results
            )

    def test_web_fetch(self, mcp_client):
        url = "https://ollama.com/"
        result = asyncio.run(self.call_tool("web_fetch", mcp_client, url=url))
        for content in result.content:
            assert isinstance(content, TextContent)
            validated_result = WebFetchResponse.model_validate_json(content.text)
            assert validated_result is not None
