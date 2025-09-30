import asyncio
from fastmcp import Client, FastMCP
from fastmcp.tools.tool import TextContent
import pytest
from dqa.mcp.ollama import OllamaMCP
from ollama import WebSearchResponse, WebFetchResponse
from ollama._types import WebSearchResult


class TestOllamaMCP:
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

    async def call_tool(self, tool_name: str, mcp_client: Client, **kwargs):
        """
        Helper method to call a tool on the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.call_tool(tool_name, arguments=kwargs)
            await mcp_client.close()
        return result

    async def read_resource(self, resource_name: str, mcp_client: Client):
        """
        Helper method to load a resource from the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.read_resource(resource_name)
            await mcp_client.close()
        return result

    async def get_prompt(self, prompt_name: str, mcp_client: Client, **kwargs):
        """
        Helper method to get a prompt from the MCP server.
        """
        async with mcp_client:
            result = await mcp_client.get_prompt(prompt_name, arguments=kwargs)
            await mcp_client.close()
        return result

    def test_web_search(self, mcp_client):
        query = "What is Ollama?"
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
            # Will this always succeed?
            assert url in validated_result.links
