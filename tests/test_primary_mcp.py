import asyncio
from fastmcp import Client
import pytest
from dqa.mcp.primary import server as primary_mcp
from tests.mcp_test_mixin import MCPTestMixin


class TestPrimaryMCP(MCPTestMixin):
    # No need to test each MCP exhaustively here, they have their own tests.
    # Just test that the composition works and the tools are available.

    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def mcp_server(cls):
        """
        Fixture to register features in an MCP server.
        """
        server = primary_mcp()
        return server

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

    def test_composition(self, mcp_client):
        """
        Test that the primary MCP server has composed the necessary MCPs correctly.
        """
        tools = asyncio.run(self.list_tools(mcp_client))
        assert len(tools) == 7
