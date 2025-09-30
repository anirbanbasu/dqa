import asyncio
from fastmcp import Client, FastMCP
from fastmcp.tools.tool import TextContent
import pytest
from dqa.mcp.arithmetic import BasicArithmeticMCP

from tests.mcp_test_mixin import MCPTestMixin


class TestBasicArithmeticMCP(MCPTestMixin):
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def mcp_server(cls):
        """
        Fixture to register features in an MCP server.
        """
        server = FastMCP()
        mcp_obj = BasicArithmeticMCP()
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

    def test_add(self, mcp_client):
        result = asyncio.run(self.call_tool("add", mcp_client, augend=5, summand=3))
        for content in result.content:
            assert isinstance(content, TextContent)
            assert content.text == "8.0"

    def test_subtract(self, mcp_client):
        result = asyncio.run(
            self.call_tool("subtract", mcp_client, minuend=10, subtrahend=4)
        )
        for content in result.content:
            assert isinstance(content, TextContent)
            assert content.text == "6.0"

    def test_multiply(self, mcp_client):
        result = asyncio.run(
            self.call_tool("multiply", mcp_client, multiplicand=7, multiplier=6)
        )
        for content in result.content:
            assert isinstance(content, TextContent)
            assert content.text == "42.0"

    def test_divide(self, mcp_client):
        result = asyncio.run(
            self.call_tool("divide", mcp_client, dividend=20, divisor=4)
        )
        for content in result.content:
            assert isinstance(content, TextContent)
            assert content.text == "5.0"

    def test_modulus(self, mcp_client):
        result = asyncio.run(
            self.call_tool("modulus", mcp_client, dividend=10, divisor=3)
        )
        for content in result.content:
            assert isinstance(content, TextContent)
            assert content.text == "1.0"

    def test_power(self, mcp_client):
        result = asyncio.run(self.call_tool("power", mcp_client, base=2, exponent=3))
        for content in result.content:
            assert isinstance(content, TextContent)
            assert content.text == "8.0"
