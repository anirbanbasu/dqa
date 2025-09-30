import asyncio
from datetime import date, datetime, timezone
from fastmcp import Client, FastMCP
from fastmcp.tools.tool import TextContent
import pytest
from dqa.mcp.datetime import DateTimeMCP

from tests.mcp_test_mixin import MCPTestMixin


class TestDateTimeMCP(MCPTestMixin):
    @pytest.fixture(scope="class", autouse=True)
    @classmethod
    def mcp_server(cls):
        """
        Fixture to register features in an MCP server.
        """
        server = FastMCP()
        mcp_obj = DateTimeMCP()
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

    def test_now(self, mcp_client):
        result = asyncio.run(self.call_tool("now", mcp_client))
        for content in result.content:
            assert isinstance(content, TextContent)
            # Validate ISO format
            parsed_datetime = datetime.fromisoformat(content.text)
            assert isinstance(parsed_datetime, datetime)
            now = datetime.now()
            # Check that the parsed datetime is within a reasonable range of now
            assert (
                abs((parsed_datetime - now).total_seconds()) < 5
            )  # 5 seconds could not have elapsed

    def test_today(self, mcp_client):
        result = asyncio.run(self.call_tool("today", mcp_client))
        for content in result.content:
            assert isinstance(content, TextContent)
            # Validate ISO format
            parsed_date = datetime.fromisoformat(content.text).date()
            assert isinstance(parsed_date, date)
            today = datetime.now().date()
            assert parsed_date == today

    def test_utc_now(self, mcp_client):
        result = asyncio.run(self.call_tool("utc_now", mcp_client))
        for content in result.content:
            assert isinstance(content, TextContent)
            # Validate ISO format
            parsed_datetime = datetime.fromisoformat(content.text)
            assert isinstance(parsed_datetime, datetime)
            utc_now = datetime.now(timezone.utc)
            # Check that the parsed datetime is within a reasonable range of utc_now
            assert (
                abs((parsed_datetime - utc_now).total_seconds()) < 5
            )  # 5 seconds could not have elapsed

    def test_day_of_week(self, mcp_client):
        test_date = "2023-10-05"  # This is a Thursday
        result = asyncio.run(
            self.call_tool("day_of_week", mcp_client, date_str=test_date)
        )
        for content in result.content:
            assert isinstance(content, TextContent)
            assert content.text == "Thursday"
