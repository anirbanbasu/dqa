# Based on https://github.com/ollama/ollama-python/blob/main/examples/web-search-mcp.py

from datetime import datetime, timezone

from dqa.mcp.mixin import MCPMixin

from importlib.metadata import version
from fastmcp import FastMCP


class DateTimeMCP(MCPMixin):
    """
    A mixin-based MCP server for date time awareness.
    """

    tools = [
        {
            "fn": "now",
            "name": "now",
            "description": "Returns the current date and time as a ISO format string.",
            "tags": ["datetime", "current time", "now"],
            "annotations": {
                "readOnlyHint": True,
            },
        },
        {
            "fn": "today",
            "name": "today",
            "description": "Returns the current date as a ISO format string.",
            "tags": ["datetime", "current date", "today"],
            "annotations": {
                "readOnlyHint": True,
            },
        },
        {
            "fn": "utc_now",
            "name": "utc_now",
            "description": "Returns the current UTC date and time as a ISO format string.",
            "tags": ["datetime", "current time", "utc now"],
            "annotations": {
                "readOnlyHint": True,
            },
        },
        {
            "fn": "day_of_week",
            "name": "day_of_week",
            "description": "Returns the day of the week for a given date string in ISO format.",
            "tags": ["datetime", "day of week"],
            "annotations": {
                "readOnlyHint": True,
            },
        },
    ]

    def now(self) -> str:
        """
        Returns the current date and time as a ISO format string.
        """

        return datetime.now().isoformat()

    def today(self) -> str:
        """
        Returns the current date as a ISO format string.
        """

        return datetime.now().date().isoformat()

    def utc_now(self) -> str:
        """
        Returns the current UTC date and time as a ISO format string.
        """

        return datetime.now(timezone.utc).isoformat()

    def day_of_week(self, date_str: str) -> str:
        """
        Returns the day of the week for a given date string in ISO format.
        """

        parsed_date = datetime.fromisoformat(date_str)
        return parsed_date.strftime("%A")


def app() -> FastMCP:  # pragma: no cover
    app = FastMCP(
        name="dqa-datetime",
        version=version("dqa"),
        instructions="A MCP for date time awareness.",
        on_duplicate_prompts="error",
        on_duplicate_resources="error",
        on_duplicate_tools="error",
    )
    mcp_obj = DateTimeMCP()
    app_with_features = mcp_obj.register_features(app)
    return app_with_features
