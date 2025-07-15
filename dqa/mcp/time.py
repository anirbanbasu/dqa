from typing import Annotated
from fastmcp import FastMCP
from datetime import datetime, timezone

app = FastMCP(
    instructions="A collection of time awareness tools.",
    on_duplicate_prompts="error",
    on_duplicate_resources="error",
    on_duplicate_tools="error",
)


@app.tool(
    tags=["time", "awareness", "now"],
)
def what_is_now() -> str:
    """
    Outputs the ISO format date and time for now.
    This is the local time of the server.
    """
    return datetime.now().isoformat()


@app.tool(
    tags=["time", "awareness", "now"],
)
def what_is_today() -> str:
    """
    Outputs the ISO format date for today.
    This is the local date of the server.
    """
    return datetime.today().isoformat()


@app.tool(
    tags=["time", "awareness", "now", "utc"],
)
def what_is_utc_now() -> str:
    """
    Outputs the ISO format date and time for now in UTC.
    """
    return datetime.now(timezone.utc).isoformat()


@app.tool(
    tags=["time", "awareness", "now", "utc"],
)
def what_day_of_the_week(date: Annotated[str, "ISO format date"]) -> str:
    """
    Returns the day of the week for a given date in ISO format.
    The date should be in the format YYYY-MM-DD.
    """
    dt = datetime.fromisoformat(date)
    return dt.strftime("%A")
