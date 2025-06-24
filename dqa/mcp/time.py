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
    name="what_is_now",
    description="Outputs the ISO format date and time for now.",
    tags=["time", "awareness", "now"],
)
def what_is_now() -> str:
    return datetime.now().isoformat()


@app.tool(
    name="what_is_today",
    description="Outputs the ISO format date for today.",
    tags=["time", "awareness", "now"],
)
def what_is_today() -> str:
    return datetime.today().isoformat()


@app.tool(
    name="what_is_utc_now",
    description="Outputs the ISO format date and time for now in UTC.",
    tags=["time", "awareness", "now", "utc"],
)
def what_is_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@app.tool(
    name="what_day_of_the_week",
    description="Outputs the day of the week for a given date in ISO format.",
    tags=["time", "awareness", "now", "utc"],
)
def what_day_of_the_week(date: Annotated[str, "ISO format date"]) -> str:
    dt = datetime.fromisoformat(date)
    return dt.strftime("%A")
