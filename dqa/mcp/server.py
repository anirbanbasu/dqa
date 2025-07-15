import signal
import sys
import base64

from dotenv import load_dotenv
from fastmcp import FastMCP

from rich import print as print
from dqa.common import EnvironmentVariables, ic
from dqa.utils import parse_env

from dqa.mcp.arithmetic import app as arithmetic_mcp
from dqa.mcp.time import app as time_app

app = FastMCP(
    name="dqa-mcp",
    instructions="A FastMCP server for the DQA project.",
    on_duplicate_prompts="error",
    on_duplicate_resources="error",
    on_duplicate_tools="error",
)


@app.resource(
    uri="data://logo",
    mime_type="application/json",
    tags=["logo", "dqa", "metadata"],
)
def logo():
    """
    Returns the logo of the DQA project. [Rather pointless, but useful for testing purposes.]
    """
    with open("assets/logo.svg", "r") as f:
        logo_data = f.read()
        f.close()
    # Encode the logo data in base64
    logo_data_base64 = base64.b64encode(logo_data.encode()).decode()
    return {
        "name": "DQA logo",
        "description": "The logo of the DQA project.",
        "mime_type": "image/svg+xml",
        "encoding": "base64",
        "data": logo_data_base64,
    }


def main():
    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        print("[green]Attempting graceful shutdown, please wait...[/green]")
        # This is absolutely necessary to exit the program
        sys.exit(0)

    ic(load_dotenv())
    # TODO: Should we also catch SIGTERM, SIGKILL, etc.? What about Windows?
    signal.signal(signal.SIGINT, sigint_handler)

    app.mount(
        prefix="arithmetic",
        server=arithmetic_mcp,
    )

    app.mount(
        prefix="time",
        server=time_app,
    )

    transport_type = parse_env(
        EnvironmentVariables.MCP_SERVER_TRANSPORT,
        default_value=EnvironmentVariables.DEFAULT__MCP_SERVER_TRANSPORT,
        allowed_values=EnvironmentVariables.ALLOWED__MCP_SERVER_TRANSPORT,
    )

    print(
        f"[green]Starting DQA MCP server with {transport_type} transport, press CTRL+C to exit...[/green]"
    )

    if transport_type == "stdio":
        app.run()
    else:
        app.run(
            transport=transport_type,
            host=parse_env(
                EnvironmentVariables.MCP_SERVER_HOST,
                default_value=EnvironmentVariables.DEFAULT__MCP_SERVER_HOST,
            ),
            port=parse_env(
                EnvironmentVariables.MCP_SERVER_PORT,
                default_value=EnvironmentVariables.DEFAULT__MCP_SERVER_PORT,
                type_cast=int,
            ),
            uvicorn_config={
                "timeout_graceful_shutdown": 5,  # seconds
            },
        )


if __name__ == "__main__":
    main()
