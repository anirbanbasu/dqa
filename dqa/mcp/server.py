import signal
import sys
import base64

from dotenv import load_dotenv
from fastmcp import FastMCP

from rich import print as print
from dqa.common import EnvironmentVariables, MCPParameters, ic
from dqa.utils import parse_env

from dqa.mcp.arithmetic import app as arithmetic_mcp

app = FastMCP(
    name="dqa-mcp",
    description="A FastMCP server for the DQA project.",
    on_duplicate_prompts="error",
    on_duplicate_resources="error",
    on_duplicate_tools="error",
)


@app.resource(
    uri="data://logo",
    name="logo",
    description="The logo of the DQA project.",
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
    logo_data_base64 = base64.b64encode(logo_data.encode("utf-8")).decode("utf-8")
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

    print("[green]Starting DQA MCP server, press CTRL+C to exit...[/green]")
    app.mount(
        prefix="arithmetic",
        server=arithmetic_mcp,
        tool_separator=MCPParameters.TOOL_SEPARATOR,
        resource_separator=MCPParameters.RESOURCE_SEPARATOR,
        prompt_separator=MCPParameters.PROMPT_SEPARATOR,
    )
    app.run(
        transport=parse_env(
            EnvironmentVariables.DQA__MCP_SERVER_TRANSPORT,
            default_value=EnvironmentVariables.DEFAULT_DQA__MCP_SERVER_TRANSPORT,
            allowed_values=EnvironmentVariables.ALLOWED__DQA_MCP_SERVER_TRANSPORT,
        )
    )


if __name__ == "__main__":
    main()
