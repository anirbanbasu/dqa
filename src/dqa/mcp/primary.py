import logging
import signal
import sys

from fastmcp import FastMCP
from dqa import env
from marshmallow.validate import OneOf

from dqa.mcp.ollama import app as ollama_mcp
from dqa.mcp.datetime import app as datetime_mcp
from dqa.mcp.arithmetic import app as basic_arithmetic_mcp
from frankfurtermcp.server import app as frankfurter_mcp

logger = logging.getLogger(__name__)


def server():
    server = FastMCP(
        name="dqa-mcp",
        instructions="An MCP server for DQA.",
        on_duplicate_prompts="error",
        on_duplicate_resources="error",
        on_duplicate_tools="error",
    )

    logger.info("Mounting Ollama web search and fetch MCP...")
    server.mount(ollama_mcp(), prefix="ollama")
    logger.info("Mounting Frankfurter currency conversion MCP...")
    server.mount(frankfurter_mcp(), prefix="currency")
    logger.info("Mounting DateTime MCP...")
    server.mount(datetime_mcp(), prefix="datetime")
    logger.info("Mounting Basic Arithmetic MCP...")
    server.mount(basic_arithmetic_mcp(), prefix="arithmetic")

    return server


def main():  # pragma: no cover
    MCP_SERVER_TRANSPORT = "DQA_MCP_SERVER_TRANSPORT"
    DEFAULT__MCP_SERVER_TRANSPORT = "streamable-http"
    ALLOWED__MCP_SERVER_TRANSPORT = ["stdio", "sse", "streamable-http"]

    MCP_SERVER_HOST = "FASTMCP_HOST"
    DEFAULT__MCP_SERVER_HOST = "localhost"

    MCP_SERVER_PORT = "FASTMCP_PORT"
    DEFAULT__MCP_SERVER_PORT = 8000

    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        logger.info("Attempting graceful shutdown")
        # This is absolutely necessary to exit the program
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    transport_type = env.str(
        MCP_SERVER_TRANSPORT,
        default=DEFAULT__MCP_SERVER_TRANSPORT,
        validate=OneOf(ALLOWED__MCP_SERVER_TRANSPORT),
    )

    app = server()

    logger.info(
        f"Starting DQA MCP server with {transport_type} transport, press CTRL+C to exit..."
    )
    if transport_type == "stdio":
        app.run()
    else:
        app.run(
            transport=transport_type,
            host=env.str(
                MCP_SERVER_HOST,
                default=DEFAULT__MCP_SERVER_HOST,
            ),
            port=env.int(
                MCP_SERVER_PORT,
                default=DEFAULT__MCP_SERVER_PORT,
            ),
            uvicorn_config={
                "timeout_graceful_shutdown": 5,  # seconds
            },
        )


if __name__ == "__main__":  # pragma: no cover
    main()
