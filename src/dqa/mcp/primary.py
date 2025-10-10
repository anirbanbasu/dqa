import logging
import signal
import sys

from fastmcp import FastMCP, Client
from dqa import ParsedEnvVars

from dqa.mcp.ollama import app as ollama_mcp
from dqa.mcp.datetime import app as datetime_mcp
from dqa.mcp.arithmetic import app as basic_arithmetic_mcp
from frankfurtermcp.server import app as frankfurter_mcp
from yfmcp.server import mcp as yfmcp_mcp

logger = logging.getLogger(__name__)


def server():
    server = FastMCP(
        name="dqa-mcp",
        instructions="An MCP server for DQA.",
        on_duplicate_prompts="error",
        on_duplicate_resources="error",
        on_duplicate_tools="error",
    )

    # See: https://gofastmcp.com/servers/composition
    # Live linking
    server.mount(frankfurter_mcp(), prefix="currency")
    logger.info("Mounted Frankfurter currency conversion MCP.")
    server.mount(datetime_mcp(), prefix="datetime")
    logger.info("Mounted DateTime MCP.")
    server.mount(basic_arithmetic_mcp(), prefix="arithmetic")
    logger.info("Mounted Basic Arithmetic MCP.")
    server.mount(yfmcp_mcp, prefix="yfmcp", as_proxy=True)
    logger.info("Mounted YFMCP.")

    if ParsedEnvVars().API_KEY_ALPHAVANTAGE:
        # See: https://github.com/alphavantage/alpha_vantage_mcp
        alphavantage_remote_url = f"https://mcp.alphavantage.co/mcp?apikey={ParsedEnvVars().API_KEY_ALPHAVANTAGE}"
        alphavantage_remote_proxy = FastMCP.as_proxy(Client(alphavantage_remote_url))
        server.mount(alphavantage_remote_proxy, prefix="alphavantage")
        logger.info("Mounted AlphaVantage remote MCP.")
        logger.warning(
            "AlphaVantage MCP is a rate-limited remote service, expect higher latency."
        )
    else:
        logger.warning(
            "No AlphaVantage API key found in environment variable 'API_KEY_ALPHAVANTAGE'. Skipping mounting AlphaVantage remote MCP."
        )

    if ParsedEnvVars().API_KEY_TAVILY:
        # See: https://github.com/alphavantage/alpha_vantage_mcp
        tavily_remote_url = (
            f"https://mcp.tavily.com/mcp/?tavilyApiKey={ParsedEnvVars().API_KEY_TAVILY}"
        )
        tavily_remote_proxy = FastMCP.as_proxy(Client(tavily_remote_url))
        # Prefix is not necessary because Tavily tools are prefixed with "tavily_" already.
        server.mount(tavily_remote_proxy)
        logger.info("Mounted Tavily remote MCP.")
        logger.warning(
            "Tavily MCP is a rate-limited remote service, expect higher latency."
        )
    else:
        logger.warning(
            "No AlphaVantage API key found in environment variable 'TAVILY_API_KEY'. Skipping mounting Tavily remote MCP."
        )

    if ParsedEnvVars().API_KEY_OLLAMA:
        server.mount(ollama_mcp(), prefix="ollama")
        logger.info("Mounted Ollama MCP.")
    else:
        logger.warning(
            "No Ollama API key found in environment variable 'API_KEY_OLLAMA'. Skipping mounting Ollama MCP."
        )

    return server


def main():  # pragma: no cover
    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        logger.info("Attempting graceful shutdown")
        # This is absolutely necessary to exit the program
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    transport_type = ParsedEnvVars().DQA_MCP_SERVER_TRANSPORT

    app = server()

    logger.info(
        f"Starting DQA MCP server with {transport_type} transport, press CTRL+C to exit..."
    )
    if transport_type == "stdio":
        app.run()
    else:
        app.run(
            transport=transport_type,
            host=ParsedEnvVars().MCP_SERVER_HOST,
            port=ParsedEnvVars().MCP_SERVER_PORT,
            uvicorn_config={
                "timeout_graceful_shutdown": 5,  # seconds
            },
        )


if __name__ == "__main__":  # pragma: no cover
    main()
