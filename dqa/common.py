try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from importlib import metadata

package_metadata = metadata.metadata("dqa")


class EnvironmentVariables:
    """
    List of environment variables used in the DQA project.
    """

    MCP_SERVER_TRANSPORT = "DQA_MCP_SERVER_TRANSPORT"
    DEFAULT__MCP_SERVER_TRANSPORT = "streamable-http"
    ALLOWED__MCP_SERVER_TRANSPORT = ["stdio", "sse", "streamable-http"]

    DQA_LLM_CONFIG = "DQA_LLM_CONFIG"
    DEFAULT__DQA_LLM_CONFIG = "config/llms.json"

    DQA_USE_MCP = "DQA_USE_MCP"
    DEFAULT_DQA_USE_MCP = "True"

    DQA_LOGGING_LEVEL = "DQA_LOGGING_LEVEL"
    DEFAULT_DQA_LOGGING_LEVEL = "INFO"

    DQA_MCP_CLIENT_CONFIG = "DQA_MCP_CLIENT_CONFIG"
    DEFAULT_DQA_MCP_CLIENT_CONFIG = "config/mcp-client.json"

    MCP_SERVER_HOST = "FASTMCP_HOST"
    DEFAULT__MCP_SERVER_HOST = "localhost"

    MCP_SERVER_PORT = "FASTMCP_PORT"
    DEFAULT__MCP_SERVER_PORT = 8000

    DQA_ACP_HOST = "DQA_ACP_HOST"
    DEFAULT__DQA_ACP_HOST = "127.0.0.1"

    DQA_ACP_PORT = "DQA_ACP_PORT"
    DEFAULT__DQA_ACP_PORT = 8192

    DQA_ACP_CLIENT_ACCESS_URL = "DQA_ACP_CLIENT_ACCESS_URL"
    DEFAULT__DQA_ACP_CLIENT_ACCESS_URL = "http://localhost:8192"


class MCPParameters:
    """
    List of parameters used in the DQA MCP server.
    """

    # Deprecated parameters
    TOOL_SEPARATOR = "__"
    RESOURCE_SEPARATOR = "."
    PROMPT_SEPARATOR = "-"
