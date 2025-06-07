try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class EnvironmentVariables:
    """
    List of environment variables used in the DQA project.
    """

    DQA__MCP_SERVER_TRANSPORT = "DQA_MCP_SERVER_TRANSPORT"
    DEFAULT_DQA__MCP_SERVER_TRANSPORT = "sse"
    ALLOWED__DQA_MCP_SERVER_TRANSPORT = ["sse", "streamable-http"]

    DQA_LLM_CONFIG = "DQA_LLM_CONFIG"
    DEFAULT__DQA_LLM_CONFIG = "config/chat-ollama.json"

    DQA_MCP_CLIENT_CONFIG = "DQA_MCP_CLIENT_CONFIG"
    DEFAULT_DQA_MCP_CLIENT_CONFIG = "config/mcp-client.json"


class MCPParameters:
    """
    List of parameters used in the DQA MCP server.
    """

    TOOL_SEPARATOR = "__"
    RESOURCE_SEPARATOR = "."
    PROMPT_SEPARATOR = "-"
