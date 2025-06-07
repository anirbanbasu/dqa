try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class EnvironmentVariables:
    """
    List of environment variables used in the DQA project.
    """

    ENVVAR_DQA__MCP_SERVER_TRANSPORT = "DQA_MCP_SERVER_TRANSPORT"
    ENVVAR_DEFAULT_DQA__MCP_SERVER_TRANSPORT = "sse"
    ENVVAR_ALLOWED__DQA_MCP_SERVER_TRANSPORT = ["sse", "streamable-http"]

    ENVVAR__DQA_LLM_CONFIG = "DQA_LLM_CONFIG"
    ENVVAR_DEFAULT__DQA_LLM_CONFIG = "config/chat-ollama.json"
