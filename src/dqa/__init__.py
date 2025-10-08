import logging
from typing import ClassVar
from rich.logging import RichHandler
from environs import Env

from marshmallow.validate import OneOf

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # pragma: no cover
    # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

env = Env()
env.read_env()


class ParsedEnvVars:
    APP_LOG_LEVEL: str = env.str("APP_LOG_LEVEL", default="INFO").upper()
    DQA_MCP_SERVER_TRANSPORT: str = env.str(
        "DQA_MCP_SERVER_TRANSPORT",
        default="stdio",
        validate=OneOf(["stdio", "sse", "streamable-http"]),
    )
    FASTMCP_HOST: str = env.str("FASTMCP_HOST", default="localhost")
    FASTMCP_PORT: int = env.int("FASTMCP_PORT", default=8000)
    LLM_CONFIG_FILE: str = env.str("LLM_CONFIG_FILE", default="conf/llm.json")
    MCP_CONFIG_FILE: str = env.str("MCP_CONFIG_FILE", default="conf/mcp.json")
    APP_DAPR_SVC_HOST: str = env.str("APP_DAPR_SVC_HOST", default="127.0.0.1")
    APP_DAPR_SVC_PORT: int = env.int("APP_DAPR_SVC_PORT", default=32768)
    APP_A2A_SRV_HOST: str = env.str("APP_A2A_SRV_HOST", default="127.0.0.1")
    APP_MHQA_A2A_SRV_PORT: int = env.int("APP_MHQA_A2A_SRV_PORT", default=32770)
    APP_ECHO_A2A_SRV_PORT: int = env.int("APP_ECHO_A2A_SRV_PORT", default=32769)
    DAPR_PUBSUB_NAME: str = env.str("DAPR_PUBSUB_NAME", default="pubsub")
    MCP_SERVER_HOST: str = env.str("FASTMCP_HOST", default="localhost")
    MCP_SERVER_PORT: int = env.int("FASTMCP_PORT", default=8000)
    BROWSER_STATE_SECRET: str = env.str(
        "BROWSER_STATE_SECRET", default="a2a_dapr_bstate_secret"
    )
    BROWSER_STATE_CHAT_HISTORIES: str = env.str(
        "BROWSER_STATE_CHAT_HISTORIES", default="a2a_dapr_chat_histories"
    )

    _instance: ClassVar = None

    def __new__(cls: type["ParsedEnvVars"]) -> "ParsedEnvVars":
        if cls._instance is None:
            # Create instance using super().__new__ to bypass any recursion
            instance = super().__new__(cls)
            cls._instance = instance
        return cls._instance


logging.basicConfig(
    level=ParsedEnvVars().APP_LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_time=True, show_level=True, show_path=True)],
)
