import logging
import re
from typing import ClassVar
from rich.logging import RichHandler
from environs import Env

from marshmallow.validate import OneOf, Range, Regexp

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # pragma: no cover
    # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

env = Env()
env.read_env()


class EnvVars:
    APP_LOG_LEVEL: str = env.str("APP_LOG_LEVEL", default="INFO").upper()
    HTTPX_TIMEOUT: float = env.float(
        "HTTPX_TIMEOUT",
        default=120.0,
        validate=Range(min=5.0, max=600.0),
    )
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
    APP_DAPR_SVC_PORT: int = env.int(
        "APP_DAPR_SVC_PORT",
        default=32768,
        validate=Range(min=4096, max=65535),
    )
    APP_DAPR_PUBSUB_STALE_MSG_SECS: int = env.int(
        "APP_DAPR_PUBSUB_STALE_MSG_SECS",
        default=60,
        validate=Range(min=5, max=3600),
    )
    APP_DAPR_ACTOR_RETRY_ATTEMPTS: int = env.int(
        "APP_DAPR_ACTOR_RETRY_ATTEMPTS",
        default=3,
        validate=Range(min=1, max=10),
    )
    AGENT_RETRY_ATTEMPTS: int = env.int(
        "AGENT_RETRY_ATTEMPTS",
        default=3,
        validate=Range(min=1, max=10),
    )
    APP_A2A_SRV_HOST: str = env.str("APP_A2A_SRV_HOST", default="127.0.0.1")
    APP_MHQA_A2A_SRV_PORT: int = env.int(
        "APP_MHQA_A2A_SRV_PORT",
        default=32770,
        validate=Range(min=4096, max=65535),
    )
    APP_MHQA_A2A_REMOTE_URL: str = env.str("APP_MHQA_A2A_REMOTE_URL", default=None)
    DAPR_PUBSUB_NAME: str = env.str("DAPR_PUBSUB_NAME", default="pubsub")
    MCP_SERVER_HOST: str = FASTMCP_HOST
    MCP_SERVER_PORT: int = FASTMCP_PORT
    BROWSER_STATE_SECRET: str = env.str(
        "BROWSER_STATE_SECRET",
        default="a2a_dapr_bstate_secret",
        validate=Regexp(
            r"^[A-Za-z0-9]{1,8}(?:_[A-Za-z0-9]{1,8}){1,4}$", flags=re.IGNORECASE
        ),
    )
    BROWSER_STATE_CHAT_HISTORIES: str = env.str(
        "BROWSER_STATE_CHAT_HISTORIES",
        default="a2a_dapr_chat_histories",
        validate=Regexp(
            r"^[A-Za-z0-9]{1,8}(?:_[A-Za-z0-9]{1,8}){1,4}$", flags=re.IGNORECASE
        ),
    )

    PREFECT_API_URL: str = env.str("PREFECT_API_URL", default=None)

    API_KEY_ALPHAVANTAGE: str = env.str("ALPHAVANTAGE_API_KEY", default=None)
    API_KEY_TAVILY: str = env.str("TAVILY_API_KEY", default=None)
    API_KEY_OLLAMA: str = env.str("OLLAMA_API_KEY", default=None)

    _instance: ClassVar = None

    def __new__(cls: type["EnvVars"]) -> "EnvVars":
        if cls._instance is None:
            # Create instance using super().__new__ to bypass any recursion
            instance = super().__new__(cls)
            cls._instance = instance
        return cls._instance


logging.basicConfig(
    level=EnvVars.APP_LOG_LEVEL,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(show_time=True, show_level=True, show_path=True)],
)
