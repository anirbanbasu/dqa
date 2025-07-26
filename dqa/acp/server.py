import asyncio
import signal
import sys
import json
import uvicorn
from collections.abc import AsyncGenerator
from acp_sdk.server import agent, create_app, Context, RunYield, RunYieldResume
from acp_sdk.models import Artifact, Message, Metadata


from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

from rich import print as print

from dqa.agent.orchestrator import DQAOrchestrator
from dqa.common import EnvironmentVariables, ic
from dqa.utils import parse_env
from dotenv import load_dotenv

dqa_orchestrators = {}


def get_orchestrator(session_id: str) -> DQAOrchestrator:
    """
    Create a DQAOrchestrator instance for the given session ID.
    """
    if dqa_orchestrators.get(session_id) is None:
        dqa_orchestrators[session_id] = DQAOrchestrator(session_id=session_id)
    return dqa_orchestrators[session_id]


@agent(
    input_content_types=["text/plain"],
    output_content_types=["text/plain"],
    metadata=Metadata(
        license="MIT",
        programming_language="python",
        natural_languages=["english"],
        framework="ACP SDK, LlamaIndex Workflows",
        tags=["chat", "multi-hop question-answering", "llamaindex"],
        recommended_models=["ollama/mistral-nemo:12b", "ollama/qwen3:8b"],
    ),
)
async def chat(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Responds to non-trivial questions from the user."""
    session_id = str(context.session.id)
    query = str(input[-1])
    orchestrator: DQAOrchestrator = await asyncio.get_event_loop().run_in_executor(
        None, get_orchestrator, session_id
    )
    workflow_handler = orchestrator.run(query=query)
    async for ev in workflow_handler.stream_events():
        if isinstance(ev, AgentStream):
            yield ev.delta
        elif isinstance(ev, ToolCallResult):
            yield Artifact(
                name=f"tool_call_{ev.tool_id}",
                content_type="application/json",
                content=json.dumps(
                    ev.model_dump(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                ),
            )
        elif isinstance(ev, ToolCall):
            pass
        elif isinstance(ev, AgentOutput):
            yield "\n"
            yield "".join([block.text for block in ev.response.blocks])


@agent(
    input_content_types=["text/plain"],
    output_content_types=["application/json"],
    metadata=Metadata(
        license="MIT",
        programming_language="python",
        natural_languages=["english"],
        framework="ACP SDK, LlamaIndex Workflows",
        tags=["mcp", "features", "tools", "prompts", "resources"],
    ),
)
async def list_session_mcp_features(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    List all available MCP features for the session.
    """
    session_id = str(context.session.id)
    orchestrator: DQAOrchestrator = await asyncio.get_event_loop().run_in_executor(
        None, get_orchestrator, session_id
    )
    yield Artifact(
        name=f"mcp_features_{session_id}",
        content_type="application/json",
        content=json.dumps(
            [
                {
                    "name": feature.__dict__["_metadata"].name,
                    "description": feature.__dict__["_metadata"].description,
                }
                for feature in orchestrator.mcp_features
            ],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )


@agent(
    input_content_types=["text/plain"],
    output_content_types=["application/json"],
    metadata=Metadata(
        license="MIT",
        programming_language="python",
        natural_languages=["english"],
        framework="ACP SDK, LlamaIndex Workflows",
        tags=["llm", "config"],
    ),
)
async def get_session_llm_config(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Get LLM configuration for the session.
    """
    session_id = str(context.session.id)
    orchestrator: DQAOrchestrator = await asyncio.get_event_loop().run_in_executor(
        None, get_orchestrator, session_id
    )
    yield Artifact(
        name=f"llm_config_{session_id}",
        content_type="application/json",
        content=json.dumps(
            orchestrator.llm_config,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )


@agent(
    input_content_types=["text/plain"],
    output_content_types=["application/json"],
    metadata=Metadata(
        license="MIT",
        programming_language="python",
        natural_languages=["english"],
        framework="ACP SDK, LlamaIndex Workflows",
        tags=["chat", "history"],
    ),
)
async def get_session_chat_history(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Get chat history for the session.
    """
    session_id = str(context.session.id)
    orchestrator: DQAOrchestrator = await asyncio.get_event_loop().run_in_executor(
        None, get_orchestrator, session_id
    )
    yield Artifact(
        name=f"chat_history_{session_id}",
        content_type="application/json",
        content=json.dumps(
            [message.model_dump() for message in orchestrator.get_chat_history()],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )


@agent(
    input_content_types=["text/plain"],
    output_content_types=["application/json"],
    metadata=Metadata(
        license="MIT",
        programming_language="python",
        natural_languages=["english"],
        framework="ACP SDK, LlamaIndex Workflows",
        tags=["chat", "history", "reset"],
    ),
)
async def reset_session_chat_history(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """
    Get chat history for the session.
    """
    session_id = str(context.session.id)
    orchestrator: DQAOrchestrator = await asyncio.get_event_loop().run_in_executor(
        None, get_orchestrator, session_id
    )
    orchestrator.reset_chat_history()
    yield Artifact(
        name=f"chat_history_{session_id}",
        content_type="application/json",
        content=json.dumps(
            [message.model_dump() for message in orchestrator.get_chat_history()],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
    )


dqa_acp_app = create_app(
    chat,
    list_session_mcp_features,
    get_session_llm_config,
    get_session_chat_history,
    reset_session_chat_history,
)


async def uvicorn_serve():
    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        print("[green]Attempting graceful shutdown, please wait...[/green]")
        # This is absolutely necessary to exit the program
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    config = uvicorn.Config(
        dqa_acp_app,
        host=parse_env(
            var_name=EnvironmentVariables.DQA_ACP_HOST,
            default_value=EnvironmentVariables.DEFAULT__DQA_ACP_HOST,
        ),
        port=parse_env(
            var_name=EnvironmentVariables.DQA_ACP_PORT,
            default_value=EnvironmentVariables.DEFAULT__DQA_ACP_PORT,
            type_cast=int,
        ),
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """
    Main function to run the ACP server.
    """
    ic(load_dotenv())
    asyncio.run(uvicorn_serve())


if __name__ == "__main__":
    main()
