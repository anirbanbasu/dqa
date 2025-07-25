import asyncio
import signal
import sys
from collections.abc import AsyncGenerator
from acp_sdk.server import Server, Context, RunYield, RunYieldResume
from acp_sdk.models import Message, Metadata


from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

from rich import print as print

from dqa.agent.orchestrator import DQAOrchestrator

server = Server()


dqa_orchestrators = {}


def get_orchestrator(session_id: str) -> DQAOrchestrator:
    """
    Create a DQAOrchestrator instance for the given session ID.
    """
    if dqa_orchestrators.get(session_id) is None:
        dqa_orchestrators[session_id] = DQAOrchestrator(session_id=session_id)
    return dqa_orchestrators[session_id]


@server.agent(
    metadata=Metadata(
        license="MIT",
        programming_language="python",
        natural_languages=["english"],
        framework="ACP SDK, LlamaIndex Workflows",
        tags=["dqa_chat"],
        recommended_models=["ollama/mistral-nemo:12b", "ollama/qwen3:8b"],
    )
)
async def dqa_chat(
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
            pass
        elif isinstance(ev, ToolCall):
            pass
        elif isinstance(ev, AgentOutput):
            yield "\n"
            yield "".join([block.text for block in ev.response.blocks])
    # response = await workflow_handler
    # yield str(response)


def main():
    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        print("[green]Attempting graceful shutdown, please wait...[/green]")
        # This is absolutely necessary to exit the program
        sys.exit(0)

    signal.signal(signal.SIGINT, sigint_handler)

    server.run(
        # Let's make these configurable
        port=8192,
        # store=MemoryStore(limit=10000, ttl=60),
    )


if __name__ == "__main__":
    main()
