import asyncio
import signal
import sys
from collections.abc import AsyncGenerator
from acp_sdk.server import Server, Context, RunYield, RunYieldResume
from acp_sdk.models import Message, Metadata

from datetime import datetime, timezone

from rich import print as print

from dqa.agent.orchestrator import DQAOrchestrator
from dqa.common import ic

server = Server()


@server.agent(
    metadata=Metadata(
        license="MIT",
        programming_language="python",
        natural_languages=["english"],
        framework="ACP SDK",
        tags=["echo", "example"],
        recommended_models=["nothing as we do not use any model here"],
    )
)
async def echo(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Echoes every message in the list with a UTC timestamp, and a final message after a little delay."""
    ic(context.__dict__)
    async for message in context.session.load_history():
        yield message
    state = await context.session.load_state()
    ic(state)
    for message in input:
        await asyncio.sleep(0.5)
        yield f"{message} @{datetime.now(timezone.utc).isoformat()}\n"


async def dqa_chat(input: list[Message]):
    """Responds to non-trivial questions from the user."""
    # TODO: This is a placeholder for the DQA orchestrator.
    orchestrator = DQAOrchestrator()
    for message in input:
        # Need to convert LlamaIndex events to ACP compatible ones.
        orchestrator.run(query=message)


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
