import signal
import sys
import time
import random
from acp_sdk.server import Server
from acp_sdk.models import Message, Metadata

from datetime import datetime, timezone

from rich import print as print

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
async def echo(input: list[Message]):
    """Echoes every message in the list with a UTC timestamp, and a final message after a little delay."""
    for message in input:
        yield f"{message} @{datetime.now(timezone.utc).isoformat()}\n"
    time.sleep(random.randint(1, 4))  # Simulate some processing delay
    yield "I wasn't done but now I am! This is a final message after a little delay."


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
    )


if __name__ == "__main__":
    main()
