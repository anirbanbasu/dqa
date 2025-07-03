import signal
import sys
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
        recommended_models=["ollama/mistral-nemo"],
    )
)
async def echo(input: list[Message]):
    """Echoes every message in the list with a UTC timestamp"""
    for message in input:
        yield f"{message} @{datetime.now(timezone.utc).isoformat()}"


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
