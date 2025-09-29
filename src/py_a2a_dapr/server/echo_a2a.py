# server.py
import asyncio
import signal
import sys
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from py_a2a_dapr import env
from py_a2a_dapr.executor.echo_task import EchoAgentExecutor
from py_a2a_dapr.model.echo_task import EchoAgentSkills


async def uvicorn_serve():
    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        print("Attempting graceful shutdown, please wait...")
        # This is absolutely necessary to exit the program
        sys.exit(0)

    _a2a_uvicorn_host = env.str("APP_A2A_SRV_HOST", "127.0.0.1")
    _a2a_uvicorn_port = env.int("APP_ECHO_A2A_SRV_PORT", 32769)
    signal.signal(signal.SIGINT, sigint_handler)

    echo_skill = AgentSkill(
        id=f"{EchoAgentSkills.ECHO}_skill",
        name=EchoAgentSkills.ECHO.capitalize(),
        description="Echo input messages along with a history of the conversation.",
        tags=["hello", EchoAgentSkills.ECHO],
        examples=["Ahoy, matey!", "Hello, world!", "Good morning!"],
    )

    history_skill = AgentSkill(
        id=f"{EchoAgentSkills.HISTORY}_skill",
        name=EchoAgentSkills.HISTORY.capitalize(),
        description="Responds with a history of past messages and their corresponding echoed responses.",
        tags=[EchoAgentSkills.HISTORY],
    )

    delete_history_skill = AgentSkill(
        id=f"{EchoAgentSkills.DELETE_HISTORY}_skill",
        name=EchoAgentSkills.DELETE_HISTORY.capitalize(),
        description="Deletes the history of past messages and their corresponding echoed responses.",
        tags=[EchoAgentSkills.DELETE_HISTORY],
    )
    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name="Echo Agent",
        description="An agent that can echo input messages, among other things.",
        url=f"http://{_a2a_uvicorn_host}:{_a2a_uvicorn_port}/",
        version="0.1.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            echo_skill,
            history_skill,
            delete_history_skill,
        ],  # Only the basic skill for the public card
        supports_authenticated_extended_card=False,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=EchoAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    a2a_app = A2AStarletteApplication(
        agent_card=public_agent_card,
        http_handler=request_handler,
    )
    config = uvicorn.Config(
        a2a_app.build(),
        host=_a2a_uvicorn_host,
        port=_a2a_uvicorn_port,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


def main():
    """
    Main function to run the ACP server.
    """
    asyncio.run(uvicorn_serve())


if __name__ == "__main__":
    main()
