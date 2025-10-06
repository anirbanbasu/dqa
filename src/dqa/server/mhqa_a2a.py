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

from dqa import ParsedEnvVars
from dqa.executor.mhqa import MHQAAgentExecutor
from dqa.model.mhqa import MHQAAgentSkills


async def uvicorn_serve():
    def sigint_handler(signal, frame):
        """
        Signal handler to shut down the server gracefully.
        """
        print("Attempting graceful shutdown, please wait...")
        # This is absolutely necessary to exit the program
        sys.exit(0)

    _a2a_uvicorn_host = ParsedEnvVars().APP_A2A_SRV_HOST
    _a2a_uvicorn_port = ParsedEnvVars().APP_MHQA_A2A_SRV_PORT
    signal.signal(signal.SIGINT, sigint_handler)

    respond_skill = AgentSkill(
        id=f"{MHQAAgentSkills.Respond}_skill",
        name=MHQAAgentSkills.Respond,
        description="Responds to a multi-hop question from the user.",
        tags=["chat", MHQAAgentSkills.Respond],
        examples=["Hello there, tell me about your capabilities!"],
    )

    get_history_skill = AgentSkill(
        id=f"{MHQAAgentSkills.GetChatHistory}_skill",
        name=MHQAAgentSkills.GetChatHistory,
        description="Responds with a history of past user questions and their corresponding responses.",
        tags=[MHQAAgentSkills.GetChatHistory],
    )

    delete_history_skill = AgentSkill(
        id=f"{MHQAAgentSkills.ResetChatHistory}_skill",
        name=MHQAAgentSkills.ResetChatHistory,
        description="Deletes the history of past user questions and their corresponding responses.",
        tags=[MHQAAgentSkills.ResetChatHistory],
    )
    # This will be the public-facing agent card
    public_agent_card = AgentCard(
        name="Multi-Hop Question Answering Agent",
        description="An agent that can respond to multi-hop questions from the user, among other things.",
        url=f"http://{_a2a_uvicorn_host}:{_a2a_uvicorn_port}/",
        version="0.1.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[
            respond_skill,
            get_history_skill,
            delete_history_skill,
        ],  # Only the basic skill for the public card
        supports_authenticated_extended_card=False,
    )

    request_handler = DefaultRequestHandler(
        agent_executor=MHQAAgentExecutor(),
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
