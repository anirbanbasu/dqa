from functools import partial
import logging

from typing import List
from uuid import uuid4
from asyncer import syncify

from pydantic import TypeAdapter
from rich import print_json

from a2a.utils import get_message_text

import httpx

from py_a2a_dapr import env

from a2a.client import A2ACardResolver, ClientFactory, ClientConfig
from a2a.types import (
    AgentCard,
    Message,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
)

from py_a2a_dapr.model.echo_task import (
    DeleteEchoHistoryInput,
    EchoAgentA2AInputMessage,
    EchoAgentSkills,
    EchoHistoryInput,
    EchoInput,
    EchoResponse,
    EchoResponseWithHistory,
)

from py_a2a_dapr import ic  # noqa: F401

import typer

logger = logging.getLogger(__name__)  # Get a logger instance

a2a_asgi_host = env.str("APP_A2A_SRV_HOST", "127.0.0.1")
echo_a2a_asgi_port = env.int("APP_ECHO_A2A_SRV_PORT", 32769)
base_url = f"http://{a2a_asgi_host}:{echo_a2a_asgi_port}"

cli_app = typer.Typer(
    name="a2a-client",
    help="An A2A client example for py-a2a-dapr",
    no_args_is_help=True,
    add_completion=False,
)


@cli_app.command()
def hello(
    name: str = typer.Argument(default="World", help="The name to greet."),
) -> None:
    """
    A simple hello world command. This is a placeholder to ensure that
    there are more than one actual commands in this CLI app.
    """
    print(f"Hello, {name}!")


@cli_app.command()
@partial(syncify, raise_sync_error=False)
async def echo_a2a_echo(
    message: str = typer.Argument(
        default="Hello there, from an A2A client!",
        help="The message to send to the A2A endpoint.",
    ),
    thread_id: str = typer.Option(
        default=str(uuid4()),
        help="A thread ID to identify your conversation. If not specified, a random UUID will be used.",
    ),
) -> None:
    """
    Query the echo A2A endpoint with a message and print the response.
    """

    async with httpx.AsyncClient() as httpx_client:
        # initialise A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
            # agent_card_path uses default, extended_agent_card_path also uses default
        )
        final_agent_card_to_use: AgentCard | None = None

        logger.info(
            f"Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}"
        )
        _public_card = (
            await resolver.get_agent_card()
        )  # Fetches from default public path
        logger.info("Successfully fetched public agent card:")
        logger.info(_public_card.model_dump_json(indent=2, exclude_none=True))
        final_agent_card_to_use = _public_card

        client = ClientFactory(
            config=ClientConfig(streaming=True, polling=True, httpx_client=httpx_client)
        ).create(card=final_agent_card_to_use)
        logger.info("A2A client initialised.")

        message_payload = EchoAgentA2AInputMessage(
            skill=EchoAgentSkills.ECHO,
            data=EchoInput(
                thread_id=thread_id,
                user_input=message,
            ),
        )

        send_message = Message(
            role="user",
            parts=[{"kind": "text", "text": message_payload.model_dump_json()}],
            message_id=str(uuid4()),
        )
        logger.info("Sending message to the A2A endpoint")
        streaming_response = client.send_message(send_message)
        logger.info("Parsing streaming response from the A2A endpoint")
        async for response in streaming_response:
            if isinstance(response, Message):
                full_message_content = get_message_text(response)
                validated_response = EchoResponseWithHistory.model_validate_json(
                    full_message_content
                )
                validated_response.past = validated_response.past[
                    ::-1
                ]  # Reverse to chronological order to look right in the CLI
                print_json(validated_response.model_dump_json())


@cli_app.command()
@partial(syncify, raise_sync_error=False)
async def echo_a2a_history(
    thread_id: str = typer.Option(
        help="A thread ID to identify your conversation.",
    ),
) -> None:
    """
    Retrieve the history of messages for a given thread ID from the A2A endpoint.
    """

    async with httpx.AsyncClient() as httpx_client:
        # initialise A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
            # agent_card_path uses default, extended_agent_card_path also uses default
        )
        final_agent_card_to_use: AgentCard | None = None

        logger.info(
            f"Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}"
        )
        _public_card = (
            await resolver.get_agent_card()
        )  # Fetches from default public path
        logger.info("Successfully fetched public agent card:")
        logger.info(_public_card.model_dump_json(indent=2, exclude_none=True))
        final_agent_card_to_use = _public_card

        client = ClientFactory(
            config=ClientConfig(streaming=True, polling=True, httpx_client=httpx_client)
        ).create(card=final_agent_card_to_use)
        logger.info("A2A client initialised.")

        message_payload = EchoAgentA2AInputMessage(
            skill=EchoAgentSkills.HISTORY,
            data=EchoHistoryInput(
                thread_id=thread_id,
            ),
        )

        send_message = Message(
            role="user",
            parts=[{"kind": "text", "text": message_payload.model_dump_json()}],
            message_id=str(uuid4()),
        )
        logger.info("Sending message to the A2A endpoint")
        streaming_response = client.send_message(send_message)
        logger.info("Parsing streaming response from the A2A endpoint")
        response_adapter = TypeAdapter(List[EchoResponse])
        async for response in streaming_response:
            if isinstance(response, Message):
                full_message_content = get_message_text(response)
                validated_response = response_adapter.validate_json(
                    full_message_content
                )
                validated_response = validated_response[
                    ::-1
                ]  # Reverse to chronological order to look right in the CLI
                print_json(response_adapter.dump_json(validated_response).decode())


@cli_app.command()
@partial(syncify, raise_sync_error=False)
async def echo_a2a_delete_history(
    thread_id: str = typer.Option(
        help="A thread ID to identify your conversation.",
    ),
) -> None:
    """
    Delete the history of messages for a given thread ID from the A2A endpoint.
    """

    async with httpx.AsyncClient() as httpx_client:
        # initialise A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=base_url,
            # agent_card_path uses default, extended_agent_card_path also uses default
        )
        final_agent_card_to_use: AgentCard | None = None

        logger.info(
            f"Attempting to fetch public agent card from: {base_url}{AGENT_CARD_WELL_KNOWN_PATH}"
        )
        _public_card = (
            await resolver.get_agent_card()
        )  # Fetches from default public path
        logger.info("Successfully fetched public agent card:")
        logger.info(_public_card.model_dump_json(indent=2, exclude_none=True))
        final_agent_card_to_use = _public_card

        client = ClientFactory(
            config=ClientConfig(streaming=True, polling=True, httpx_client=httpx_client)
        ).create(card=final_agent_card_to_use)
        logger.info("A2A client initialised.")

        message_payload = EchoAgentA2AInputMessage(
            skill=EchoAgentSkills.DELETE_HISTORY,
            data=DeleteEchoHistoryInput(
                thread_id=thread_id,
            ),
        )

        send_message = Message(
            role="user",
            parts=[{"kind": "text", "text": message_payload.model_dump_json()}],
            message_id=str(uuid4()),
        )
        logger.info("Sending message to the A2A endpoint")
        streaming_response = client.send_message(send_message)
        logger.info("Parsing streaming response from the A2A endpoint")
        async for response in streaming_response:
            if isinstance(response, Message):
                full_message_content = get_message_text(response)
                print(full_message_content)


def main():  # pragma: no cover
    try:
        cli_app()
    except Exception as e:
        logger.error(f"Critical error running the CLI app. {e}", exc_info=True)


if __name__ == "__main__":  # pragma: no cover
    main()
