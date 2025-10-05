import asyncio
import logging

import signal
import sys
from types import FrameType
from typing import List
from uuid import uuid4

from pydantic import TypeAdapter
from rich import print_json

from a2a.utils import get_message_text

import httpx

from dqa import env

from a2a.types import Message

from dqa.model.echo_task import (
    DeleteEchoHistoryInput,
    EchoAgentA2AInputMessage,
    EchoAgentSkills,
    EchoHistoryInput,
    EchoInput,
    EchoResponse,
    EchoResponseWithHistory,
)

from dqa import ic  # noqa: F401

import typer


from dqa.client.a2a_mixin import A2AClientMixin

from dqa.model.mhqa import (
    MHQAAgentInputMessage,
    MHQAAgentSkills,
    MHQADeleteHistoryInput,
    MHQAHistoryInput,
    MHQAInput,
    MHQAResponse,
)

logger = logging.getLogger(__name__)  # Get a logger instance

app = typer.Typer(
    name="DQA A2A CLI",
    help="A command-line interface for DQA",
    no_args_is_help=True,
    add_completion=False,
)


class DQACliApp(A2AClientMixin):
    def __init__(self):
        # Set up signal handlers for graceful shutdown
        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self._interrupt_handler)

    def _interrupt_handler(self, signum: int, frame: FrameType | None):
        logger.warning("Interrupt signal received, performing clean shutdown")
        logger.debug(f"Interrupt signal number: {signum}. Frame: {frame}")
        # Cleanup will be performed due to the finally block in each command
        sys.exit(0)

    def _initialize(self):
        logger.debug("Initialising A2A server URLs...")
        a2a_asgi_host = env.str("APP_A2A_SRV_HOST", "127.0.0.1")
        echo_a2a_asgi_port = env.int("APP_ECHO_A2A_SRV_PORT", 32769)
        self.echo_base_url = f"http://{a2a_asgi_host}:{echo_a2a_asgi_port}"

        mhqa_a2a_asgi_port = env.int("APP_ECHO_A2A_SRV_PORT", 32770)
        self.mhqa_base_url = f"http://{a2a_asgi_host}:{mhqa_a2a_asgi_port}"
        logger.debug(f"Echo A2A base URL: {self.echo_base_url}")
        logger.debug(f"MHQA A2A base URL: {self.mhqa_base_url}")

    def _cleanup(self):
        logger.debug("Running cleanup...")
        logger.debug("Cleanup completed.")

    async def _hello(self, name: str) -> str:
        return f"Hello, {name}!"

    async def run_hello(self, name: str):
        try:
            self._initialize()
            result = await self._hello(name=name)
            print(result)
        except Exception as e:
            logger.error(f"Error in showing config. {e}")
        finally:
            self._cleanup()

    async def _echo_a2a_echo(
        self,
        message: str,
        thread_id: str,
    ) -> EchoResponseWithHistory:
        async with httpx.AsyncClient() as httpx_client:
            client = await self.obtain_a2a_client(
                httpx_client=httpx_client,
                base_url=self.echo_base_url,
            )

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
            full_message_content = ""
            async for response in streaming_response:
                if isinstance(response, Message):
                    full_message_content += get_message_text(response)
            validated_response = EchoResponseWithHistory.model_validate_json(
                full_message_content
            )
            validated_response.past = validated_response.past[
                ::-1
            ]  # Reverse to chronological order to look right in the CLI
            return validated_response

    async def run_echo_a2a_echo(
        self,
        message: str,
        thread_id: str,
    ):
        try:
            self._initialize()
            response = await self._echo_a2a_echo(
                message=message,
                thread_id=thread_id,
            )
            print_json(response.model_dump_json())
        except Exception as e:
            logger.error(f"Error in echo A2A echo. {e}")
        finally:
            self._cleanup()

    async def _echo_a2a_history(
        self,
        thread_id: str,
    ) -> List[EchoResponse]:
        async with httpx.AsyncClient() as httpx_client:
            client = await self.obtain_a2a_client(
                httpx_client=httpx_client,
                base_url=self.echo_base_url,
            )

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
            full_message_content = ""
            async for response in streaming_response:
                if isinstance(response, Message):
                    full_message_content += get_message_text(response)
            response_adapter = TypeAdapter(List[EchoResponse])
            validated_response = response_adapter.validate_json(full_message_content)
            validated_response = validated_response[
                ::-1
            ]  # Reverse to chronological order to look right in the CLI
            return validated_response

    async def run_echo_a2a_history(
        self,
        thread_id: str,
    ):
        try:
            self._initialize()
            response = await self._echo_a2a_history(
                thread_id=thread_id,
            )
            response_adapter = TypeAdapter(List[EchoResponse])
            print_json(response_adapter.dump_json(response).decode())
        except Exception as e:
            logger.error(f"Error in echo A2A history. {e}")
        finally:
            self._cleanup()

    async def _echo_a2a_delete_history(
        self,
        thread_id: str,
    ) -> str:
        async with httpx.AsyncClient() as httpx_client:
            client = await self.obtain_a2a_client(
                httpx_client=httpx_client,
                base_url=self.echo_base_url,
            )

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
            full_message_content = ""
            async for response in streaming_response:
                if isinstance(response, Message):
                    full_message_content += get_message_text(response)
            return full_message_content

    async def run_echo_a2a_delete_history(
        self,
        thread_id: str,
    ):
        try:
            self._initialize()
            response = await self._echo_a2a_delete_history(
                thread_id=thread_id,
            )
            print(response)
        except Exception as e:
            logger.error(f"Error in echo A2A delete history. {e}")
        finally:
            self._cleanup()

    async def _mhqa_chat(
        self,
        message: str,
        thread_id: str,
    ) -> MHQAResponse:
        async with httpx.AsyncClient() as httpx_client:
            client = await self.obtain_a2a_client(
                httpx_client=httpx_client,
                base_url=self.mhqa_base_url,
            )

            message_payload = MHQAAgentInputMessage(
                skill=MHQAAgentSkills.Respond,
                data=MHQAInput(
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
            full_message_content = ""
            async for response in streaming_response:
                if isinstance(response, Message):
                    full_message_content += get_message_text(response)
            validated_response = MHQAResponse.model_validate_json(full_message_content)
            return validated_response

    async def run_mhqa_chat(
        self,
        message: str,
        thread_id: str,
    ):
        try:
            self._initialize()
            response = await self._mhqa_chat(
                message=message,
                thread_id=thread_id,
            )
            print_json(response.model_dump_json())
        except Exception as e:
            logger.error(f"Error in MHQA chat. {e}")
        finally:
            self._cleanup()

    async def _mhqa_get_history(
        self,
        thread_id: str,
    ) -> List[MHQAResponse]:
        async with httpx.AsyncClient() as httpx_client:
            client = await self.obtain_a2a_client(
                httpx_client=httpx_client,
                base_url=self.mhqa_base_url,
            )

            message_payload = MHQAAgentInputMessage(
                skill=MHQAAgentSkills.GetChatHistory,
                data=MHQAHistoryInput(thread_id=thread_id),
            )

            send_message = Message(
                role="user",
                parts=[{"kind": "text", "text": message_payload.model_dump_json()}],
                message_id=str(uuid4()),
            )
            logger.info("Sending message to the A2A endpoint")
            streaming_response = client.send_message(send_message)
            logger.info("Parsing streaming response from the A2A endpoint")
            full_message_content = ""
            async for response in streaming_response:
                if isinstance(response, Message):
                    full_message_content += get_message_text(response)
            response_adapter = TypeAdapter(List[MHQAResponse])
            validated_response = response_adapter.validate_json(full_message_content)
            validated_response = validated_response[
                ::-1
            ]  # Reverse to chronological order to look right in the CLI
            return validated_response

    async def run_mhqa_get_history(
        self,
        thread_id: str,
    ):
        try:
            self._initialize()
            response = await self._mhqa_get_history(
                thread_id=thread_id,
            )
            response_adapter = TypeAdapter(List[MHQAResponse])
            print_json(response_adapter.dump_json(response).decode())
        except Exception as e:
            logger.error(f"Error in MHQA get history. {e}")
        finally:
            self._cleanup()

    async def _mhqa_delete_history(
        self,
        thread_id: str,
    ) -> str:
        async with httpx.AsyncClient() as httpx_client:
            client = await self.obtain_a2a_client(
                httpx_client=httpx_client,
                base_url=self.mhqa_base_url,
            )

            message_payload = MHQAAgentInputMessage(
                skill=MHQAAgentSkills.ResetChatHistory,
                data=MHQADeleteHistoryInput(thread_id=thread_id),
            )

            send_message = Message(
                role="user",
                parts=[{"kind": "text", "text": message_payload.model_dump_json()}],
                message_id=str(uuid4()),
            )
            logger.info("Sending message to the A2A endpoint")
            streaming_response = client.send_message(send_message)
            logger.info("Parsing streaming response from the A2A endpoint")
            full_message_content = ""
            async for response in streaming_response:
                if isinstance(response, Message):
                    full_message_content += get_message_text(response)
            return full_message_content

    async def run_mhqa_delete_history(
        self,
        thread_id: str,
    ):
        try:
            self._initialize()
            response = await self._mhqa_delete_history(
                thread_id=thread_id,
            )
            print(response)
        except Exception as e:
            logger.error(f"Error in MHQA delete history. {e}")
        finally:
            self._cleanup()


@app.command()
def hello(
    name: str = typer.Argument(default="World", help="The name to greet."),
) -> None:
    """
    A simple hello world command. This is a placeholder to ensure that
    there are more than one actual commands in this CLI app.
    """
    app_handler = DQACliApp()
    asyncio.run(app_handler.run_hello(name=name))


@app.command()
def echo_a2a_echo(
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
    app_handler = DQACliApp()
    asyncio.run(app_handler.run_echo_a2a_echo(message=message, thread_id=thread_id))


@app.command()
def echo_a2a_history(
    thread_id: str = typer.Option(
        help="A thread ID to identify your conversation.",
    ),
) -> None:
    """
    Retrieve the history of messages for a given thread ID from the A2A endpoint.
    """

    app_handler = DQACliApp()
    asyncio.run(app_handler.run_echo_a2a_history(thread_id=thread_id))


@app.command()
def echo_a2a_delete_history(
    thread_id: str = typer.Option(
        help="A thread ID to identify your conversation.",
    ),
) -> None:
    """
    Delete the history of messages for a given thread ID from the A2A endpoint.
    """

    app_handler = DQACliApp()
    asyncio.run(app_handler.run_echo_a2a_delete_history(thread_id=thread_id))


@app.command()
def mhqa_chat(
    message: str = typer.Argument(
        default="Hello there, tell me about your capabilities!",
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

    app_handler = DQACliApp()
    asyncio.run(app_handler.run_mhqa_chat(message=message, thread_id=thread_id))


@app.command()
def mhqa_get_history(
    thread_id: str = typer.Option(
        help="A thread ID to identify your conversation.",
    ),
) -> None:
    """
    Obtain the history of messages for a given thread ID from the A2A endpoint.
    """

    app_handler = DQACliApp()
    asyncio.run(app_handler.run_mhqa_get_history(thread_id=thread_id))


@app.command()
async def mhqa_delete_history(
    thread_id: str = typer.Option(
        help="A thread ID to identify your conversation.",
    ),
) -> None:
    """
    Delete the history of messages for a given thread ID from the A2A endpoint.
    """

    app_handler = DQACliApp()
    asyncio.run(app_handler.run_mhqa_delete_history(thread_id=thread_id))


def main():  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
