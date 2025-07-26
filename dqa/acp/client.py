import asyncio
import sys
import uuid
from acp_sdk import GenericEvent, MessageCompletedEvent, MessagePartEvent, Session
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart


from rich import print_json
from rich import print as print

from textual.app import App, ComposeResult
from textual.widgets import Header

from dqa.common import EnvironmentVariables
from dqa.utils import parse_env


def get_client_session(existing_session_id: str | None = None) -> Client:
    existing_session = None
    if existing_session_id and len(existing_session_id.strip()) > 0:
        print(
            f"[bold yellow]Attempting to connect to existing session ID: {existing_session_id}[/bold yellow]"
        )
        existing_session = Session(id=uuid.UUID(existing_session_id))
    client = Client(
        base_url=parse_env(
            var_name=EnvironmentVariables.DQA_ACP_CLIENT_ACCESS_URL,
            default_value=EnvironmentVariables.DEFAULT__DQA_ACP_CLIENT_ACCESS_URL,
        )
    )
    client_session = client.session(session=existing_session)
    return client_session


async def acp_client(existing_session_id: str | None = None):
    async with get_client_session(existing_session_id) as client_session:
        async for agent in client_session.agents():
            print_json(agent.model_dump_json())
        while True:
            user_message = input(
                "(Type 'exit' or 'quit' to stop. Otherwise, enter your query) >>> "
            )
            if user_message.lower() in ("exit", "quit"):
                print("[bold red]Exiting...[/bold red]")
                break
            user_message_input = Message(parts=[MessagePart(content=user_message)])

            # run = await client_session.run_sync(
            #     agent="dqa_chat",
            #     input=[user_message_input],
            # )
            # print_json(run.output[-1].parts[-1].model_dump_json())

            agent_name = input("Name the agent to invoke [chat]: ").strip() or "chat"
            log_type = None
            async for event in client_session.run_stream(
                agent=agent_name,
                input=[user_message_input],
            ):
                match event:
                    case MessagePartEvent(part=MessagePart(content=content)):
                        if log_type:
                            print()
                            log_type = None
                        print(content, end="", flush=True)
                    case GenericEvent():
                        [(new_log_type, content)] = event.generic.model_dump().items()
                        if new_log_type != log_type:
                            if log_type is not None:
                                print()
                            print(
                                f"{new_log_type}: ", end="", file=sys.stdout, flush=True
                            )
                            log_type = new_log_type
                        print(content, end="", file=sys.stdout, flush=True)
                    case MessageCompletedEvent():
                        print()
                    case _:
                        if log_type:
                            print()
                            log_type = None
                        match event.type:
                            case "message.part":
                                print(
                                    f"[bold green]ⓘ {event.type}[/bold green]\n{event.part.content}",
                                    file=sys.stdout,
                                )
                            case _:
                                status_message = (
                                    f"[bold green]ⓘ {event.type}[/bold green]"
                                )
                                if hasattr(event, "run"):
                                    if hasattr(event.run, "run_id"):
                                        status_message += f" run_id: {event.run.run_id}"
                                    if hasattr(event.run, "session_id"):
                                        status_message += (
                                            f" session_id: {event.run.session_id}"
                                        )
                                print(
                                    status_message,
                                    file=sys.stdout,
                                )


class ACPClient(App):
    def compose(self) -> ComposeResult:
        yield Header(name="DQA ACP Client", show_clock=True, time_format="%H:%M:%S")


def main():
    session_id_from_env = parse_env(var_name="DQA_SESSION_ID", default_value="")
    asyncio.run(acp_client(session_id_from_env))
    # app = ACPClient()
    # app.run()


if __name__ == "__main__":
    main()
