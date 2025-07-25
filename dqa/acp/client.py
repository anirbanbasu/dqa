import asyncio
import sys
from acp_sdk import GenericEvent, MessageCompletedEvent, MessagePartEvent
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart


from rich import print_json
from rich import print as print


async def try_client():
    async with (
        Client(base_url="http://localhost:8192") as client,
        client.session() as client_session,
    ):
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

            log_type = None
            async for event in client_session.run_stream(
                agent="dqa_chat",
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
                        status_message = f"[bold green]ⓘ {event.type}[/bold green]"
                        if hasattr(event, "run"):
                            if hasattr(event.run, "run_id"):
                                status_message += f" run_id: {event.run.run_id}"
                            if hasattr(event.run, "session_id"):
                                status_message += f" session_id: {event.run.session_id}"
                        print(
                            status_message,
                            file=sys.stdout,
                        )


def main():
    asyncio.run(try_client())


if __name__ == "__main__":
    main()
