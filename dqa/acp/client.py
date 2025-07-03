import asyncio
from acp_sdk.client import Client
from acp_sdk.models import Message, MessagePart

from rich import print_json


async def try_client():
    async with Client(base_url="http://localhost:8192") as client:
        async for agent in client.agents():
            print_json(agent.model_dump_json())
        run = await client.run_sync(
            agent="echo",
            input=[
                Message(parts=[MessagePart(content="Howdy!")]),
                Message(parts=[MessagePart(content="How are you going?")]),
            ],
        )
        print_json(run.model_dump_json())


def main():
    asyncio.run(try_client())


if __name__ == "__main__":
    main()
