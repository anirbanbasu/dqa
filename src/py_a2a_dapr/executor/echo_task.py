from dapr.actor import ActorProxy, ActorId, ActorProxyFactory
from dapr.clients.retry import RetryPolicy
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from py_a2a_dapr.actor.echo_task import EchoTaskActorInterface
from py_a2a_dapr.model.echo_task import (
    DeleteEchoHistoryInput,
    EchoHistoryInput,
    EchoInput,
    EchoAgentA2AInputMessage,
    EchoAgentSkills,
)


class EchoAgentExecutor(AgentExecutor):
    def __init__(self):
        self._actor_type = "EchoTaskActor"
        self._factory = ActorProxyFactory(retry_policy=RetryPolicy(max_attempts=3))

    async def perform_echo(self, data: EchoInput) -> str:
        proxy = ActorProxy.create(
            actor_type=self._actor_type,
            actor_id=ActorId(actor_id=data.thread_id),
            actor_interface=EchoTaskActorInterface,
            actor_proxy_factory=self._factory,
        )
        result = await proxy.invoke_method(
            method="Echo", raw_body=data.model_dump_json().encode()
        )
        return result.decode().strip("\"'")

    async def perform_history(self, data: EchoHistoryInput) -> str:
        proxy = ActorProxy.create(
            actor_type=self._actor_type,
            actor_id=ActorId(actor_id=data.thread_id),
            actor_interface=EchoTaskActorInterface,
            actor_proxy_factory=self._factory,
        )
        result = await proxy.invoke_method(method="History")
        return result.decode().strip("\"'")

    async def perform_delete_history(self, data: DeleteEchoHistoryInput) -> str:
        proxy = ActorProxy.create(
            actor_type=self._actor_type,
            actor_id=ActorId(actor_id=data.thread_id),
            actor_interface=EchoTaskActorInterface,
            actor_proxy_factory=self._factory,
        )
        result = await proxy.invoke_method(
            method="DeleteHistory",
        )
        return result.decode().strip("\"'")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        message_payload = EchoAgentA2AInputMessage.model_validate_json(
            context.get_user_input()
        )
        if (
            not message_payload
            or not message_payload.data
            or message_payload.data.thread_id.strip() == ""
        ):
            raise ValueError(("Missing mandatory thread_id in the input!"))

        response = None
        match message_payload.skill:
            case EchoAgentSkills.ECHO:
                response = await self.perform_echo(data=message_payload.data)
            case EchoAgentSkills.HISTORY:
                response = await self.perform_history(data=message_payload.data)
            case EchoAgentSkills.DELETE_HISTORY:
                response = await self.perform_delete_history(data=message_payload.data)
            case _:
                raise ValueError(f"Unknown skill '{message_payload.skill}' requested!")
        if response:
            await event_queue.enqueue_event(new_agent_text_message(text=response))
        else:
            raise ValueError("No response received from the actor(s)!")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        message_payload = EchoAgentA2AInputMessage.model_validate_json(
            context.get_user_input()
        )
        if (
            not message_payload
            or not message_payload.data
            or message_payload.data.thread_id.strip() == ""
        ):
            raise ValueError(("Missing mandatory thread_id in the input!"))

        proxy = ActorProxy.create(
            actor_type=self._actor_type,
            actor_id=ActorId(actor_id=message_payload.data.thread_id),
            actor_interface=EchoTaskActorInterface,
            actor_proxy_factory=self._factory,
        )
        result = await proxy.invoke_method(method="Cancel")
        await event_queue.enqueue_event(
            new_agent_text_message(text=result.decode().strip("\"'"))
        )
