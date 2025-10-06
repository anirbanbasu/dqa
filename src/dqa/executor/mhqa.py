from dapr.actor import ActorProxy, ActorId, ActorProxyFactory
from dapr.clients.retry import RetryPolicy
from dapr.common.pubsub.subscription import SubscriptionMessage
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.utils import new_agent_text_message

from dapr.clients.grpc._response import TopicEventResponse

from dqa import ParsedEnvVars
from dqa.actor.mhqa import MHQAActor, MHQAActorInterface, MHQAActorMethods
from dqa.model.mhqa import (
    MHQAAgentSkills,
    MHQADeleteHistoryInput,
    MHQAHistoryInput,
    MHQAInput,
    MHQAAgentInputMessage,
)

from dqa import ic

from dapr.clients import DaprClient


class MHQAAgentExecutor(AgentExecutor):
    def __init__(self):
        self._actor_mhqa = MHQAActor.__name__
        self._factory = ActorProxyFactory(retry_policy=RetryPolicy(max_attempts=1))

    def message_handler(self, message: SubscriptionMessage) -> TopicEventResponse:
        """Callback function to handle incoming messages."""
        ic(message)
        ic(message.__dict__)
        ic(message.data(), message.topic())
        # Return success to acknowledge the message
        return TopicEventResponse("SUCCESS")

    async def do_mhqa_respond(self, data: MHQAInput):
        proxy = ActorProxy.create(
            actor_type=self._actor_mhqa,
            actor_id=ActorId(actor_id=data.thread_id),
            actor_interface=MHQAActorInterface,
            actor_proxy_factory=self._factory,
        )
        # asyncio.create_task(
        #     proxy.invoke_method(
        #         method=MHQAActorMethods.Respond,
        #         raw_body=data.model_dump_json().encode(),
        #     )
        # )

        with DaprClient() as dc:
            ic(f"topic-{self._actor_mhqa}-{data.thread_id}-respond")
            # close_fn =
            dc.subscribe_with_handler(
                pubsub_name=ParsedEnvVars().DAPR_PUBSUB_NAME,
                topic=f"topic-{self._actor_mhqa}-{data.thread_id}-respond",
                handler_fn=self.message_handler,
            )
        result = await proxy.invoke_method(
            method=MHQAActorMethods.Respond,
            raw_body=data.model_dump_json().encode(),
        )
        # close_fn()  # Unsubscribe from the topic

        return result.decode().strip("\"'")

    async def do_mhqa_get_history(self, data: MHQAHistoryInput) -> str:
        proxy = ActorProxy.create(
            actor_type=self._actor_mhqa,
            actor_id=ActorId(actor_id=data.thread_id),
            actor_interface=MHQAActorInterface,
            actor_proxy_factory=self._factory,
        )
        result = await proxy.invoke_method(method=MHQAActorMethods.GetChatHistory)
        return result.decode().strip("\"'")

    async def do_mhqa_delete_history(self, data: MHQADeleteHistoryInput) -> str:
        proxy = ActorProxy.create(
            actor_type=self._actor_mhqa,
            actor_id=ActorId(actor_id=data.thread_id),
            actor_interface=MHQAActorInterface,
            actor_proxy_factory=self._factory,
        )
        result = await proxy.invoke_method(
            method=MHQAActorMethods.ResetChatHistory,
        )
        return result.decode().strip("\"'")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        message_payload = MHQAAgentInputMessage.model_validate_json(
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
            case MHQAAgentSkills.Respond:
                response = await self.do_mhqa_respond(data=message_payload.data)
            case MHQAAgentSkills.GetChatHistory:
                response = await self.do_mhqa_get_history(data=message_payload.data)
            case MHQAAgentSkills.ResetChatHistory:
                response = await self.do_mhqa_delete_history(data=message_payload.data)
            case _:
                raise ValueError(f"Unknown skill '{message_payload.skill}' requested!")
        if response:
            await event_queue.enqueue_event(new_agent_text_message(text=response))
        else:
            raise ValueError("No response received from the actor(s)!")

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        message_payload = MHQAAgentInputMessage.model_validate_json(
            context.get_user_input()
        )
        if (
            not message_payload
            or not message_payload.data
            or message_payload.data.thread_id.strip() == ""
        ):
            raise ValueError(("Missing mandatory thread_id in the input!"))

        proxy = ActorProxy.create(
            actor_type=self._actor_mhqa,
            actor_id=ActorId(actor_id=message_payload.data.thread_id),
            actor_interface=MHQAActorInterface,
            actor_proxy_factory=self._factory,
        )
        result = await proxy.invoke_method(method=MHQAActorMethods.Cancel)
        await event_queue.enqueue_event(
            new_agent_text_message(text=result.decode().strip("\"'"))
        )
