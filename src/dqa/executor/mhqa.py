import asyncio
import json
import logging
from dapr.actor import ActorProxy, ActorId, ActorProxyFactory
from dapr.clients.retry import RetryPolicy
from dapr.common.pubsub.subscription import SubscriptionMessage
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message, new_task
from a2a.types import TaskState

from dapr.clients.grpc._response import TopicEventResponse, TopicEventResponseStatus

from dqa import ParsedEnvVars, ic
from dqa.actor.mhqa import MHQAActor, MHQAActorInterface, MHQAActorMethods
from dqa.actor.pubsub_topics import PubSubTopics
from dqa.model.mhqa import (
    MHQAAgentSkills,
    MHQADeleteHistoryInput,
    MHQAHistoryInput,
    MHQAInput,
    MHQAAgentInputMessage,
    MHQAResponse,
)


from dapr.clients import DaprClient

logger = logging.getLogger(__name__)


class MHQAAgentExecutor(AgentExecutor):
    def __init__(self):
        self._actor_mhqa = MHQAActor.__name__
        self._factory = ActorProxyFactory(retry_policy=RetryPolicy(max_attempts=1))

    async def do_mhqa_respond(self, data: MHQAInput):
        def message_handler(message: SubscriptionMessage):
            try:
                logger.debug(
                    f"Received message: {message.data()} at {message.pubsub_name()}:{message.topic()}"
                )
                result = (
                    MHQAResponse.model_validate_json(message.data())
                    if type(message.data()) is str
                    else MHQAResponse.model_validate(message.data())
                )
                logger.info(result.agent_output)
                return TopicEventResponse(TopicEventResponseStatus.success)

            except Exception as e:
                logger.error(f"Error processing message: {e}")
                return TopicEventResponse(TopicEventResponseStatus.retry)

        proxy = ActorProxy.create(
            actor_type=self._actor_mhqa,
            actor_id=ActorId(actor_id=data.thread_id),
            actor_interface=MHQAActorInterface,
            actor_proxy_factory=self._factory,
        )

        pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{data.thread_id}"
        with DaprClient() as dc:
            # close_fn = dc.subscribe_with_handler(
            #     pubsub_name=ParsedEnvVars().DAPR_PUBSUB_NAME,
            #     topic=pubsub_topic_name,
            #     handler_fn=message_handler,
            # )
            # result = await proxy.invoke_method(
            #     method=MHQAActorMethods.Respond,
            #     raw_body=data.model_dump_json().encode(),
            # )
            # close_fn()  # Unsubscribe from the topic
            subscription = dc.subscribe(
                pubsub_name=ParsedEnvVars().DAPR_PUBSUB_NAME, topic=pubsub_topic_name
            )
            task = asyncio.create_task(
                proxy.invoke_method(
                    method=MHQAActorMethods.Respond,
                    raw_body=data.model_dump_json().encode(),
                )
            )
            for msg in subscription:
                result = MHQAResponse.model_validate_json(msg.data())
                yield json.dumps(result.model_dump())
                if not subscription._is_stream_active() or task.done():
                    break
            subscription.close()

        # return result.decode().strip("\"'")

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

        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        response = None
        match message_payload.skill:
            case MHQAAgentSkills.Respond:
                async for partial_response in self.do_mhqa_respond(
                    data=message_payload.data
                ):
                    # logger.info(f"Partial response: {partial_response}")
                    ic(partial_response)
                    response = partial_response
                    await task_updater.update_status(
                        TaskState.working, new_agent_text_message(text=response)
                    )
            case MHQAAgentSkills.GetChatHistory:
                response = await self.do_mhqa_get_history(data=message_payload.data)
            case MHQAAgentSkills.ResetChatHistory:
                response = await self.do_mhqa_delete_history(data=message_payload.data)
            case _:
                raise ValueError(f"Unknown skill '{message_payload.skill}' requested!")
        if response:
            # if message_payload.skill != MHQAAgentSkills.Respond:
            #     await event_queue.enqueue_event(new_agent_text_message(text=response))
            # else:
            await task_updater.update_status(
                TaskState.completed, message=new_agent_text_message(text=response)
            )
            # await task_updater.complete(new_agent_text_message(text=response))
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
