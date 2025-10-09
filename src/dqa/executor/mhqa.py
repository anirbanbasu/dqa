import datetime
import logging
import math
import anyio
from dapr.actor import ActorProxy, ActorId, ActorProxyFactory
from dapr.clients.retry import RetryPolicy
from dapr.clients.grpc.subscription import SubscriptionMessage, TopicEventResponse
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message, new_task
from a2a.types import TaskState


from dqa import ParsedEnvVars
from dqa.actor.mhqa import MHQAActor, MHQAActorInterface, MHQAActorMethods
from dqa.actor.pubsub_topics import PubSubTopics
from dqa.model.mhqa import (
    MHQAAgentSkills,
    MHQADeleteHistoryInput,
    MHQAHistoryInput,
    MHQAInput,
    MHQAAgentInputMessage,
    MHQAResponse,
    MHQAResponseStatus,
)


from dapr.clients import DaprClient

logger = logging.getLogger(__name__)


class MHQAAgentExecutor(AgentExecutor):
    def __init__(self):
        self._actor_mhqa = MHQAActor.__name__
        self._factory = ActorProxyFactory(retry_policy=RetryPolicy(max_attempts=1))

    async def do_mhqa_respond(self, data: MHQAInput):
        # TODO: Potential memory leak without closing the streams?
        send_stream, recv_stream = anyio.create_memory_object_stream(math.inf)

        def message_handler(message: SubscriptionMessage) -> TopicEventResponse:
            # TODO: Is this a reasonable way to drop stale messages?
            parsed_timestamp = message.extensions().get("time", None)
            if parsed_timestamp is not None:
                timenow = datetime.datetime.now(datetime.timezone.utc)
                timestamp = datetime.datetime.fromisoformat(parsed_timestamp)
                td = timenow - timestamp
                if td > datetime.timedelta(
                    seconds=ParsedEnvVars().APP_DAPR_PUBSUB_STALE_MSG_SECS
                ):
                    logger.warning(
                        f"Dropping stale message for topic={message.topic()} with age {td} seconds"
                    )
                    return TopicEventResponse("drop")
            send_stream.send_nowait(message.data())
            return TopicEventResponse("success")

        async def invoke_actor():
            proxy = ActorProxy.create(
                actor_type=self._actor_mhqa,
                actor_id=ActorId(actor_id=data.thread_id),
                actor_interface=MHQAActorInterface,
                actor_proxy_factory=self._factory,
            )
            return await proxy.invoke_method(
                method=MHQAActorMethods.Respond,
                raw_body=data.model_dump_json().encode(),
            )

        with DaprClient() as dc:
            async with anyio.create_task_group() as tg:
                pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{data.thread_id}"
                dc.subscribe_with_handler(
                    pubsub_name=ParsedEnvVars().DAPR_PUBSUB_NAME,
                    topic=pubsub_topic_name,
                    handler_fn=message_handler,
                )
                tg.start_soon(invoke_actor)

            async for item in recv_stream:
                yield item

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
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
            if task.status.state not in [
                TaskState.submitted,
                TaskState.completed,
                TaskState.failed,
                TaskState.canceled,
                TaskState.rejected,
            ]:
                raise ValueError(
                    f"Task {task.id} is in incomplete state '{task.status}'. Marking it as failed."
                )
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
                    response_generator = self.do_mhqa_respond(data=message_payload.data)
                    async for partial_response in response_generator:
                        response = partial_response
                        parsed_response = MHQAResponse.model_validate_json(response)
                        if parsed_response.status == MHQAResponseStatus.completed:
                            break
                        await task_updater.start_work(
                            new_agent_text_message(
                                text=response,
                                task_id=task.id,
                                context_id=task.context_id,
                            )
                        )
                case MHQAAgentSkills.GetChatHistory:
                    response = await self.do_mhqa_get_history(data=message_payload.data)
                case MHQAAgentSkills.ResetChatHistory:
                    response = await self.do_mhqa_delete_history(
                        data=message_payload.data
                    )
                case _:
                    raise ValueError(
                        f"Unknown skill '{message_payload.skill}' requested!"
                    )
            if response:
                await task_updater.complete(
                    new_agent_text_message(
                        text=response,
                        task_id=task.id,
                        context_id=task.context_id,
                    )
                )
            else:
                raise ValueError("No response received from the actor(s)!")
        except Exception as e:
            logger.error(f"Error in MHQAAgentExecutor. {e}")
            await task_updater.failed(
                message=new_agent_text_message(
                    # FIXME: The output will fail JSON validation in the client side
                    # because it is not of type MHQAResponse
                    text=str(e),
                    task_id=task.id,
                    context_id=task.context_id,
                )
            )

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)
        task_updater = TaskUpdater(event_queue, task.id, task.context_id)
        try:
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
            await task_updater.complete(
                new_agent_text_message(
                    text=result.decode().strip("\"'"),
                    task_id=task.id,
                    context_id=task.context_id,
                )
            )
        except Exception as e:
            logger.error(f"Error in MHQAAgentExecutor cancel. {e}")
            await task_updater.failed(
                message=new_agent_text_message(
                    text=str(e),
                    task_id=task.id,
                    context_id=task.context_id,
                )
            )
