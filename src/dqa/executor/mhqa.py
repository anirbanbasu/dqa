import logging
import anyio
from dapr.actor import ActorProxy, ActorId, ActorProxyFactory
from dapr.clients.retry import RetryPolicy
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.utils import new_agent_text_message, new_task
from a2a.types import TaskState


from dqa import ParsedEnvVars, ic
from dqa.actor.mhqa import MHQAActor, MHQAActorInterface, MHQAActorMethods
from dqa.actor.pubsub_topics import PubSubTopics
from dqa.model.mhqa import (
    MHQAAgentSkills,
    MHQADeleteHistoryInput,
    MHQAHistoryInput,
    MHQAInput,
    MHQAAgentInputMessage,
)


from dapr.clients import DaprClient

logger = logging.getLogger(__name__)


class MHQAAgentExecutor(AgentExecutor):
    def __init__(self):
        self._actor_mhqa = MHQAActor.__name__
        self._factory = ActorProxyFactory(retry_policy=RetryPolicy(max_attempts=1))

    async def do_mhqa_respond(self, data: MHQAInput):
        # def message_handler(message: SubscriptionMessage) -> TopicEventResponse:
        #     try:
        #         logger.debug(
        #             f"Received message: {message.data()} at {message.pubsub_name()}:{message.topic()}"
        #         )
        #         result = (
        #             MHQAResponse.model_validate_json(message.data())
        #             if type(message.data()) is str
        #             else MHQAResponse.model_validate(message.data())
        #         )
        #         logger.info(result.agent_output)
        #         return TopicEventResponse(TopicEventResponseStatus.success)

        #     except Exception as e:
        #         logger.error(f"Error processing message: {e}")
        #         return TopicEventResponse(TopicEventResponseStatus.retry)

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
            async with anyio.create_task_group() as tg:
                pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{data.thread_id}"
                subscription = dc.subscribe(
                    pubsub_name=ParsedEnvVars().DAPR_PUBSUB_NAME,
                    topic=pubsub_topic_name,
                )
                tg.start_soon(invoke_actor)
            while True:
                # FIXME: How do we discard messages not relevant to this request?
                msg = (
                    subscription.next_message()
                    if subscription._is_stream_active()
                    else None
                )
                if msg is None:
                    break
                result = msg.data()
                subscription.respond_success(msg)
                await anyio.sleep(0.01)
                yield result
            ic("Exited subscription loop")
            subscription.close()
            # yield result

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
                        ic(f"Parsed from generator: {response}")
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
                ic(f"Parsed from generator FINAL: {response}")
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
