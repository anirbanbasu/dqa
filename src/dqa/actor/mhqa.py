import logging
from typing import AsyncIterable, ClassVar, List


from dqa.agent_workflow.mhqa_workflow import MHQAWorkflowHelper

from pydantic_ai import (
    AgentStreamEvent,
    FinalResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    PartDeltaEvent,
    PartStartEvent,
    RunContext,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPartDelta,
    ToolReturnPart,
)


from abc import abstractmethod
from dapr.actor import Actor, ActorInterface, actormethod
from dapr.clients import DaprClient

from dqa import EnvVars, ic
from dqa.actor import MHQAActorMethods
from dqa.actor.pubsub_topics import PubSubTopics
from dqa.model.mhqa import (
    MCPToolInvocation,
    MHQAResponse,
    MHQAResponseStatus,
    MHQAResponsesTypeAdapter,
)


logger = logging.getLogger(__name__)


class MHQAActorInterface(ActorInterface):
    @abstractmethod
    @actormethod(name=MHQAActorMethods.Respond)
    async def respond(self, data: dict) -> dict: ...

    @abstractmethod
    @actormethod(name=MHQAActorMethods.GetChatHistory)
    async def get_chat_history(self) -> list: ...

    @abstractmethod
    @actormethod(name=MHQAActorMethods.ResetChatHistory)
    async def reset_chat_history(self) -> bool: ...

    @abstractmethod
    @actormethod(name=MHQAActorMethods.Cancel)
    async def cancel(self) -> bool: ...


class MHQAActor(Actor, MHQAActorInterface):
    _internal_conversation_memory: ClassVar[str] = "internal_conversation_memory"
    _user_conversation_memory: ClassVar[str] = "user_conversation_memory"

    def __init__(self, ctx, actor_id):
        super().__init__(ctx, actor_id)
        self._cancelled = False

    async def _on_activate(self) -> None:
        if not hasattr(self, "_wf_helper"):
            self._wf_helper = MHQAWorkflowHelper(
                agent_event_stream_handler=self.event_stream_handler
            )
        if not self._wf_helper.initialised:
            raise ValueError("Agent workflow helper could not be initialised.")

        saved_internal_conversation_memory = (
            await self._state_manager.get_state(self._internal_conversation_memory)
            if await self._state_manager.contains_state(
                self._internal_conversation_memory
            )
            else None
        )
        if saved_internal_conversation_memory and isinstance(
            saved_internal_conversation_memory, str
        ):
            self._wf_helper.update_message_history_from_json(
                saved_internal_conversation_memory
            )
            logger.info("Restored internal conversation history from state store.")
            ic(self._wf_helper._message_history)

        saved_user_conversation_memory = (
            await self._state_manager.get_state(self._user_conversation_memory)
            if await self._state_manager.contains_state(self._user_conversation_memory)
            else None
        )
        self.user_conversation_history: List[MHQAResponse] = []
        if saved_user_conversation_memory and isinstance(
            saved_user_conversation_memory, str
        ):
            self.user_conversation_history = MHQAResponsesTypeAdapter.validate_json(
                saved_user_conversation_memory
            )
            logger.info("Restored user conversation history from state store.")

        logger.info(f"{self.__class__.__name__} ({self.id}) activated")

    async def _on_deactivate(self) -> None:
        logger.info(f"{self.__class__.__name__} ({self.id}) deactivated")

    async def event_stream_handler(
        self,
        ctx: RunContext,
        event_stream: AsyncIterable[AgentStreamEvent],
    ):
        pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{self.id}"
        agent_output_message: str | None = None
        current_tool_invocation: MCPToolInvocation | None = None
        with DaprClient() as dc:
            async for event in event_stream:
                if isinstance(event, PartStartEvent):
                    if isinstance(event.part, ThinkingPart):
                        agent_output_message = f"\n[Thinking]\n{event.part.content}"
                    else:
                        pass
                elif isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, TextPartDelta):
                        # This handler chooses to output deltas for the thinking part but not the text parts
                        pass
                    elif isinstance(event.delta, ThinkingPartDelta):
                        agent_output_message += event.delta.content_delta
                    elif isinstance(event.delta, ToolCallPartDelta):
                        # We don't output deltas for tool calls
                        pass
                elif isinstance(event, FunctionToolCallEvent):
                    current_tool_invocation = MCPToolInvocation(
                        name=event.part.tool_name,
                        input=str(event.part.args),
                        tool_call_id=event.part.tool_call_id,
                    )
                elif isinstance(event, FunctionToolResultEvent):
                    if (
                        event.result
                        and current_tool_invocation
                        and isinstance(event.result, ToolReturnPart)
                    ):
                        if current_tool_invocation.tool_call_id == event.tool_call_id:
                            current_tool_invocation.output = (
                                str(event.result.content)
                                if hasattr(event.result, "content")
                                else None
                            )
                            current_tool_invocation.metadata = (
                                str(event.result.metadata)
                                if hasattr(event.result, "metadata")
                                else None
                            )
                            self._current_run_tool_invocations.append(
                                current_tool_invocation
                            )
                elif isinstance(event, FinalResultEvent):
                    # We don't output this through the event stream handler
                    pass
                response = MHQAResponse(
                    thread_id=str(self.id),
                    user_input=self._wf_helper._initial_run_state.user_message,
                    agent_output=agent_output_message,
                    tool_invocations=self._current_run_tool_invocations,
                )
                dc.publish_event(
                    pubsub_name=EnvVars.DAPR_PUBSUB_NAME,
                    topic_name=pubsub_topic_name,
                    data=response.model_dump_json().encode(),
                )

    async def respond(self, data: dict) -> dict:
        user_input = data.get("user_input", "")
        if not user_input or user_input == "":
            raise ValueError("User input cannot be empty.")
        self._current_run_tool_invocations: List[MCPToolInvocation] = []
        result = await self._wf_helper.run_workflow(user_message=user_input)
        pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{self.id}"
        with DaprClient() as dc:
            response = MHQAResponse(
                thread_id=str(self.id),
                user_input=user_input,
                agent_output=result.output,
                tool_invocations=self._current_run_tool_invocations,
                status=MHQAResponseStatus.completed,
            )
            dc.publish_event(
                pubsub_name=EnvVars.DAPR_PUBSUB_NAME,
                topic_name=pubsub_topic_name,
                data=response.model_dump_json().encode(),
            )
        await self._state_manager.set_state(
            self._internal_conversation_memory,
            self._wf_helper._message_history_json,
        )
        self.user_conversation_history.append(response)
        await self._state_manager.set_state(
            self._user_conversation_memory,
            # FIXME: Should we use to_jsonable_python and then json.dumps?
            MHQAResponsesTypeAdapter.dump_json(self.user_conversation_history).decode(),
        )
        await self._state_manager.save_state()
        return response.model_dump()

    async def get_chat_history(self) -> list:
        response: List[MHQAResponse] = []
        if not self._cancelled:
            response = self.user_conversation_history
        return [r.model_dump() for r in response]

    async def reset_chat_history(self) -> bool:
        result = False
        if not self._cancelled:
            self._wf_helper.update_message_history_from_json()
            self.user_conversation_history = []
            if await self._state_manager.contains_state(
                self._internal_conversation_memory
            ):
                await self._state_manager.remove_state(
                    self._internal_conversation_memory
                )
                result = True

            if await self._state_manager.contains_state(self._user_conversation_memory):
                await self._state_manager.remove_state(self._user_conversation_memory)
                result = True

            if result:
                await self._state_manager.save_state()
        return result

    async def cancel(self) -> bool:
        if not self._cancelled:
            self._cancelled = True
            return True
        else:
            return False
