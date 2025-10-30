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
)


from abc import abstractmethod
from dapr.actor import Actor, ActorInterface, actormethod
from dapr.clients import DaprClient

from dqa import EnvVars
from dqa.actor import MHQAActorMethods
from dqa.actor.pubsub_topics import PubSubTopics
from dqa.model.mhqa import MCPToolInvocation, MHQAResponse, MHQAResponseStatus


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
    _chat_memory_key: ClassVar[str] = "chat_memory"

    def __init__(self, ctx, actor_id):
        super().__init__(ctx, actor_id)
        self._cancelled = False

    async def _on_activate(self) -> None:
        if not hasattr(self, "_wf_helper"):
            self._wf_helper = MHQAWorkflowHelper(
                agent_event_stream_handler=self.event_stream_handler
            )
        if not self._wf_helper.initialised:
            await self._wf_helper.initialise(str(self.id))

            if not self._wf_helper.initialised:
                raise ValueError("Agent workflow helper could not be initialised.")

            logger.info(
                f"MCP features available to {self.__class__.__name__} ({self.id}): {len(self._wf_helper.mcp_features)} tools"
            )

            saved_memory_messages = (
                await self._state_manager.get_state(self._chat_memory_key)
                if await self._state_manager.contains_state(self._chat_memory_key)
                else None
            )
            if saved_memory_messages and isinstance(saved_memory_messages, str):
                self._wf_helper.update_message_history_from_json(saved_memory_messages)
                logger.info("Restored message history from state store.")
        logger.info(f"{self.__class__.__name__} ({self.id}) activated")

    async def _on_deactivate(self) -> None:
        logger.info(f"{self.__class__.__name__} ({self.id}) deactivated")

    async def event_stream_handler(
        self,
        ctx: RunContext,
        event_stream: AsyncIterable[AgentStreamEvent],
    ):
        pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{self.id}"
        async for event in event_stream:
            with DaprClient() as dc:
                await self.handle_stream_event(event)
                response = MHQAResponse(
                    thread_id=str(self.id),
                    user_input="I forgot already!",
                    agent_output="",
                    tool_invocations=[],
                )
                dc.publish_event(
                    pubsub_name=EnvVars.DAPR_PUBSUB_NAME,
                    topic_name=pubsub_topic_name,
                    data=response.model_dump_json().encode(),
                )

    async def handle_stream_event(self, event: AgentStreamEvent):
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, ThinkingPart):
                print(f"\n[Thinking]\n{event.part.content}", flush=True, end="")
            else:
                print()
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                # This handler chooses to output deltas for the thinking part but not the text parts
                pass
            elif isinstance(event.delta, ThinkingPartDelta):
                print(f"{event.delta.content_delta}", flush=True, end="")
            elif isinstance(event.delta, ToolCallPartDelta):
                # We don't output deltas for tool calls
                pass
        elif isinstance(event, FunctionToolCallEvent):
            print(
                f"[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})"
            )
        elif isinstance(event, FunctionToolResultEvent):
            print(
                f"[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}"
            )
        elif isinstance(event, FinalResultEvent):
            # We don't output this through the event stream handler
            pass

    async def respond(self, data: dict) -> dict:
        user_input = data.get("user_input", "")
        if not user_input or user_input == "":
            raise ValueError("User input cannot be empty.")
        result = await self._wf_helper.run_workflow(user_message=user_input)
        full_response = result.output
        tool_invocations: List[MCPToolInvocation] = []
        pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{self.id}"
        with DaprClient() as dc:
            # current_agent = ""
            # async for event in wf_handler.stream_events():
            #     if (
            #         hasattr(event, "current_agent_name")
            #         and event.current_agent_name != current_agent
            #     ):
            #         current_agent = event.current_agent_name
            #         print(f"\n{'=' * 50}", flush=True)
            #         print(f"🤖 Agent: {current_agent}", flush=True)
            #         print(f"{'=' * 50}\n", flush=True)
            #     if isinstance(event, AgentStream):
            #         if event.delta:
            #             print(event.delta, end="", flush=True)
            #             full_response += event.delta
            #     elif isinstance(event, AgentInput):
            #         print(
            #             f"📥 {event.current_agent_name} Input:", event.input, flush=True
            #         )
            #     elif isinstance(event, ToolCall):
            #         print(f"🔨 Calling Tool: {event.tool_name}", flush=True)
            #         print(f"  With arguments: {event.tool_kwargs}", flush=True)
            #     elif isinstance(event, ToolCallResult):
            #         print(f"🔧 Tool Result ({event.tool_name}):", flush=True)
            #         print(f"  Arguments: {event.tool_kwargs}", flush=True)
            #         print(f"  Output: {event.tool_output}", flush=True)
            #         parsed_tool_output = (
            #             MHQAWorkflowOrchestrator.parse_tool_message_from_str(
            #                 event.tool_output.blocks[0].text
            #             )
            #             if type(event.tool_output) is ToolOutput
            #             else None
            #         )
            #         tool_invocations.append(
            #             MCPToolInvocation(
            #                 name=event.tool_name or event.tool_id,
            #                 input=json.dumps(event.tool_kwargs)
            #                 if type(event.tool_kwargs) is dict
            #                 else str(event.tool_kwargs),
            #                 output=parsed_tool_output.get("result", None)
            #                 if parsed_tool_output
            #                 else str(event.tool_output),
            #                 metadata=parsed_tool_output.get("content_meta", None)
            #                 if parsed_tool_output
            #                 else None,
            #             )
            #         )
            #     elif isinstance(event, AgentOutput):
            #         if event.response.content:
            #             print(
            #                 f"📤 {event.current_agent_name} Output:",
            #                 event.response.content,
            #                 flush=True,
            #             )
            #         if event.tool_calls:
            #             print(
            #                 "🛠️  Planning to use tools:",
            #                 [call.tool_name for call in event.tool_calls],
            #                 flush=True,
            #             )
            #     # elif isinstance(event, InputRequiredEvent):
            #     #     ic("Input required event encountered in MHQAActor.respond")
            #     #     ic(event)
            #     # elif isinstance(event, HumanResponseEvent):
            #     #     ic("Human response event encountered in MHQAActor.respond")
            #     #     ic(event)
            #     else:
            #         ic(type(event))
            #         ...

            #     response = MHQAResponse(
            #         thread_id=str(self.id),
            #         user_input=user_input,
            #         agent_output=full_response,
            #         tool_invocations=tool_invocations,
            #     )
            #     if (
            #         isinstance(event, AgentStream)
            #         and event.delta.strip() != ""
            #         and response
            #         and response.agent_output.strip() != ""
            #     ):
            #         # logger.info(f"Publishing: {response.agent_output}")
            #         dc.publish_event(
            #             pubsub_name=EnvVars.DAPR_PUBSUB_NAME,
            #             topic_name=pubsub_topic_name,
            #             data=response.model_dump_json().encode(),
            #         )

            response = MHQAResponse(
                thread_id=str(self.id),
                user_input=user_input,
                agent_output=full_response,
                tool_invocations=tool_invocations,
                status=MHQAResponseStatus.completed,
            )

            # response.status = MHQAResponseStatus.completed
            dc.publish_event(
                pubsub_name=EnvVars.DAPR_PUBSUB_NAME,
                topic_name=pubsub_topic_name,
                data=response.model_dump_json().encode(),
            )
        await self._state_manager.set_state(
            self._chat_memory_key,
            self._wf_helper.message_history_json,
        )
        await self._state_manager.save_state()
        return response.model_dump()

    async def get_chat_history(self) -> list:
        response: List[MHQAResponse] = []
        # if not self._cancelled:
        #     chat_messages = await self._wf_helper.workflow_memory.aget_all()
        #     for msg in chat_messages:
        #         # ic(msg.role, msg.content, type(msg.content), msg.additional_kwargs)
        #         # Is this list traversal in a consistent order?
        #         if msg.role == MessageRole.USER:
        #             user_input: str = msg.content
        #             tool_invocations: List[MCPToolInvocation] = []
        #         elif msg.role == MessageRole.ASSISTANT:
        #             if hasattr(msg, "content") and msg.content != "":
        #                 agent_output: str = msg.content
        #                 response.append(
        #                     MHQAResponse(
        #                         thread_id=str(self.id),
        #                         user_input=user_input,
        #                         agent_output=agent_output,
        #                         tool_invocations=tool_invocations,
        #                         status=MHQAResponseStatus.completed,
        #                     )
        #                 )
        #                 tool_name = ""
        #                 tool_input = ""
        #             else:
        #                 # Possible tool call but there could be many of these so why do we look at only the first one?
        #                 tool_calls = msg.additional_kwargs.get("tool_calls", [])
        #                 if len(tool_calls) > 0:
        #                     tool_function = tool_calls[0].get("function", {})
        #                     tool_name = tool_function.get("name", "")
        #                     tool_input = tool_function.get("arguments", None)
        #         elif msg.role == MessageRole.TOOL:
        #             parsed_tool_output = (
        #                 MHQAWorkflowOrchestrator.parse_tool_message_from_str(
        #                     msg.content
        #                 )
        #             )
        #             tool_invocations.append(
        #                 MCPToolInvocation(
        #                     name=tool_name,
        #                     input=json.dumps(tool_input)
        #                     if type(tool_input) is dict
        #                     else str(tool_input),
        #                     output=parsed_tool_output.get("result", None),
        #                     metadata=parsed_tool_output.get("content_meta", None),
        #                 )
        #             )
        #         else:
        #             ...
        return [r.model_dump() for r in response]

    async def reset_chat_history(self) -> bool:
        if not self._cancelled:
            self._wf_helper.update_message_history_from_json()
            if await self._state_manager.contains_state(self._chat_memory_key):
                await self._state_manager.remove_state(self._chat_memory_key)
                await self._state_manager.save_state()
                return True
            else:
                return False
        else:
            return False

    async def cancel(self) -> bool:
        if not self._cancelled:
            self._cancelled = True
            return True
        else:
            return False
