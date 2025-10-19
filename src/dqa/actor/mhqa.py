import json
import logging
from typing import List


from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools.types import ToolOutput


from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)


from abc import abstractmethod
from dapr.actor import Actor, ActorInterface, actormethod
from dapr.clients import DaprClient
from pydantic import TypeAdapter

from dqa import EnvVars
from dqa.actor import MHQAActorMethods
from dqa.actor.pubsub_topics import PubSubTopics
from dqa.agent_workflow.llamaindex_agentworkflow import MHQAAgentWorkflowOrchestrator
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
    _chat_memory_key = "chat_memory"
    _memory_messages_type_adapter = TypeAdapter(List[ChatMessage])
    _wf_orchestrator: MHQAAgentWorkflowOrchestrator = MHQAAgentWorkflowOrchestrator()

    def __init__(self, ctx, actor_id):
        super().__init__(ctx, actor_id)
        self._cancelled = False

    async def _on_activate(self) -> None:
        if not self._wf_orchestrator.initialised:
            await self._wf_orchestrator.initialise(str(self.id))

            if not self._wf_orchestrator.initialised:
                raise ValueError(
                    "Agent workflow orchestrator could not be initialised."
                )

            logger.info(
                f"MCP features available to {self.__class__.__name__} ({self.id}): {len(self._wf_orchestrator.mcp_features)} tools"
            )

            saved_memory_messages = (
                await self._state_manager.get_state(self._chat_memory_key)
                if await self._state_manager.contains_state(self._chat_memory_key)
                else None
            )
            if saved_memory_messages and isinstance(saved_memory_messages, str):
                parsed_messages = self._memory_messages_type_adapter.validate_json(
                    saved_memory_messages
                )
                for msg in parsed_messages:
                    self._wf_orchestrator.workflow_memory.put(msg)
                logger.info(
                    f"Restored {len(parsed_messages)} messages from state store"
                )
        logger.info(f"{self.__class__.__name__} ({self.id}) activated")

    async def _on_deactivate(self) -> None:
        logger.info(f"{self.__class__.__name__} ({self.id}) deactivated")

    async def respond(self, data: dict) -> dict:
        user_input = data.get("user_input", "")
        wf_handler = self._wf_orchestrator.workflow.run(
            user_msg=user_input,
            memory=self._wf_orchestrator.workflow_memory,
            context=self._wf_orchestrator.workflow_context,
            max_iterations=5,
        )
        full_response = ""
        tool_invocations: List[MCPToolInvocation] = []
        pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{self.id}"
        with DaprClient() as dc:
            current_agent = ""
            async for event in wf_handler.stream_events(expose_internal=True):
                if (
                    hasattr(event, "current_agent_name")
                    and event.current_agent_name != current_agent
                ):
                    current_agent = event.current_agent_name
                    print(f"\n{'=' * 50}")
                    print(f"🤖 Agent: {current_agent}")
                    print(f"{'=' * 50}\n")
                if isinstance(event, AgentStream):
                    full_response += event.delta
                elif isinstance(event, ToolCall):
                    print(f"🔨 Calling Tool: {event.tool_name}")
                    print(f"  With arguments: {event.tool_kwargs}")
                elif isinstance(event, ToolCallResult):
                    print(f"🔧 Tool Result ({event.tool_name}):")
                    print(f"  Arguments: {event.tool_kwargs}")
                    print(f"  Output: {event.tool_output}")
                    parsed_tool_output = (
                        MHQAAgentWorkflowOrchestrator.parse_tool_message_from_str(
                            event.tool_output.blocks[0].text
                        )
                        if type(event.tool_output) is ToolOutput
                        else None
                    )
                    tool_invocations.append(
                        MCPToolInvocation(
                            name=event.tool_name or event.tool_id,
                            input=json.dumps(event.tool_kwargs)
                            if type(event.tool_kwargs) is dict
                            else str(event.tool_kwargs),
                            output=parsed_tool_output.get("result", None)
                            if parsed_tool_output
                            else str(event.tool_output),
                            metadata=parsed_tool_output.get("content_meta", None)
                            if parsed_tool_output
                            else None,
                        )
                    )
                elif isinstance(event, AgentOutput):
                    if event.response.content:
                        print("📤 Output:", event.response.content)
                    if event.tool_calls:
                        print(
                            "🛠️  Planning to use tools:",
                            [call.tool_name for call in event.tool_calls],
                        )
                # elif isinstance(ev, InputRequiredEvent):
                #     ic("Input required event encountered in MHQAActor.respond")
                #     ic(ev)
                else:
                    ...

                response = MHQAResponse(
                    thread_id=str(self.id),
                    user_input=user_input,
                    agent_output=full_response,
                    tool_invocations=tool_invocations,
                )
                if (
                    isinstance(event, AgentStream)
                    and event.delta.strip() != ""
                    and response
                    and response.agent_output.strip() != ""
                ):
                    # logger.info(f"Publishing: {response.agent_output}")
                    dc.publish_event(
                        pubsub_name=EnvVars.DAPR_PUBSUB_NAME,
                        topic_name=pubsub_topic_name,
                        data=response.model_dump_json().encode(),
                    )
            response.status = MHQAResponseStatus.completed
            dc.publish_event(
                pubsub_name=EnvVars.DAPR_PUBSUB_NAME,
                topic_name=pubsub_topic_name,
                data=response.model_dump_json().encode(),
            )
        memory_messages = await self._wf_orchestrator.workflow_memory.aget_all()
        await self._state_manager.set_state(
            self._chat_memory_key,
            # Be careful with the dump_json method because it returns bytes
            self._memory_messages_type_adapter.dump_json(memory_messages).decode(),
        )
        await self._state_manager.save_state()
        return response.model_dump()

    async def get_chat_history(self) -> list:
        response: List[MHQAResponse] = []
        if not self._cancelled:
            chat_messages = await self._wf_orchestrator.workflow_memory.aget_all()
            for msg in chat_messages:
                # ic(msg.role, msg.content, type(msg.content), msg.additional_kwargs)
                # Is this list traversal in a consistent order?
                if msg.role == MessageRole.USER:
                    user_input: str = msg.content
                    tool_invocations: List[MCPToolInvocation] = []
                elif msg.role == MessageRole.ASSISTANT:
                    if hasattr(msg, "content") and msg.content != "":
                        agent_output: str = msg.content
                        response.append(
                            MHQAResponse(
                                thread_id=str(self.id),
                                user_input=user_input,
                                agent_output=agent_output,
                                tool_invocations=tool_invocations,
                                status=MHQAResponseStatus.completed,
                            )
                        )
                        tool_name = ""
                        tool_input = ""
                    else:
                        # Possible tool call but there could be many of these so why do we look at only the first one?
                        tool_calls = msg.additional_kwargs.get("tool_calls", [])
                        if len(tool_calls) > 0:
                            tool_function = tool_calls[0].get("function", {})
                            tool_name = tool_function.get("name", "")
                            tool_input = tool_function.get("arguments", None)
                elif msg.role == MessageRole.TOOL:
                    parsed_tool_output = (
                        MHQAAgentWorkflowOrchestrator.parse_tool_message_from_str(
                            msg.content
                        )
                    )
                    tool_invocations.append(
                        MCPToolInvocation(
                            name=tool_name,
                            input=json.dumps(tool_input)
                            if type(tool_input) is dict
                            else str(tool_input),
                            output=parsed_tool_output.get("result", None),
                            metadata=parsed_tool_output.get("content_meta", None),
                        )
                    )
                else:
                    ...
        return [r.model_dump() for r in response]

    async def reset_chat_history(self) -> bool:
        if not self._cancelled:
            await self._wf_orchestrator.workflow_memory.areset()
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
