from enum import StrEnum
import json
import logging
import re
from typing import Any, Dict, List

from llama_index.tools.mcp import aget_tools_from_mcp_url, BasicMCPClient

from llama_index.core.memory import Memory
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from llama_index.core.tools.types import ToolOutput

from dqa import ic


from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)

from llama_index.core.workflow.events import InputRequiredEvent, HumanResponseEvent

from abc import abstractmethod
import os
from dapr.actor import Actor, ActorInterface, actormethod
from dapr.clients import DaprClient
from pydantic import TypeAdapter

from dqa import ParsedEnvVars
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


class MHQAAgentNames(StrEnum):
    DECOMPOSER = "Decomposer"
    RESPONDER = "Responder"
    REASONER = "Reasoner"
    REVIEWER = "Reviewer"


class MHQAActor(Actor, MHQAActorInterface):
    _chat_memory_key = "chat_memory"
    _memory_messages_type_adapter = TypeAdapter(List[ChatMessage])

    @staticmethod
    async def obtain_human_check_for_plan(ctx: Context, plan: str) -> str:
        """
        Obtain human approval or modification for the given plan. Alternatively, the human may provide a modified plan.
        """
        ic(ctx, plan)
        question = f"Does you approve the following plan? (You may also provide me with a modified plan.)\n{plan}"
        human_response = await ctx.wait_for_event(
            HumanResponseEvent,
            waiter_id=question,
            waiter_event=InputRequiredEvent(
                prefix=question,
                user_name="Reviewer",  # you may track a specific user
            ),
            requirements={"user_name": "Reviewer"},
        )
        return human_response.response()

    @staticmethod
    def parse_tool_message_from_str(msg: str) -> Dict[str, Any]:
        """
        Parse a message string representing an MCP tool call, extracting:
        - is_error (bool)
        - tool result (parsed JSON or raw text)
        - content-level metadata (dict or None)
        - outer-level meta (raw text or None)
        """
        # 1. Extract isError part
        m_err = re.search(r"\bisError\s*=\s*(True|False)", msg)
        is_error = None
        if m_err:
            is_error = m_err.group(1) == "True"
        else:
            # fallback default or raise
            is_error = False

        # 2. Extract the content block, i.e. the TextContent(...) part
        # We look for content=\[TextContent( ... )\]
        # This is a bit fragile but should work for typical formatting
        m_content = re.search(r"content=\[TextContent\((.*?)\)\]", msg, re.DOTALL)
        tool_result = None
        content_meta = None
        if m_content:
            inner = m_content.group(1)
            # inner is something like
            # "type='text', text='...json...', annotations=None, meta={...}"
            # Extract the text='...'
            m_text = re.search(r"text='(.*?)'", inner, re.DOTALL)
            if m_text:
                tool_result = m_text.group(1)
            # Extract the meta={...} inside
            m_cmeta = re.search(r"meta=\{(.*)\}\s*(?:,|$)", inner, re.DOTALL)
            if m_cmeta:
                meta_body = m_cmeta.group(1)
                # meta_body is something like "'frankfurtermcp': {'version': '0.3.6', ... }"
                # We can wrap braces and convert quotes to valid JSON-like string
                meta_text = "{" + meta_body + "}"
                # But Python single quotes make it invalid JSON. Replace single quotes with double quotes.
                # This is approximate and may break for nested cases; for more robust solution use an AST parser.
                content_meta = meta_text.replace("'", '"')

        # 3. Extract outer-level meta=... before content=...
        m_outer = re.search(r"\bmeta\s*=\s*(None|\{.*?\})\s+content=", msg, re.DOTALL)
        outer_meta = None
        if m_outer:
            outer = m_outer.group(1)
            if outer == "None":
                outer_meta = None
            else:
                # similar parse as for content_meta
                outer_meta = outer.strip()

        return {
            "is_error": is_error,
            "result": tool_result,
            "content_meta": content_meta,
            "outer_meta": outer_meta,
        }

    def __init__(self, ctx, actor_id):
        super().__init__(ctx, actor_id)
        self._cancelled = False

    async def _on_activate(self) -> None:
        if not hasattr(self, "llm_config"):
            self.llm_config = {}
            llm_config_file = ParsedEnvVars().LLM_CONFIG_FILE
            if os.path.exists(llm_config_file):
                with open(llm_config_file, "r") as f:
                    self.llm_config = json.load(f)

        if not hasattr(self, "mcp_config") or not hasattr(self, "mcp_features"):
            self.mcp_features = []
            self.mcp_config = {}
            try:
                mcp_config_file = ParsedEnvVars().MCP_CONFIG_FILE
                if os.path.exists(mcp_config_file):
                    with open(mcp_config_file, "r") as f:
                        self.mcp_config = json.load(f)
                for _, config in self.mcp_config.items():
                    mcp_client = BasicMCPClient(
                        command_or_url=(
                            config.get("url", "")
                            if config.get("transport", "") != "stdio"
                            else config.get("command", "")
                        ),
                        args=config.get("args", []),
                        env=config.get("env", {}),
                        timeout=config.get("timeout", 30),
                    )
                    mcp_features = await aget_tools_from_mcp_url(
                        command_or_url=None,
                        client=mcp_client,
                    )
                    self.mcp_features.extend(mcp_features)
            except Exception as e:
                logger.error(f"Error parsing MCP config. {e}")
                logger.exception(e)

        logger.info(
            f"MCP features available to {self.__class__.__name__} ({self.id}): {len(self.mcp_features)}"
        )

        if not hasattr(self, "workflow"):
            decomposer_agent = FunctionAgent(
                name=MHQAAgentNames.DECOMPOSER,
                description="Decomposes complex questions into simpler sub-questions.",
                system_prompt=(
                    f"You are the {MHQAAgentNames.DECOMPOSER} agent. "
                    "Determine if the user query is a question or a statement.\n"
                    "If the user query is a question and it has no direct answer, decompose it into smaller sub-questions. "
                    "If the user query is a simple question then decompose it into just one single sub-question, which is the question posed by the user."
                    "If the user input is only a statement but not a question then respond with an acknowledgment only.\n"
                    "Store in state the decomposed sub-questions in a list format. "
                    f"If the user query is not a statement, hand off to the {MHQAAgentNames.RESPONDER} agent.\n"
                ),
                tools=[],
                can_handoff_to=[MHQAAgentNames.RESPONDER],
                llm=Ollama(**self.llm_config[MHQAAgentNames.DECOMPOSER.lower()]),
            )

            responder_agent = FunctionAgent(
                name=MHQAAgentNames.RESPONDER,
                description="Responds to sub-questions using available tools.",
                system_prompt=(
                    f"You are the {MHQAAgentNames.RESPONDER} agent.\n"
                    f"You would be provided with a list of questions by the {MHQAAgentNames.DECOMPOSER} agent. "
                    "Call the appropriate tools for each question given to you and output a list of tool call responses. "
                    "Store in state the responses you have gathered for each question. "
                    f"Once finished with all the questions, hand off to the {MHQAAgentNames.REASONER} agent.\n"
                ),
                tools=self.mcp_features,
                can_handoff_to=[MHQAAgentNames.REASONER],
                llm=Ollama(**self.llm_config[MHQAAgentNames.RESPONDER.lower()]),
            )

            reasoner_agent = ReActAgent(
                name=MHQAAgentNames.REASONER,
                description="Reasons through gathered evidences to formulate a combined response.",
                system_prompt=(
                    f"You are the {MHQAAgentNames.REASONER} agent.\n"
                    f"You would receive from the {MHQAAgentNames.RESPONDER} agent a list of questions and the evidences it gathered for each question. "
                    "Reason through the evidences for the individual questions and combine them into a single response that serves as a coherent answer to the original user query.\n"
                    f"Hand off to the {MHQAAgentNames.REVIEWER} agent for a review of your combined response.\n"
                ),
                tools=[],
                can_handoff_to=[MHQAAgentNames.REVIEWER],
                llm=Ollama(**self.llm_config[MHQAAgentNames.REASONER.lower()]),
            )

            reviewer_agent = ReActAgent(
                name=MHQAAgentNames.REVIEWER,
                description="Reviews the combined response for quality and completeness.",
                system_prompt=(
                    f"You are the {MHQAAgentNames.REVIEWER} agent.\n"
                    f"You would receive from the {MHQAAgentNames.REASONER} agent a response to the user question. "
                    "Review the response in terms of quality and highlight any potential gaps. "
                    "Once done, provide the response and your review to the user. "
                    "Make sure that your final answer is in properly formatted Markdown.\n"
                ),
                tools=[],
                llm=Ollama(**self.llm_config[MHQAAgentNames.REVIEWER.lower()]),
            )

            self.workflow = AgentWorkflow(
                agents=[
                    decomposer_agent,
                    responder_agent,
                    reasoner_agent,
                    reviewer_agent,
                ],
                initial_state={
                    "user_query": "to be provided",
                    "sub_questions": [],
                    "evidences": [],
                    "combined_response": "not written yet",
                    "final_response": "not written yet",
                    "review_notes": "not written yet",
                },
                root_agent=decomposer_agent.name,
            )
            self.workflow_context = Context(
                workflow=self.workflow,
            )

            self.workflow_memory = Memory.from_defaults(
                session_id=str(self.id),
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
                    self.workflow_memory.put(msg)
                logger.info(
                    f"Restored {len(parsed_messages)} messages from state store"
                )

        logger.info(f"{self.__class__.__name__} ({self.id}) activated")

    async def _on_deactivate(self) -> None:
        logger.info(f"{self.__class__.__name__} ({self.id}) deactivated")

    async def respond(self, data: dict) -> dict:
        user_input = data.get("user_input", "")
        wf_handler = self.workflow.run(
            user_msg=user_input,
            memory=self.workflow_memory,
            context=self.workflow_context,
            max_iterations=5,
        )
        full_response = ""
        tool_invocations: List[MCPToolInvocation] = []
        pubsub_topic_name = f"{PubSubTopics.MHQA_RESPONSE}/{self.id}"
        with DaprClient() as dc:
            current_agent = ""
            async for event in wf_handler.stream_events():
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
                        MHQAActor.parse_tool_message_from_str(
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
                        pubsub_name=ParsedEnvVars().DAPR_PUBSUB_NAME,
                        topic_name=pubsub_topic_name,
                        data=response.model_dump_json().encode(),
                    )
            response.status = MHQAResponseStatus.completed
            dc.publish_event(
                pubsub_name=ParsedEnvVars().DAPR_PUBSUB_NAME,
                topic_name=pubsub_topic_name,
                data=response.model_dump_json().encode(),
            )
        memory_messages = await self.workflow_memory.aget_all()
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
            chat_messages = await self.workflow_memory.aget_all()
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
                    parsed_tool_output = MHQAActor.parse_tool_message_from_str(
                        msg.content
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
            await self.workflow_memory.areset()
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
