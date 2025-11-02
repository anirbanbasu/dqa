from __future__ import annotations as _annotations
from enum import StrEnum, auto
import json
import logging


from dataclasses import dataclass, field
import os
from typing import AsyncIterable

from pydantic import BaseModel

from pydantic_ai import (
    ModelMessage,
    ModelMessagesTypeAdapter,
    format_as_xml,
)

from pydantic_ai import (
    Agent,
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

from pydantic_ai.agent.abstract import EventStreamHandler
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_core import to_jsonable_python
from pydantic_graph import BaseNode, End, Graph, GraphRunContext

from dqa import EnvVars, ic


logger = logging.getLogger(__name__)


class MHQAWorkflowAgentType(StrEnum):
    RESPONDER = auto()
    REVIEWER = auto()


@dataclass
class ResponseState:
    user_message: str
    responder_messages: list[ModelMessage] = field(default_factory=list)


@dataclass
class Response:
    is_user_message_a_statement: bool
    body: str


class ResponseRevisionRequired(BaseModel):
    review: str


class ResponseOK(BaseModel):
    pass


class MHQAWorkflowHelper:
    """Multi-Hop Question Answering Workflow Helper."""

    def __init__(
        self,
        message_history_json: str | None = None,
        agent_event_stream_handler: EventStreamHandler | None = None,
    ):
        if not hasattr(self, "initialised"):
            self.update_message_history_from_json(message_history_json)
            if agent_event_stream_handler is not None:
                self.agent_event_stream_handler = agent_event_stream_handler
            else:
                # self.agent_event_stream_handler = (
                #     MHQAWorkflowHelper.event_stream_handler
                # )
                self.agent_event_stream_handler = None
            self.initialise()

    def initialise(self):
        try:
            if not hasattr(self, "llm_config"):
                self.llm_config = {}
                llm_config_file = EnvVars.LLM_CONFIG_FILE
                if os.path.exists(llm_config_file):
                    with open(llm_config_file, "r") as f:
                        self.llm_config = json.load(f)
            if MHQAWorkflowAgentType.RESPONDER.value.lower() not in self.llm_config:
                raise ValueError(
                    f"LLM configuration for responder agent not found in {llm_config_file}"
                )
            if MHQAWorkflowAgentType.REVIEWER.value.lower() not in self.llm_config:
                raise ValueError(
                    f"LLM configuration for reviewer agent not found in {llm_config_file}"
                )
            if not hasattr(self, "mcp_config"):
                self.mcp_config = {}
                mcp_config_file = EnvVars.MCP_CONFIG_FILE
                if os.path.exists(mcp_config_file):
                    with open(mcp_config_file, "r") as f:
                        self.mcp_config = json.load(f)
                    configured_mcp_servers = self.mcp_config.get("mcpServers", {})
                    logger.info(
                        f"Loaded {len(configured_mcp_servers)} MCP configurations from {mcp_config_file}: {list(configured_mcp_servers.keys())}"
                    )

            if EnvVars.PREFECT_API_URL:
                logger.warning(
                    f"Prefect API URL is set to '{EnvVars.PREFECT_API_URL}'. Durable Agents will be used. However, this is an experimental feature!"
                )

            responder_llm_config = self.llm_config.get(
                MHQAWorkflowAgentType.RESPONDER.value.lower(), {}
            )

            basic_responder_agent = Agent(
                name=f"MHQA{MHQAWorkflowAgentType.RESPONDER.value.capitalize()}Agent",
                model=OpenAIChatModel(
                    model_name=responder_llm_config.get("model", None),
                    provider=OllamaProvider(
                        base_url=responder_llm_config.get("base_url", None)
                    ),
                ),
                system_prompt=(
                    "You are a multi-purpose multi-hop question answering agent capable of decomposing questions from the user, "
                    "gathering evidences using available tools, reasoning through them, "
                    "and providing a final response to the user.\n"
                    "If the user query is not a question but a statement, respond with an acknowledgment only.\n"
                ),
                output_type=Response,
                toolsets=[FastMCPToolset(self.mcp_config)],
                retries=EnvVars.AGENT_RETRY_ATTEMPTS,
                event_stream_handler=self.agent_event_stream_handler,
            )

            self._responder_agent = (
                PrefectAgent(basic_responder_agent)
                if EnvVars.PREFECT_API_URL
                else basic_responder_agent
            )

            reviewer_llm_config = self.llm_config.get(
                MHQAWorkflowAgentType.REVIEWER.value.lower(), {}
            )

            basic_reviewer_agent = Agent[None, ResponseRevisionRequired | ResponseOK](
                name=f"MHQA{MHQAWorkflowAgentType.REVIEWER.value.capitalize()}Agent",
                model=OpenAIChatModel(
                    model_name=reviewer_llm_config.get("model", None),
                    provider=OllamaProvider(
                        base_url=reviewer_llm_config.get("base_url", None)
                    ),
                ),
                system_prompt=(
                    "You are a reviewer agent. Your task is to review an answer given the user message in terms of its correctness and completeness.\n"
                    "Only if the answer needs revision, including further evidence gathering, provide specific feedback.\n"
                ),
                output_type=ResponseRevisionRequired | ResponseOK,
                retries=EnvVars.AGENT_RETRY_ATTEMPTS,
                event_stream_handler=self.agent_event_stream_handler,
            )

            self._reviewer_agent = (
                PrefectAgent(basic_reviewer_agent)
                if EnvVars.PREFECT_API_URL
                else basic_reviewer_agent
            )

            self._mhqa_graph = Graph(nodes=(Respond, Review))

            self.initialised = (
                hasattr(self, "llm_config")
                and hasattr(self, "mcp_config")
                and hasattr(self, "_message_history")
                and hasattr(self, "_responder_agent")
                and hasattr(self, "_reviewer_agent")
                and hasattr(self, "_mhqa_graph")
            )
        except Exception as e:
            logger.error(f"Error initialising MHQA workflow. {e}")

    def update_message_history_from_json(self, message_history_json: str | None = None):
        self._message_history_json = message_history_json
        if hasattr(self, "_message_history_json") and self._message_history_json:
            self._message_history = ModelMessagesTypeAdapter.validate_python(
                json.loads(self._message_history_json)
            )
        else:
            self._message_history = []

    async def run_workflow(self, user_message: str):
        state = ResponseState(
            user_message=user_message, responder_messages=self._message_history
        )
        self._initial_run_state = state
        result = await self._mhqa_graph.run(Respond(helper=self), state=state)
        self._message_history = ModelMessagesTypeAdapter.validate_python(
            to_jsonable_python(result.state.responder_messages)
        )
        self._message_history_json = ModelMessagesTypeAdapter.dump_json(
            self._message_history
        ).decode()
        return result

    def create_respond_node(self, response_feedback: str | None = None) -> Respond:
        return Respond(helper=self, response_feedback=response_feedback)

    def create_review_node(self, response_text: str | None = None) -> Review:
        return Review(helper=self, response_text=response_text)

    async def handle_event(event: AgentStreamEvent):
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, ThinkingPart):
                print(f"\n[Thinking]\n{event.part.content}", flush=True, end="")
            else:
                # Such as starting TextPart, ToolCallPart, etc.
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

    async def event_stream_handler(
        ctx: RunContext,
        event_stream: AsyncIterable[AgentStreamEvent],
    ):
        async for event in event_stream:
            await MHQAWorkflowHelper.handle_event(event)


class Respond(BaseNode[ResponseState]):
    # response_feedback: str | None = None

    def __init__(
        self, helper: MHQAWorkflowHelper, response_feedback: str | None = None
    ):
        super().__init__()
        self._helper = helper
        if not self._helper.initialised:
            self._helper.initialise()
        self.response_feedback = response_feedback

    async def run(self, ctx: GraphRunContext[ResponseState]) -> Review | End[str]:
        if self.response_feedback:
            prompt = (
                f"Your previous response was reviewed with the following feedback.\n{self.response_feedback}\n"
                "Revise your response accordingly and output your updated response."
            )
        else:
            prompt = f"Provide your response to the message from the user.\n{ctx.state.user_message}"
        ic(ctx.state.responder_messages)
        result = await self._helper._responder_agent.run(
            prompt, message_history=ctx.state.responder_messages
        )
        ctx.state.responder_messages.extend(result.new_messages())
        if result.output.is_user_message_a_statement:
            return End(result.output.body)
        else:
            return self._helper.create_review_node(response_text=result.output.body)


class Review(BaseNode[ResponseState, None, str]):
    # response_text: str | None

    def __init__(self, helper: MHQAWorkflowHelper, response_text: str | None = None):
        super().__init__()
        self._helper = helper
        if not self._helper.initialised:
            self._helper.initialise()
        self.response_text = response_text

    async def run(self, ctx: GraphRunContext[ResponseState]) -> Respond | End[str]:
        prompt = format_as_xml(
            {
                "user_message": ctx.state.user_message,
                "response": self.response_text,
                "supporting_evidences": ctx.state.responder_messages,
            }
        )
        result = await self._helper._reviewer_agent.run(prompt)
        if isinstance(result.output, ResponseRevisionRequired):
            return self._helper.create_respond_node(
                response_feedback=result.output.review
            )
        else:
            return End(self.response_text)
