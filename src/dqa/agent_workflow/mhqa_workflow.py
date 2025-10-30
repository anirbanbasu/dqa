from __future__ import annotations as _annotations
from enum import StrEnum, auto
import json
import logging

from dataclasses import dataclass, field
import os
from typing import AsyncIterable, ClassVar

from pydantic import BaseModel

from pydantic_ai import ModelMessage, format_as_xml

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
from pydantic_ai.durable_exec.prefect import PrefectAgent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from pydantic_graph import BaseNode, End, GraphRunContext

from dqa import EnvVars


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


class MHQAWorkflow:
    """Multi-Hop Question Answering Workflow."""

    _instance: ClassVar = None

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
                event_stream_handler=MHQAWorkflow.event_stream_handler,
            )

            self.responder_agent = (
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
                event_stream_handler=MHQAWorkflow.event_stream_handler,
            )

            self.reviewer_agent = (
                PrefectAgent(basic_reviewer_agent)
                if EnvVars.PREFECT_API_URL
                else basic_reviewer_agent
            )

            # ic(self.responder_agent, self.reviewer_agent)

            self.initialised = (
                hasattr(self, "llm_config")
                and hasattr(self, "mcp_config")
                and hasattr(self, "responder_agent")
                and hasattr(self, "reviewer_agent")
            )
        except Exception as e:
            logger.error(f"Error initialising MHQA workflow. {e}")

    async def handle_event(event: AgentStreamEvent):
        if isinstance(event, PartStartEvent):
            if isinstance(event.part, ThinkingPart):
                print(f"\n[Thinking]\n{event.part.content}", flush=True, end="")
            else:
                print()
        elif isinstance(event, PartDeltaEvent):
            if isinstance(event.delta, TextPartDelta):
                # print(
                #     f"[Request] Part {event.index} text delta: {event.delta.content_delta!r}"
                # )
                pass
            elif isinstance(event.delta, ThinkingPartDelta):
                print(f"{event.delta.content_delta}", flush=True, end="")
            elif isinstance(event.delta, ToolCallPartDelta):
                # print(
                #     f"[Request] Part {event.index} args delta: {event.delta.args_delta}"
                # )
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
            # print(
            #     f"[Result] The model starting producing a final result (tool_name={event.tool_name})"
            # )
            pass

    async def event_stream_handler(
        ctx: RunContext,
        event_stream: AsyncIterable[AgentStreamEvent],
    ):
        async for event in event_stream:
            await MHQAWorkflow.handle_event(event)

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MHQAWorkflow, cls).__new__(cls)
            cls._instance.initialise()
        return cls._instance


class Respond(BaseNode[ResponseState]):
    response_feedback: str | None = None

    async def run(self, ctx: GraphRunContext[ResponseState]) -> Review | End[str]:
        if self.response_feedback:
            prompt = (
                f"Your previous response was reviewed with the following feedback.\n{self.response_feedback}\n"
                "Revise your response accordingly and output your updated response."
            )
        else:
            prompt = f"Provide your response to the message from the user.\n{ctx.state.user_message}"
        result = await MHQAWorkflow().responder_agent.run(
            prompt, message_history=ctx.state.responder_messages
        )
        ctx.state.responder_messages.extend(result.new_messages())
        if result.output.is_user_message_a_statement:
            return End(result.output.body)
        else:
            review_node = Review()
            review_node.response_text = result.output.body
            return review_node


class Review(BaseNode[ResponseState, None, str]):
    response_text: str

    async def run(self, ctx: GraphRunContext[ResponseState]) -> Respond | End[str]:
        prompt = format_as_xml(
            {
                "user_message": ctx.state.user_message,
                "response": self.response_text,
                "supporting_evidences": ctx.state.responder_messages,
            }
        )
        result = await MHQAWorkflow().reviewer_agent.run(prompt)
        if isinstance(result.output, ResponseRevisionRequired):
            respond_node = Respond()
            respond_node.response_feedback = result.output.review
            return respond_node
        else:
            return End(self.response_text)
