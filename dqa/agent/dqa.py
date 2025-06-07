import asyncio
import json
from typing import Any, AsyncIterable, Literal
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel
from langchain_core.messages import AIMessage, ToolMessage

from dqa.common import EnvironmentVariables
from dqa.utils import parse_env
from dqa.common import ic

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_community.tools import DuckDuckGoSearchResults


class ResponseFormat(BaseModel):
    """Respond to the user in this format."""

    status: Literal["input_required", "completed", "error"] = "input_required"
    message: str


class DQAAgent:
    SYSTEM_INSTRUCTION = (
        "You are a specialised assistant for answering multi-hop questions. "
        "Your task is to answer the user's question by breaking it down into smaller, manageable sub-questions. "
        "You should use the tools provided to gather information and answer each sub-question. "
        "If you use a tool, you must summarise the tool response in your final answer. "
        "Set response status to input_required if the user needs to provide more information to continue answering the given question. "
        "Set response status to error if there is an error while processing the request. "
        "Set response status to completed if the request is complete."
    )

    def __init__(self):
        """Initialize the DQA Agent."""
        llm_config_file = parse_env(
            EnvironmentVariables.DQA_LLM_CONFIG,
            EnvironmentVariables.DEFAULT__DQA_LLM_CONFIG,
        )
        with open(llm_config_file, "r") as f:
            self.llm_config = json.load(f)
            f.close()
        ic(self.llm_config, type(self.llm_config))
        self.memory = MemorySaver()
        self.model = ChatOllama(**self.llm_config)
        self.tools = [DuckDuckGoSearchResults()]
        with open(
            parse_env(
                EnvironmentVariables.DQA_MCP_CLIENT_CONFIG,
                EnvironmentVariables.DEFAULT_DQA_MCP_CLIENT_CONFIG,
            ),
            "r",
        ) as f:
            mcp_config = json.load(f)
            f.close()
        client = MultiServerMCPClient(connections=mcp_config)

        mcp_tools = asyncio.get_event_loop().run_until_complete(client.get_tools())
        if mcp_tools:
            ic(mcp_tools, mcp_config)
            self.tools.extend(mcp_tools)
        self.graph = create_react_agent(
            model=self.model,
            prompt=DQAAgent.SYSTEM_INSTRUCTION,
            checkpointer=self.memory,
            tools=self.tools,
            response_format=ResponseFormat,
        )

    async def ainvoke(self, query, context_id) -> str:
        config = {"configurable": {"thread_id": context_id}}
        await self.graph.ainvoke({"messages": [("user", query)]}, config)
        return self.get_agent_response(config)

    async def astream(self, query, context_id) -> AsyncIterable[dict[str, Any]]:
        inputs = {"messages": [("user", query)]}
        config = {"configurable": {"thread_id": context_id}}

        async for item in self.graph.astream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            ic(message, type(message))
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Attempting to answer the question...",
                }
            elif isinstance(message, ToolMessage):
                yield {
                    "is_task_complete": False,
                    "require_user_input": False,
                    "content": "Processing tool responses...",
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get("structured_response")
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == "input_required":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "error":
                return {
                    "is_task_complete": False,
                    "require_user_input": True,
                    "content": structured_response.message,
                }
            if structured_response.status == "completed":
                return {
                    "is_task_complete": True,
                    "require_user_input": False,
                    "content": structured_response.message,
                }

        return {
            "is_task_complete": False,
            "require_user_input": True,
            "content": (
                "Unable to answer the question at this time. "
                "Please provide more information or try again later."
            ),
        }
