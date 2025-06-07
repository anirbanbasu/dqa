import asyncio
import json
from typing import Any, AsyncIterable, Literal
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel

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
        "Even if you know the answer to the question from your memory, try using the relevant tools to find the answer. "
        "Do not hallucinate or make up answers. "
        "If you cannot answer the question, respond with 'I don't know'. "
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
        ic(self.llm_config)
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
            self.tools.extend(mcp_tools)
        self.graph = create_react_agent(
            model=self.model,
            prompt=DQAAgent.SYSTEM_INSTRUCTION,
            checkpointer=self.memory,
            tools=self.tools,
        )

    async def ainvoke(self, query_history, context_id) -> str:
        if len(query_history) == 1:
            # This is the first message in a new conversation, so delete the entire thread from memory
            await self.graph.checkpointer.adelete_thread(context_id)
        config = {"configurable": {"thread_id": context_id}}
        extracted_messages = []
        for message in query_history:
            if isinstance(message, dict):
                extracted_messages.append((message["role"], message["content"]))
            else:
                extracted_messages.append((message.role, message.content))
        ic(extracted_messages)
        await self.graph.ainvoke({"messages": [("user", extracted_messages)]}, config)
        return self.get_agent_response(config)

    async def astream(self, query_history, context_id) -> AsyncIterable[dict[str, Any]]:
        """Stream the response from the agent based on the query history."""
        if len(query_history) == 1:
            # This is the first message in a new conversation, so delete the entire thread from memory
            await self.graph.checkpointer.adelete_thread(context_id)
        config = {"configurable": {"thread_id": context_id}}
        extracted_messages = []
        for message in query_history:
            if isinstance(message, dict):
                extracted_messages.append((message["role"], message["content"]))
            else:
                extracted_messages.append((message.role, message.content))
        ic(extracted_messages)
        inputs = {"messages": extracted_messages}

        async for item in self.graph.astream(inputs, config, stream_mode="values"):
            message = item["messages"][-1]
            ic(message, type(message))
            yield message
