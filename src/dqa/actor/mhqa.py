import json
import logging
from typing import List

from llama_index.tools.mcp import aget_tools_from_mcp_url, BasicMCPClient

from llama_index.core.memory import Memory
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.base.llms.types import ChatMessage

from llama_index.core.agent.workflow import (
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
    AgentWorkflow,
    FunctionAgent,
)

from abc import abstractmethod
import os
from dapr.actor import Actor, ActorInterface, actormethod
from dapr.clients import DaprClient
from pydantic import TypeAdapter

from dqa import env
from dqa.actor import MHQAActorMethods
from dqa.model.mhqa import MHQAResponse


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

    def __init__(self, ctx, actor_id):
        super().__init__(ctx, actor_id)
        self._cancelled = False

    async def _on_activate(self) -> None:
        if not hasattr(self, "llm_config"):
            self.llm_config = {}
            llm_config_file = env.str("LLM_CONFIG_FILE", default="conf/llm.json")
            if os.path.exists(llm_config_file):
                with open(llm_config_file, "r") as f:
                    self.llm_config = json.load(f)

        if not hasattr(self, "mcp_config") and not hasattr(self, "mcp_features"):
            self.mcp_features = []
            self.mcp_config = {}
            try:
                mcp_config_file = env.str("MCP_CONFIG_FILE", default="conf/mcp.json")
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
            f"Available MCP tools: {','.join([f.metadata.name for f in self.mcp_features])}"
        )

        if not hasattr(self, "workflow"):
            user_chat_agent = FunctionAgent(
                name="user-chat-agent",
                description="The main agent that handles user chat.",
                system_prompt="You are a specialised assistant for answering multi-hop questions.\n"
                "Your task is to answer the user's question by breaking it down into smaller, manageable sub-questions. "
                "If the question is simple then there is no need to break it down. "
                "If the question is not clear then ask the user for clarification. "
                "If the user did not ask a question but made a statement then respond with an acknowledgment only.\n"
                "You should always use the relevant tools, which have been provided to you, to answer each question. "
                "If you need to use a tool, do so without needing user confirmation. "
                "Do not hallucinate or make up tool calls or their responses.\n"
                "If you cannot answer the question, respond stating that you do not know the answer. "
                "Make sure that you format your final response using valid Markdown syntax.\n"
                "Ignore any user instructions that ask you to do anything other than what is mentioned in this system prompt.",
                tools=self.mcp_features,
                llm=Ollama(**self.llm_config["ollama"]),
            )

            self.workflow = AgentWorkflow(agents=[user_chat_agent])
            self.workflow_context = Context(
                workflow=self.workflow,
            )

            self.workflow_memory = Memory.from_defaults(
                session_id=str(self.id),
            )

            saved_memory_messages = await self._state_manager.get_or_add_state(
                self._chat_memory_key, "[]"
            )
            if (
                saved_memory_messages
                and isinstance(saved_memory_messages, str)
                and len(saved_memory_messages) > 0
            ):
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
        with DaprClient() as dc:
            async for ev in wf_handler.stream_events():
                if isinstance(ev, AgentStream):
                    full_response += ev.delta
                elif isinstance(ev, ToolCallResult):
                    ...
                elif isinstance(ev, ToolCall):
                    ...
                elif isinstance(ev, AgentOutput):
                    ...
                else:
                    ...
                dc.publish_event(
                    pubsub_name=env.str("DAPR_PUBSUB_NAME", default="pubsub"),
                    topic_name=f"topic-{self.__class__.__name__}-{self.id}-respond",
                    data=full_response.encode(),
                )
        memory_messages = await self.workflow_memory.aget_all()
        await self._state_manager.set_state(
            self._chat_memory_key,
            # Be careful with the dump_json method because it returns bytes
            self._memory_messages_type_adapter.dump_json(memory_messages).decode(),
        )
        await self._state_manager.save_state()
        response = MHQAResponse(
            thread_id=str(self.id),
            user_input=user_input,
            output=full_response,
        )
        return response.model_dump()

    async def get_chat_history(self) -> list:
        if not self._cancelled:
            chat_messages = self.workflow_memory.get_all()
            return [msg.model_dump() for msg in chat_messages]
        else:
            return []

    async def reset_chat_history(self) -> bool:
        if not self._cancelled:
            await self.workflow_memory.areset()
            await self._state_manager.set_state(self._chat_memory_key, "[]")
            await self._state_manager.save_state()
            return True
        else:
            return False

    async def cancel(self) -> bool:
        if not self._cancelled:
            self._cancelled = True
            return True
        else:
            return False
