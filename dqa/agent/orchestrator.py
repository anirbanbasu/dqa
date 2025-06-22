import json
from dqa.common import EnvironmentVariables
from dqa.utils import parse_env
from dqa.common import ic

from llama_index.tools.mcp import (
    get_tools_from_mcp_url,
)

from llama_index.core.memory import Memory
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context
from llama_index.core.workflow.handler import WorkflowHandler
from llama_index.core.agent.workflow import AgentWorkflow, FunctionAgent


class DQAOrchestrator:
    def __init__(self, session_id: str, use_mcp: bool = True):
        self.tools = []
        llm_config_file = parse_env(
            EnvironmentVariables.DQA_LLM_CONFIG,
            EnvironmentVariables.DEFAULT__DQA_LLM_CONFIG,
        )
        with open(llm_config_file, "r") as f:
            self.llm_config = json.load(f)
            f.close()
        ic(self.llm_config)

        if use_mcp:
            with open(
                parse_env(
                    EnvironmentVariables.DQA_MCP_CLIENT_CONFIG,
                    EnvironmentVariables.DEFAULT_DQA_MCP_CLIENT_CONFIG,
                ),
                "r",
            ) as f:
                mcp_configurations = json.load(f)
                f.close()
                ic(mcp_configurations)

                for config_name, config in mcp_configurations.items():
                    mcp_tools = get_tools_from_mcp_url(
                        command_or_url=(
                            config["url"]
                            if config["transport"] != "stdio"
                            else config["command"]
                        ),
                    )
                    self.tools.extend(mcp_tools)

        chat_agent = FunctionAgent(
            name="DQA Chat Agent",
            description="The main agent that handles user chat.",
            system_prompt="You are a specialised assistant for answering multi-hop questions. "
            "Your task is to answer the user's question by breaking it down into smaller, manageable sub-questions. "
            "You should use the tools provided to gather information and answer each sub-question. "
            "Do not hallucinate or make up tool calls or answers. "
            "If you cannot answer the question, respond with 'I don't know'. ",
            tools=self.tools,
            llm=Ollama(**self.llm_config["ollama"]),
        )

        self.workflow_memory = Memory.from_defaults(
            session_id=session_id,
        )

        self.workflow = AgentWorkflow(agents=[chat_agent])
        self.workflow_context = Context(
            workflow=self.workflow,
        )

    def run(self, query: str) -> WorkflowHandler:
        return self.workflow.run(
            user_msg=query,
            memory=self.workflow_memory,
            context=self.workflow_context,
        )
