import json
import logging
import os
import re
from typing import Any, Dict

from dqa import ic

from llama_index.tools.mcp import aget_tools_from_mcp_url, BasicMCPClient

from llama_index.core.workflow.events import InputRequiredEvent, HumanResponseEvent

from llama_index.core.memory import Memory
from llama_index.llms.ollama import Ollama
from llama_index.core.workflow import Context

from llama_index.core.agent.workflow import (
    AgentWorkflow,
    FunctionAgent,
    ReActAgent,
)


from dqa import EnvVars

logger = logging.getLogger(__name__)


class MHQAAgentWorkflowOrchestrator:
    DECOMPOSER = "Decomposer"
    RESPONDER = "Responder"
    REASONER = "Reasoner"
    REVIEWER = "Reviewer"

    def __init__(self):
        self.initialised = False

    async def initialise(self, actor_id: str):
        if not hasattr(self, "llm_config"):
            self.llm_config = {}
            llm_config_file = EnvVars.LLM_CONFIG_FILE
            if os.path.exists(llm_config_file):
                with open(llm_config_file, "r") as f:
                    self.llm_config = json.load(f)
        if not hasattr(self, "mcp_config"):
            self.mcp_features = []
            self.mcp_config = {}
            try:
                mcp_config_file = EnvVars.MCP_CONFIG_FILE
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
        if self.mcp_features is None:
            raise ValueError(
                "mcp_features must be set before initialising the orchestrator."
            )

        if not hasattr(self, "workflow"):
            decomposer_agent = FunctionAgent(
                name=MHQAAgentWorkflowOrchestrator.DECOMPOSER,
                description="Decomposes complex questions into simpler sub-questions.",
                system_prompt=(
                    f"You are the {MHQAAgentWorkflowOrchestrator.DECOMPOSER} agent. "
                    "Determine if the user query is a question or a statement.\n"
                    "If the user query is a question and it has no direct answer, decompose it into smaller sub-questions. "
                    "If the user query is a simple question then decompose it into just one single sub-question, which is the question posed by the user."
                    "If the user input is only a statement but not a question then respond with an acknowledgment only.\n"
                    "Store in state the decomposed sub-questions in a list format. "
                    f"If the user query is not a statement, hand off to the {MHQAAgentWorkflowOrchestrator.RESPONDER} agent.\n"
                ),
                tools=[],
                can_handoff_to=[MHQAAgentWorkflowOrchestrator.RESPONDER],
                llm=Ollama(
                    **self.llm_config[MHQAAgentWorkflowOrchestrator.DECOMPOSER.lower()]
                ),
            )

            responder_agent = FunctionAgent(
                name=MHQAAgentWorkflowOrchestrator.RESPONDER,
                description="Responds to sub-questions using available tools.",
                system_prompt=(
                    f"You are the {MHQAAgentWorkflowOrchestrator.RESPONDER} agent.\n"
                    # f"You would receive a list of questions by the {MHQAAgentWorkflowOrchestrator.DECOMPOSER} agent. "
                    "Call the appropriate tools for each question given to you and output a list of tool call responses. "
                    "Store in state the responses you have gathered for each question. "
                    f"Once you are finished with all the questions, hand off to the {MHQAAgentWorkflowOrchestrator.REASONER} agent.\n"
                ),
                tools=self.mcp_features,
                can_handoff_to=[MHQAAgentWorkflowOrchestrator.REASONER],
                llm=Ollama(
                    **self.llm_config[MHQAAgentWorkflowOrchestrator.RESPONDER.lower()]
                ),
            )

            reasoner_agent = ReActAgent(
                name=MHQAAgentWorkflowOrchestrator.REASONER,
                description="Reasons through gathered evidences to formulate a combined response.",
                system_prompt=(
                    f"You are the {MHQAAgentWorkflowOrchestrator.REASONER} agent.\n"
                    # f"You would receive from the {MHQAAgentWorkflowOrchestrator.RESPONDER} agent a list of questions and the evidences it gathered for each question. "
                    "Reason through the evidences for the individual questions and combine them into a single response that serves as a coherent answer to the original user query.\n"
                    f"Hand off to the {MHQAAgentWorkflowOrchestrator.REVIEWER} agent for a review of your combined response.\n"
                ),
                tools=[],
                can_handoff_to=[MHQAAgentWorkflowOrchestrator.REVIEWER],
                llm=Ollama(
                    **self.llm_config[MHQAAgentWorkflowOrchestrator.REASONER.lower()]
                ),
            )

            reviewer_agent = ReActAgent(
                name=MHQAAgentWorkflowOrchestrator.REVIEWER,
                description="Reviews the combined response for quality and completeness.",
                system_prompt=(
                    f"You are the {MHQAAgentWorkflowOrchestrator.REVIEWER} agent.\n"
                    # f"You would receive from the {MHQAAgentWorkflowOrchestrator.REASONER} agent a response to the user question. "
                    "Review the response in terms of quality and highlight any potential gaps. "
                    "Once done, provide the response and your review to the user. "
                    "Make sure that your final answer is in correct Markdown format.\n"
                ),
                tools=[],
                llm=Ollama(
                    **self.llm_config[MHQAAgentWorkflowOrchestrator.REVIEWER.lower()]
                ),
            )

            self.workflow = AgentWorkflow(
                agents=[
                    decomposer_agent,
                    responder_agent,
                    reasoner_agent,
                    reviewer_agent,
                ],
                initial_state={
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
                session_id=actor_id,
            )
        self.initialised = (
            hasattr(self, "mcp_features")
            and hasattr(self, "workflow")
            and hasattr(self, "workflow_context")
            and hasattr(self, "workflow_memory")
        )

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
