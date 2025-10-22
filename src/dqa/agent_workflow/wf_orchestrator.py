import asyncio
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
    FunctionAgent,
    ReActAgent,
)

from llama_index.core.agent.workflow import (
    AgentOutput,
    AgentInput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)


from dqa import EnvVars
from dqa.agent_workflow.mhqa_workflow import MHQAFlow, MHQAFlowAgentType

logger = logging.getLogger(__name__)


class MHQAWorkflowOrchestrator:
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
            self.decomposer_agent = FunctionAgent(
                name=MHQAFlowAgentType.DECOMPOSER.value,
                description="Decomposes complex questions into simpler sub-questions.",
                system_prompt=(
                    f"You are the {MHQAFlowAgentType.DECOMPOSER.value} agent. "
                    "Determine if the user query is a question or a statement.\n"
                    "If the user query is a simple question then decompose it into just one single sub-question, which is the question posed by the user."
                    "Otherwise, respond with a list of decomposed distinct sub-questions in a format as shown in the example below.\n"
                    # "The decomposed sub-questions must be precise and as necessary to answer the original question.\n" \
                    "\n<example-decomposition>\n"
                    '{"user_message": "The user message exactly as you received it."\n'
                    '"statement": false\n'
                    '"sub_questions": [\n'
                    '  "Sub-question 1",\n'
                    '  "Sub-question 2",\n'
                    "  ...\n"
                    "]}\n"
                    "</example-decomposition>\n"
                    "If the user input is only a statement but not a question then respond with an acknowledgment only, in the format shown below.\n"
                    "\n<example-acknowledgement>\n"
                    '{"user_message": "The user message exactly as you received it."\n'
                    '"statement": true\n'
                    '"acknowledgement": "Your acknowledgement to the user message."\n'
                    "</example-acknowledgement>\n"
                ),
                llm=Ollama(
                    **self.llm_config[MHQAFlowAgentType.DECOMPOSER.value.lower()]
                ),
            )

            self.responder_agent = FunctionAgent(
                name=MHQAWorkflowOrchestrator.RESPONDER,
                description="Gathers evidences for sub-questions using available tools.",
                system_prompt=(
                    f"You are the {MHQAWorkflowOrchestrator.RESPONDER} agent.\n"
                    # f"You would receive a list of questions by the {MHQAAgentWorkflowOrchestrator.DECOMPOSER} agent. "
                    "Call one or more of the tools provided to you to gather evidences/responses for each question given to you. "
                    "Generate a list of tool call responses. "
                    "Store in state the responses you have gathered for each question. "
                    f"After processing the questions given to you, NEVER respond to the user directly but always hand off to the {MHQAWorkflowOrchestrator.REASONER} agent.\n"
                ),
                can_handoff_to=[MHQAWorkflowOrchestrator.REASONER],
                tools=self.mcp_features,
                llm=Ollama(
                    **self.llm_config[MHQAWorkflowOrchestrator.RESPONDER.lower()]
                ),
            )

            self.reasoner_agent = ReActAgent(
                name=MHQAWorkflowOrchestrator.REASONER,
                description="Reasons through gathered evidences to formulate a combined response.",
                system_prompt=(
                    f"You are the {MHQAWorkflowOrchestrator.REASONER} agent.\n"
                    # f"You would receive from the {MHQAAgentWorkflowOrchestrator.RESPONDER} agent a list of questions and the evidences it gathered for each question. "
                    "Reason through the evidences for the individual questions and combine them into a single response that serves as a coherent answer to the original user query.\n"
                    f"NEVER respond to the user directly but always hand off to the {MHQAWorkflowOrchestrator.REVIEWER} agent for a review of your combined response.\n"
                ),
                can_handoff_to=[MHQAWorkflowOrchestrator.REVIEWER],
                llm=Ollama(
                    **self.llm_config[MHQAWorkflowOrchestrator.REASONER.lower()]
                ),
            )

            self.reviewer_agent = ReActAgent(
                name=MHQAWorkflowOrchestrator.REVIEWER,
                description="Reviews the combined response for quality and completeness; and responds to the user.",
                system_prompt=(
                    f"You are the {MHQAWorkflowOrchestrator.REVIEWER} agent.\n"
                    # f"You would receive from the {MHQAAgentWorkflowOrchestrator.REASONER} agent a response to the user question. "
                    "Review the response in terms of quality and highlight any potential gaps. "
                    f"Respond to the user with the {MHQAWorkflowOrchestrator.REASONER} agent response and your review. "
                    "Make sure that your final response is in correct Markdown format.\n"
                ),
                llm=Ollama(
                    **self.llm_config[MHQAWorkflowOrchestrator.REVIEWER.lower()]
                ),
            )

            self.workflow = MHQAFlow(
                agents={
                    MHQAFlowAgentType.DECOMPOSER: self.decomposer_agent,
                    MHQAFlowAgentType.RESPONDER: self.responder_agent,
                    MHQAFlowAgentType.REASONER: self.reasoner_agent,
                    MHQAFlowAgentType.REVIEWER: self.reviewer_agent,
                },
                verbose=True,
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


async def init_and_chat(actor_id: str, user_query: str):
    orchestrator = MHQAWorkflowOrchestrator()
    await orchestrator.initialise(actor_id=actor_id)
    if not orchestrator.initialised:
        raise RuntimeError("Orchestrator failed to initialise properly.")

    print(", ".join([f.metadata.name for f in orchestrator.mcp_features]))
    # Run the workflow
    wfh = orchestrator.workflow.run(
        user_msg=user_query,
        memory=orchestrator.workflow_memory,
        context=orchestrator.workflow_context,
        max_iterations=1,
    )

    full_response = ""
    current_agent = ""
    async for event in wfh.stream_events():
        if (
            hasattr(event, "current_agent_name")
            and event.current_agent_name != current_agent
        ):
            current_agent = event.current_agent_name
            print(f"\n{'=' * 50}", flush=True)
            print(f"🤖 Agent: {current_agent}", flush=True)
            print(f"{'=' * 50}\n", flush=True)
        if isinstance(event, AgentStream):
            if event.delta:
                print(event.delta, end="", flush=True)
                full_response += event.delta
        elif isinstance(event, AgentInput):
            print(f"\n📥 {event.current_agent_name} Input:", event.input, flush=True)
        elif isinstance(event, ToolCall):
            print(f"\n🔨 Calling Tool: {event.tool_name}", flush=True)
            print(f"  With arguments: {event.tool_kwargs}", flush=True)
        elif isinstance(event, ToolCallResult):
            print(f"\n🔧 Tool Result ({event.tool_name}):", flush=True)
            print(f"  Arguments: {event.tool_kwargs}", flush=True)
            print(f"  Output: {event.tool_output}", flush=True)
        elif isinstance(event, AgentOutput):
            if event.response.content:
                print(
                    f"\n📤 {event.current_agent_name} Output:",
                    event.response.content,
                    flush=True,
                )
            if event.tool_calls:
                print(
                    "\n🛠️  Planning to use tools:",
                    [call.tool_name for call in event.tool_calls],
                    flush=True,
                )
        # elif isinstance(event, InputRequiredEvent):
        #     ic("Input required event encountered in MHQAActor.respond")
        #     ic(event)
        # elif isinstance(event, HumanResponseEvent):
        #     ic("Human response event encountered in MHQAActor.respond")
        #     ic(event)
        else:
            ...

    # Retrieve the final response from the workflow state
    return full_response


def main():
    result = asyncio.run(
        init_and_chat(
            "test_actor",
            "Watson borrowed 100 Euros from Holmes, yesterday, in Paris. Upon returning to London today, how much does Watson owe Holmes in pounds?",
            # "Hi there, the name's Sherlock!"
        )
    )
    print(result)


if __name__ == "__main__":
    main()
