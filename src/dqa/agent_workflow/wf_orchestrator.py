import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncIterable, Dict

from pydantic_graph import Graph

from dqa import ic

from rich.console import Console
from rich.markdown import Markdown

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
    ModelMessagesTypeAdapter,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.toolsets.fastmcp import FastMCPToolset
from dqa import EnvVars
from dqa.agent_workflow.mhqa_workflow import Respond, ResponseState, Review

logger = logging.getLogger(__name__)


class MHQAWorkflowOrchestrator:
    RESPONDER = "Responder"
    REVIEWER = "Reviewer"

    def __init__(self):
        self.initialised = False

    def initialise(self, actor_id: str):
        if not hasattr(self, "llm_config"):
            self.llm_config = {}
            llm_config_file = EnvVars.LLM_CONFIG_FILE
            if os.path.exists(llm_config_file):
                with open(llm_config_file, "r") as f:
                    self.llm_config = json.load(f)
        if not hasattr(self, "mcp_config"):
            self.mcp_config = {}
            try:
                mcp_config_file = EnvVars.MCP_CONFIG_FILE
                if os.path.exists(mcp_config_file):
                    with open(mcp_config_file, "r") as f:
                        self.mcp_config = json.load(f)
                    logger.info(
                        f"Loaded {len(self.mcp_config.get('mcpServers', {}))} MCP configurations from {mcp_config_file}"
                    )
            except Exception as e:
                logger.error(f"Error parsing MCP config. {e}")

        if EnvVars.WORKFLOW_SINGLE_AGENT_MODE:
            logger.info("Initialising workflow in single agent mode.")
            responder_llm_config = self.llm_config.get(
                MHQAWorkflowOrchestrator.RESPONDER.lower(), {}
            )
            self.single_agent = Agent(
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
                    "Your final response to the user must always be in correct Markdown format.\n"
                ),
                retries=EnvVars.APP_DAPR_ACTOR_RETRY_ATTEMPTS,
                toolsets=[FastMCPToolset(self.mcp_config)],
            )
        else:
            logger.info("Initialising workflow in multi-agent mode.")
            # self.decomposer_agent = FunctionAgent(
            #     name=MHQAFlowAgentType.DECOMPOSER.value,
            #     description="Decomposes complex questions into simpler sub-questions.",
            #     system_prompt=(
            #         f"You are the {MHQAFlowAgentType.DECOMPOSER.value} agent. "
            #         "Determine if the user query is a question or a statement.\n"
            #         "If the user query is a simple question then decompose it into just one single sub-question, which is the question posed by the user."
            #         "Otherwise, respond with a list of decomposed distinct sub-questions in a format as shown in the example below.\n"
            #         # "The decomposed sub-questions must be precise and as necessary to answer the original question.\n" \
            #         "\n<example-decomposition>\n"
            #         '{"user_message": "The user message exactly as you received it."\n'
            #         '"statement": false\n'
            #         '"sub_questions": [\n'
            #         '  "Sub-question 1",\n'
            #         '  "Sub-question 2",\n'
            #         "  ...\n"
            #         "]}\n"
            #         "</example-decomposition>\n"
            #         "If the user input is only a statement but not a question then respond with an acknowledgment only, in the format shown below.\n"
            #         "\n<example-acknowledgement>\n"
            #         '{"user_message": "The user message exactly as you received it."\n'
            #         '"statement": true\n'
            #         '"acknowledgement": "Your acknowledgement to the user message."\n'
            #         "</example-acknowledgement>\n"
            #     ),
            #     llm=Ollama(
            #         **self.llm_config.get(
            #             MHQAFlowAgentType.DECOMPOSER.value.lower(), None
            #         )
            #     ),
            # )

            # self.responder_agent = FunctionAgent(
            #     name=MHQAWorkflowOrchestrator.RESPONDER,
            #     description="Gathers evidences for sub-questions using available tools.",
            #     system_prompt=(
            #         f"You are the {MHQAWorkflowOrchestrator.RESPONDER} agent.\n"
            #         # f"You would receive a list of questions by the {MHQAAgentWorkflowOrchestrator.DECOMPOSER} agent. "
            #         "Call one or more of the tools provided to you to gather evidences/responses for each question given to you. "
            #         "Generate a list of tool call responses. "
            #         "Store in state the responses you have gathered for each question. "
            #         f"After processing the questions given to you, NEVER respond to the user directly but always hand off to the {MHQAWorkflowOrchestrator.REASONER} agent.\n"
            #     ),
            #     can_handoff_to=[MHQAWorkflowOrchestrator.REASONER],
            #     tools=self.mcp_features,
            #     llm=Ollama(
            #         **self.llm_config[MHQAWorkflowOrchestrator.RESPONDER.lower()]
            #     ),
            # )

            # self.reasoner_agent = ReActAgent(
            #     name=MHQAWorkflowOrchestrator.REASONER,
            #     description="Reasons through gathered evidences to formulate a combined response.",
            #     system_prompt=(
            #         f"You are the {MHQAWorkflowOrchestrator.REASONER} agent.\n"
            #         # f"You would receive from the {MHQAAgentWorkflowOrchestrator.RESPONDER} agent a list of questions and the evidences it gathered for each question. "
            #         "Reason through the evidences for the individual questions and combine them into a single response that serves as a coherent answer to the original user query.\n"
            #         f"NEVER respond to the user directly but always hand off to the {MHQAWorkflowOrchestrator.REVIEWER} agent for a review of your combined response.\n"
            #     ),
            #     can_handoff_to=[MHQAWorkflowOrchestrator.REVIEWER],
            #     llm=Ollama(
            #         **self.llm_config[MHQAWorkflowOrchestrator.REASONER.lower()]
            #     ),
            # )

            # self.reviewer_agent = ReActAgent(
            #     name=MHQAWorkflowOrchestrator.REVIEWER,
            #     description="Reviews the combined response for quality and completeness; and responds to the user.",
            #     system_prompt=(
            #         f"You are the {MHQAWorkflowOrchestrator.REVIEWER} agent.\n"
            #         # f"You would receive from the {MHQAAgentWorkflowOrchestrator.REASONER} agent a response to the user question. "
            #         "Review the response in terms of quality and highlight any potential gaps. "
            #         f"Respond to the user with the {MHQAWorkflowOrchestrator.REASONER} agent response and your review. "
            #         "Make sure that your final response is in correct Markdown format.\n"
            #     ),
            #     llm=Ollama(
            #         **self.llm_config[MHQAWorkflowOrchestrator.REVIEWER.lower()]
            #     ),
            # )

            # self.workflow = MHQAFlow(
            #     agents={
            #         MHQAFlowAgentType.DECOMPOSER: self.decomposer_agent,
            #         MHQAFlowAgentType.RESPONDER: self.responder_agent,
            #         MHQAFlowAgentType.REASONER: self.reasoner_agent,
            #         MHQAFlowAgentType.REVIEWER: self.reviewer_agent,
            #     },
            #     verbose=True,
            # )

        self.initialised = (
            (hasattr(self, "mcp_config") and hasattr(self, "single_agent"))
            if EnvVars.WORKFLOW_SINGLE_AGENT_MODE
            else (
                hasattr(self, "mcp_features")
                and hasattr(self, "workflow")
                and hasattr(self, "workflow_context")
                and hasattr(self, "workflow_memory")
            )
        )

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
    # orchestrator = MHQAWorkflowOrchestrator()
    # await orchestrator.initialise(actor_id=actor_id)
    # if not orchestrator.initialised:
    #     raise RuntimeError("Orchestrator failed to initialise properly.")

    # Run the workflow
    if EnvVars.WORKFLOW_SINGLE_AGENT_MODE:
        output_messages: list[str] = []

        async def handle_event(event: AgentStreamEvent):
            if isinstance(event, PartStartEvent):
                if isinstance(event.part, ThinkingPart):
                    print(f"\n[Thinking]\n{event.part.content}", flush=True, end="")
                else:
                    print()
                # output_messages.append(
                #     f"[Request] Starting part {event.index}: {event.part!r}"
                # )
            elif isinstance(event, PartDeltaEvent):
                if isinstance(event.delta, TextPartDelta):
                    output_messages.append(
                        f"[Request] Part {event.index} text delta: {event.delta.content_delta!r}"
                    )
                elif isinstance(event.delta, ThinkingPartDelta):
                    # output_messages.append(
                    #     f"[Request] Part {event.index} thinking delta: {event.delta.content_delta!r}"
                    # )
                    print(f"{event.delta.content_delta}", flush=True, end="")
                elif isinstance(event.delta, ToolCallPartDelta):
                    output_messages.append(
                        f"[Request] Part {event.index} args delta: {event.delta.args_delta}"
                    )
            elif isinstance(event, FunctionToolCallEvent):
                output_messages.append(
                    f"[Tools] The LLM calls tool={event.part.tool_name!r} with args={event.part.args} (tool_call_id={event.part.tool_call_id!r})"
                )
            elif isinstance(event, FunctionToolResultEvent):
                output_messages.append(
                    f"[Tools] Tool call {event.tool_call_id!r} returned => {event.result.content}"
                )
            elif isinstance(event, FinalResultEvent):
                output_messages.append(
                    f"[Result] The model starting producing a final result (tool_name={event.tool_name})"
                )

        async def event_stream_handler(
            ctx: RunContext,
            event_stream: AsyncIterable[AgentStreamEvent],
        ):
            async for event in event_stream:
                await handle_event(event)

        chat_message_history_file = f"agent_run_messages_{actor_id}.json"
        existing_messages = []
        if os.path.exists(chat_message_history_file):
            with open(chat_message_history_file, "r") as f:
                existing_messages = json.load(f)
        # ic(existing_messages)
        message_history = ModelMessagesTypeAdapter.validate_python(existing_messages)
        ic(message_history)
        state = ResponseState(
            user_message=user_query, responder_messages=message_history
        )
        mhqa_graph = Graph(nodes=(Respond, Review))
        result = await mhqa_graph.run(Respond(), state=state)
        return result.output
        # async with orchestrator.single_agent.run_stream(
        #     user_query,
        #     event_stream_handler=event_stream_handler,
        #     message_history=message_history,
        # ) as agent_run:
        #     async for output in agent_run.stream_text():
        #         output_messages.append(f"[Output] {output}")
        # message_memory = to_jsonable_python(agent_run.all_messages())
        # with open(chat_message_history_file, "w") as f:
        #     json.dump(message_memory, f, indent=2)
        # return output_messages
    else:
        ...
        # wfh = orchestrator.workflow.run(
        #     user_msg=user_query,
        #     memory=orchestrator.workflow_memory,
        #     context=orchestrator.workflow_context,
        #     max_iterations=1,
        # )

        # full_response = ""
        # current_agent = ""
        # async for event in wfh.stream_events():
        #     if (
        #         hasattr(event, "current_agent_name")
        #         and event.current_agent_name != current_agent
        #     ):
        #         current_agent = event.current_agent_name
        #         print(f"\n{'=' * 50}", flush=True)
        #         print(f"🤖 Agent: {current_agent}", flush=True)
        #         print(f"{'=' * 50}\n", flush=True)
        #     if isinstance(event, AgentStream):
        #         if event.delta:
        #             print(event.delta, end="", flush=True)
        #             full_response += event.delta
        #     elif isinstance(event, AgentInput):
        #         print(f"\n📥 {event.current_agent_name} Input:", event.input, flush=True)
        #     elif isinstance(event, ToolCall):
        #         print(f"\n🔨 Calling Tool: {event.tool_name}", flush=True)
        #         print(f"  With arguments: {event.tool_kwargs}", flush=True)
        #     elif isinstance(event, ToolCallResult):
        #         print(f"\n🔧 Tool Result ({event.tool_name}):", flush=True)
        #         print(f"  Arguments: {event.tool_kwargs}", flush=True)
        #         print(f"  Output: {event.tool_output}", flush=True)
        #     elif isinstance(event, AgentOutput):
        #         if event.response.content:
        #             print(
        #                 f"\n📤 {event.current_agent_name} Output:",
        #                 event.response.content,
        #                 flush=True,
        #             )
        #         if event.tool_calls:
        #             print(
        #                 "\n🛠️  Planning to use tools:",
        #                 [call.tool_name for call in event.tool_calls],
        #                 flush=True,
        #             )
        #     # elif isinstance(event, InputRequiredEvent):
        #     #     ic("Input required event encountered in MHQAActor.respond")
        #     #     ic(event)
        #     # elif isinstance(event, HumanResponseEvent):
        #     #     ic("Human response event encountered in MHQAActor.respond")
        #     #     ic(event)
        #     else:
        #         ...

        # # Retrieve the final response from the workflow state
        # return full_response


def main():
    result = asyncio.run(
        init_and_chat(
            "test_actor",
            # "Watson borrowed 100 Euros from Holmes on October 27, 2025, in Paris. Upon returning to London today, how much does Watson owe Holmes in pounds based on the rate on the day he borrowed the money?",
            # "Hi there, the name's Sherlock! I mean, I am THE Sherlock Holmes!"
            # "Did I tell you my name?"
            # "Where were Watson and I on October 27, 2025?",
            # "Oh, I am THE Sherlock Holmes! Now, can you confidently tell where I was on October 27, 2025?",
            # "Zoe is 54 years old and her mother is 80, how many years ago was Zoe's mother's age some integer multiple of her age?"
            "What is the current share price of Hitachi (6501.T) at the Tokyo Stock Exchange?",
            # "The Eiffel Tower is located in which city?",
            # "Which David Fincher film that stars Edward Norton does not star Brad Pitt?"
        )
    )
    print("=" * 80)
    console = Console(soft_wrap=True)
    if isinstance(result, list):
        console.print("\n".join(result))
    else:
        console.print(Markdown(result))


if __name__ == "__main__":
    main()
