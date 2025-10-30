import asyncio
import json
import logging
import os
import re
from typing import Any, Dict

from pydantic_core import to_jsonable_python
from pydantic_graph import Graph

from dqa import ic

from rich.console import Console
from rich.markdown import Markdown

from prefect import flow

from pydantic_ai import (
    ModelMessagesTypeAdapter,
)
from dqa import EnvVars
from dqa.agent_workflow.mhqa_workflow import Respond, ResponseState, Review

logger = logging.getLogger(__name__)


def conditional_flow(decorator, condition: bool, *args, **kwargs):
    """
    Apply a flow decorator conditionally based on the given condition.
    If the condition is True, apply the decorator; otherwise, return the function unchanged.
    """
    if condition:
        return decorator(*args, **kwargs)
    return lambda fn: fn


class MHQAWorkflowOrchestrator:
    def __init__(self):
        self.initialised = False

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


@conditional_flow(
    flow,
    (EnvVars.PREFECT_API_URL is not None),
    name="mhqa_test_chat_flow",
    log_prints=False,
)
async def init_and_chat(actor_id: str, user_query: str):
    # orchestrator = MHQAWorkflowOrchestrator()
    # await orchestrator.initialise(actor_id=actor_id)
    # if not orchestrator.initialised:
    #     raise RuntimeError("Orchestrator failed to initialise properly.")

    # Run the workflow

    chat_message_history_file = f"agent_run_messages_{actor_id}.json"
    existing_messages = []
    if os.path.exists(chat_message_history_file):
        with open(chat_message_history_file, "r") as f:
            existing_messages = json.load(f)
    # ic(existing_messages)
    message_history = ModelMessagesTypeAdapter.validate_python(existing_messages)
    ic(message_history)
    state = ResponseState(user_message=user_query, responder_messages=message_history)
    mhqa_graph = Graph(nodes=(Respond, Review))
    result = await mhqa_graph.run(Respond(), state=state)
    message_memory = to_jsonable_python(result.state.responder_messages)
    with open(chat_message_history_file, "w") as f:
        json.dump(message_memory, f, indent=2)
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


def main():
    result = asyncio.run(
        init_and_chat(
            "test_actor",
            "Watson borrowed 100 Euros from Holmes on October 27, 2025, in Paris. Upon returning to London today, how much does Watson owe Holmes in pounds based on the rate on the day he borrowed the money?",
            # "Hi there, the name's Sherlock! I mean, I am THE Sherlock Holmes!"
            # "Did I tell you my name?"
            # "Where were Watson and I on October 27, 2025?",
            # "Oh, I am THE Sherlock Holmes! Now, can you confidently tell where I was on October 27, 2025?",
            # "Zoe is 54 years old and her mother is 80, how many years ago was Zoe's mother's age some integer multiple of her age?"
            # "What is the current share price of Hitachi (6501.T) at the Tokyo Stock Exchange?",
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
