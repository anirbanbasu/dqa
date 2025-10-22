import asyncio
from enum import StrEnum, auto
import json
import logging
from typing import Any, Dict


from dqa import ic
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)

from llama_index.core.agent.workflow import (
    CodeActAgent,
    FunctionAgent,
    ReActAgent,
)

from llama_index.core.agent.workflow import (
    AgentStream,
)

logger = logging.getLogger(__name__)


class WorkflowStatusEvent(Event):
    def __init__(self, msg: str):
        super().__init__()
        self.msg = msg


class MHQAFlowAgentType(StrEnum):
    DECOMPOSER = auto()
    RESPONDER = auto()
    REASONER = auto()
    REVIEWER = auto()


class DecompositionResultEvent(Event):
    user_message: str
    statement: bool
    sub_questions: list[str] | None
    acknowledgement: str | None


class MHQAFlow(Workflow):
    """Multi-Hop Question Answering Workflow.

    This workflow is designed to handle multi-hop question answering tasks by
    breaking down complex questions into simpler sub-questions, retrieving relevant
    information for each sub-question, and then reviewing and synthesizing the final answer.
    """

    KEY_USER_MSG = "user_msg"

    def __init__(
        self,
        agents: Dict[MHQAFlowAgentType, FunctionAgent | ReActAgent | CodeActAgent],
        *args: Any,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if len(agents) != len(MHQAFlowAgentType):
            raise ValueError(
                f"Agents for all {len(MHQAFlowAgentType)} agent types must be provided."
            )
        for agent_type in MHQAFlowAgentType:
            if agent_type not in agents:
                raise ValueError(f"Agent for {agent_type} must be provided.")
        self.agents = agents

    @step
    async def decompose(
        self, event: StartEvent, context: Context
    ) -> DecompositionResultEvent | StopEvent:
        if hasattr(event, MHQAFlow.KEY_USER_MSG):
            user_msg = getattr(event, MHQAFlow.KEY_USER_MSG)
            # We need it later?
            await context.store.set(MHQAFlow.KEY_USER_MSG, user_msg)
            ic(self.agents[MHQAFlowAgentType.DECOMPOSER])

            wf_handler = self.agents[MHQAFlowAgentType.DECOMPOSER].run(
                user_msg=user_msg,
                # ctx=context,
                # TODO: Pass the memory -- how?
            )
            result: dict = {}
            streamed_result: str = ""
            try:
                async for nested_ev in wf_handler.stream_events():
                    context.write_event_to_stream(nested_ev)
                    if isinstance(nested_ev, AgentStream):
                        streamed_result += str(nested_ev.delta)
                result = json.loads(streamed_result)
            except asyncio.CancelledError as err:
                logger.error(f"Decomposer workflow was cancelled. {err}")
            except Exception as ex:
                logger.error(f"Error in decomposer workflow. {ex}")

            if "statement" in result and result["statement"]:
                ic("it is a statement")
                return StopEvent(result=str(result))
            else:
                ic("it is a question")
                return DecompositionResultEvent.model_validate(result)
        else:
            return StopEvent(
                result="No user message provided through the parameter `user_msg`. Try again!"
            )
        ...

    @step
    async def retrieve_answers(
        self, event: DecompositionResultEvent, context: Context
    ) -> Event | StopEvent:
        context.write_event_to_stream(event)
        return StopEvent(result="Not implemented yet.")

    @step
    async def reason_over_answers(self, event: Event, context: Context) -> Event: ...

    @step
    async def review_and_synthesize(
        self, event: Event, context: Context
    ) -> StopEvent: ...
