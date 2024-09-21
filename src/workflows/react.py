# Copyright 2024 Anirban Basu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Variations of the ReAct agent."""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import uuid

# Weaker LLMs may generate horrible JSON strings.
# `dirtyjson` is more lenient than `json` in parsing JSON strings.
from typing import Any


from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)


# from llama_index.core.agent import ReActAgent

from utils import (
    EMPTY_STRING,
)  # , parse_env, EnvironmentVariables


from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.tools import ToolSelection, ToolOutput
from llama_index.core.agent.react import (
    ReActChatFormatter,
    ReActOutputParser,
)
from llama_index.core.agent.react.types import (
    ActionReasoningStep,
    ObservationReasoningStep,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.tools.types import BaseTool

from workflows.common import WorkflowStatusEvent


# ReAct Events
class ReActPrepEvent(Event):
    """Event to prepare the chat history."""

    pass


class ReActInputEvent(Event):
    """
    Event to manage the input to the LLM for the ReAct workflow.

    Fields:
        list[ChatMessage]: The chat history.
    """

    input: list[ChatMessage]


class ReActToolCallEvent(Event):
    """
    Event to call the tools in the ReAct workflow.

    Fields:
        list[ToolSelection]: The selected tools to call.
    """

    tool_calls: list[ToolSelection]


class ReActFunctionOutputEvent(Event):
    """
    Event to handle the output of a tool call in the ReAct workflow.

    Fields:
        ToolOutput: The output of the tool call.
    """

    output: ToolOutput


class ReActWorkflow(Workflow):
    """
    ## A simple ReAct agent

    This workflow implements the ReAct agent described in the paper: https://arxiv.org/abs/2210.03629.

    **Caveats**
     - May go down the wrong route of reasoning with no self-correction.
     - May not call tools correctly with low-parameter LLMs.
     - Output is JSON, hence not formatted for the DQA web UI.
    """

    KEY_CURRENT_REASONING = "current_reasoning"
    KEY_THOUGHT = "thought"
    KEY_ACTION = "action"
    KEY_RESPONSE = "response"
    KEY_REASONING = "reasoning"
    KEY_OBSERVATION = "observation"
    KEY_ERROR = "error"
    KEY_WARNING = "warning"
    KEY_SOURCES = "sources"

    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        extra_context: str | None = None,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ReAct workflow.

        Args:
            llm (LLM): The LLM instance to use.
            tools (list[BaseTool]): The list of tools to use.
            extra_context (str): The extra context to use.
            max_iterations (int): The maximum number of iterations to run.
        """
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm
        self.max_iterations = max_iterations

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.formatter = ReActChatFormatter.from_defaults(
            context=extra_context or EMPTY_STRING,
        )
        self.output_parser = ReActOutputParser()
        self.sources = []

        self._total_steps: int = 0
        self._finished_steps: int = 0

    @step
    async def new_user_msg(self, ctx: Context, ev: StartEvent) -> ReActPrepEvent:
        """
        Step to handle a new user message.

        Args:
            ctx (Context): The context object.
            ev (StartEvent): The start event.

        Returns:
            ReActPrepEvent: The event to prepare the chat history.
        """
        # clear sources
        self.sources = []
        self._current_iteration = 0
        self._total_steps += 1

        # get user input
        user_input = ev.input

        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Handling input: {user_input}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self.memory.put(user_msg)

        # clear current reasoning
        await ctx.set(ReActWorkflow.KEY_CURRENT_REASONING, [])

        self._finished_steps += 1

        return ReActPrepEvent()

    @step
    async def prepare_chat_history(
        self, ctx: Context, ev: ReActPrepEvent
    ) -> ReActInputEvent:
        """
        Step to prepare the chat history for the ReAct paradigm.

        Args:
            ctx (Context): The context object.
            ev (ReActPrepEvent): The event to prepare the chat history.

        Returns:
            ReActInputEvent: The event containing the input to be used for the LLM.
        """
        self._total_steps += 1
        # get chat history
        self._current_iteration += 1
        if self._current_iteration > self.max_iterations:
            self._finished_steps += 1
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg="Stopping because maximum number of iterations have been reached.",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )
            return StopEvent(
                result={
                    ReActWorkflow.KEY_RESPONSE: f"I must stop because I have reached the specified maximum number of iterations ({self.max_iterations}).",
                    ReActWorkflow.KEY_SOURCES: [*self.sources],
                    ReActWorkflow.KEY_REASONING: await ctx.get(
                        ReActWorkflow.KEY_CURRENT_REASONING, default=[]
                    ),
                }
            )
        chat_history = self.memory.get()
        current_reasoning = await ctx.get(
            ReActWorkflow.KEY_CURRENT_REASONING, default=[]
        )
        llm_input = self.formatter.format(
            self.tools, chat_history, current_reasoning=current_reasoning
        )
        self._finished_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Initialised LLM call with {len(chat_history)} messages in chat history.",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        return ReActInputEvent(input=llm_input)

    @step
    async def handle_llm_input(
        self, ctx: Context, ev: ReActInputEvent
    ) -> ReActToolCallEvent | StopEvent:
        """
        Step to query the LLM.

        Args:
            ctx (Context): The context object.
            ev (ReActInputEvent): The event containing the input to the LLM.

        Returns:
            ReActToolCallEvent | StopEvent: The event to call the tools or the event to stop the workflow.
        """
        chat_history = ev.input
        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Calling the LLM with {len(chat_history)} messages in chat history.",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        response = await self.llm.achat(chat_history)
        self._finished_steps += 1

        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                reasoning_step
            )
            streaming_message = EMPTY_STRING
            if hasattr(reasoning_step, ReActWorkflow.KEY_THOUGHT):
                streaming_message = f"{ReActWorkflow.KEY_THOUGHT.capitalize()}: {reasoning_step.thought}"
            if hasattr(reasoning_step, ReActWorkflow.KEY_ACTION):
                streaming_message += f"\n{ReActWorkflow.KEY_ACTION.capitalize()}: {reasoning_step.action} with {reasoning_step.action_input}"
            if reasoning_step.is_done:
                streaming_message += f"\n{ReActWorkflow.KEY_RESPONSE.capitalize()}: {reasoning_step.response}"
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=streaming_message,
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )
            if reasoning_step.is_done:
                self.memory.put(
                    ChatMessage(
                        role=MessageRole.ASSISTANT, content=reasoning_step.response
                    )
                )
                return StopEvent(
                    result={
                        ReActWorkflow.KEY_RESPONSE: reasoning_step.response,
                        ReActWorkflow.KEY_SOURCES: [*self.sources],
                        ReActWorkflow.KEY_REASONING: await ctx.get(
                            ReActWorkflow.KEY_CURRENT_REASONING, default=[]
                        ),
                    }
                )
            elif isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                return ReActToolCallEvent(
                    tool_calls=[
                        ToolSelection(
                            tool_id=f"{tool_name}-{uuid.uuid4()}",
                            tool_name=tool_name,
                            tool_kwargs=tool_args,
                        )
                    ]
                )
        except Exception as e:
            (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                ObservationReasoningStep(
                    observation=f"There was an error in parsing my reasoning: {e}"
                )
            )
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=f"{ReActWorkflow.KEY_ERROR.capitalize()}: {e}",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )

        # if no tool calls or final response, iterate again
        return ReActPrepEvent()

    @step
    async def handle_tool_calls(
        self, ctx: Context, ev: ReActToolCallEvent
    ) -> ReActPrepEvent:
        """
        Step to call the tools.

        Args:
            ctx (Context): The context object.
            ev (ReActToolCallEvent): The event containing the tool calls.

        Returns:
            ReActPrepEvent: The event to prepare for the next iteration of the ReAct paradigm.
        """

        self._total_steps += 1
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Making tool calls from {len(tool_calls)} selected tools.",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist."
                    )
                )
                ctx.write_event_to_stream(
                    WorkflowStatusEvent(
                        msg=f"{ReActWorkflow.KEY_WARNING.capitalize()}: Tool {tool_call.tool_name} does not exist.",
                        total_steps=self._total_steps,
                        finished_steps=self._finished_steps,
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
                self._finished_steps += 1
                ctx.write_event_to_stream(
                    WorkflowStatusEvent(
                        msg=f"{ReActWorkflow.KEY_OBSERVATION.capitalize()}: {tool_output.content}",
                        total_steps=self._total_steps,
                        finished_steps=self._finished_steps,
                    )
                )
            except Exception as e:
                (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                    ObservationReasoningStep(
                        observation="Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )
                self._finished_steps += 1
                ctx.write_event_to_stream(
                    WorkflowStatusEvent(
                        msg=f"{ReActWorkflow.KEY_ERROR.capitalize()}: Failed calling tool {tool.metadata.get_name()}: {e}",
                        total_steps=self._total_steps,
                        finished_steps=self._finished_steps,
                    )
                )

        # prep the next iteraiton
        return ReActPrepEvent()
