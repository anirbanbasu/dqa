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

"""Difficult Questions Attempted core module"""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import asyncio
import uuid

# Weaker LLMs may generate horrible JSON strings.
# `dirtyjson` is more lenient than `json` in parsing JSON strings.
import dirtyjson as json
from typing import Any, List
from llama_index.tools.arxiv import ArxivToolSpec

from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.tools import FunctionTool

from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)

# from llama_index.core.agent import ReActAgent

from tools import StringFunctionsToolSpec, BasicArithmeticCalculatorSpec
from utils import (
    EMPTY_STRING,
    FAKE_STRING,
    ToolNames,
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


class ReActPrepEvent(Event):
    pass


class ReActInputEvent(Event):
    input: list[ChatMessage]


class ReActToolCallEvent(Event):
    tool_calls: list[ToolSelection]


class ReActFunctionOutputEvent(Event):
    output: ToolOutput


class ReActWorkflow(Workflow):
    """A workflow implementation for a ReAct agent."""

    KEY_CURRENT_REASONING = "current_reasoning"
    KEY_THOUGHT = "thought"
    KEY_ACTION = "action"
    KEY_RESPONSE = "response"
    KEY_REASONING = "reasoning"
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
        self.formatter = ReActChatFormatter(context=extra_context or EMPTY_STRING)
        self.output_parser = ReActOutputParser()
        self.sources = []

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

        # get user input
        user_input = ev.input
        user_msg = ChatMessage(role=MessageRole.USER, content=user_input)
        self.memory.put(user_msg)

        # clear current reasoning
        await ctx.set(ReActWorkflow.KEY_CURRENT_REASONING, [])

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
        # get chat history
        self._current_iteration += 1
        if self._current_iteration > self.max_iterations:
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

        generator = await self.llm.astream_chat(chat_history)
        async for response in generator:
            pass

        try:
            reasoning_step = self.output_parser.parse(response.message.content)
            (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                reasoning_step
            )
            streaming_message = EMPTY_STRING
            if hasattr(reasoning_step, ReActWorkflow.KEY_THOUGHT):
                streaming_message = f"\n{ReActWorkflow.KEY_THOUGHT.capitalize()}: {reasoning_step.thought}"
            if hasattr(reasoning_step, ReActWorkflow.KEY_ACTION):
                streaming_message += f"\n{ReActWorkflow.KEY_ACTION.capitalize()}: {reasoning_step.action} with {reasoning_step.action_input}"
            if reasoning_step.is_done:
                streaming_message += f"\n{ReActWorkflow.KEY_RESPONSE.capitalize()}: {reasoning_step.response}"
            ctx.write_event_to_stream(Event(msg=streaming_message))
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
                Event(msg=f"There was an error in parsing my reasoning: {e}")
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
        tool_calls = ev.tool_calls
        tools_by_name = {tool.metadata.get_name(): tool for tool in self.tools}

        # call tools -- safely!
        for tool_call in tool_calls:
            tool = tools_by_name.get(tool_call.tool_name)
            if not tool:
                (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Tool {tool_call.tool_name} does not exist"
                    )
                )
                ctx.write_event_to_stream(
                    Event(msg=f"Tool {tool_call.tool_name} does not exist")
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
                ctx.write_event_to_stream(
                    Event(
                        msg=f"Observation: {tool_output.content}",
                    )
                )
            except Exception as e:
                (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                    ObservationReasoningStep(
                        observation=f"Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )
                ctx.write_event_to_stream(
                    Event(msg=f"Error calling tool {tool.metadata.get_name()}: {e}")
                )

        # prep the next iteraiton
        return ReActPrepEvent()


class DQAQueryEvent(Event):
    question: str


class DQAAnswerEvent(Event):
    question: str
    answer: str
    sources: List[Any] = []


class DQAReviewSubQuestionEvent(Event):
    questions: List[str]
    satisfied: bool = False


class DQAStatusEvent(Event):
    msg: str
    total_steps: int
    finished_steps: int


class DQAWorkflow(Workflow):
    """A workflow implementation for the DQA paradigm."""

    KEY_ORIGINAL_QUERY = "original_query"

    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the DQA workflow.

        Args:
            llm (LLM): The LLM instance to use.
            tools (list[BaseTool]): The list of tools to use.
        """
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm

        self._total_steps: int = 0
        self._finished_steps: int = 0

    @step
    async def query(self, ctx: Context, ev: StartEvent) -> DQAQueryEvent | StopEvent:
        """
        As a start event of the workflow, this step receives the original query and stores it in the context.
        It then asks the LLM to decompose the query into sub-questions. Upon decomposition, it emits every
        sub-question as a `DQAQueryEvent`.

        Args:
            ctx (Context): The context object.
            ev (StartEvent): The start event.

        Returns:
            DQAQueryEvent | StopEvent: The event containing the sub-question or the event to stop the workflow if there is no query specified.
        """
        if hasattr(ev, "query"):
            ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY] = ev.query
        else:
            return StopEvent(result="No query provided. Try again!")

        self._total_steps += 1
        ctx.write_event_to_stream(
            DQAStatusEvent(
                msg=f"Assessing query:\n\n{ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are a linguistic expert who performs query decomposition."
            "\nGiven a user question, generate a list of distinct sub-questions that you need to answer in order to answer the original question. "
            "Respond with a list containing the unmodified original question only when no decomposition is needed. Otherwise, do not include the original question in the list of sub-questions. "
            "Generate sub-questions that explicitly mention the subject by name, avoiding pronouns like 'these,' 'they,' 'he,' 'she,' 'it,', and so on. "
            "Each sub-question should clearly state the subject to ensure no ambiguity. "
            "Do not generate sub-questions that are not required to answer the original question. "
            "\n\nLastly, reflect on the generated sub-questions and output a binary response indicating whether you are satisfied with the generated sub-questions or not. "
            "\n\nExample 1:\n"
            "Question: Is Hamlet more common on IMDB than Comedy of Errors?\n"
            "Decompositions:\n"
            "{\n"
            '    "sub_questions": [\n'
            '        "How many listings of Hamlet are there on IMDB?",\n'
            '        "How many listings of Comedy of Errors is there on IMDB?"\n'
            "    ],\n"
            '    "satisfied": true\n'
            "}"
            "\n\nExample 2:\n"
            "Question: What is the capital city of Japan?\n"
            "Decompositions:\n"
            "{\n"
            '    "sub_questions": ["What is the capital city of Japan?"],\n'
            '    "satisfied": true\n'
            "}\n"
            "Note that this question above needs no decomposition. Hence, the original question is repeated as the only sub-question."
            "\n\nExample 3:\n"
            "Question: Are there more hydrogen atoms in methyl alcohol than in ethyl alcohol?\n"
            "Decompositions:\n"
            "{\n"
            '    "sub_questions": [\n'
            '        "How many hydrogen atoms are there in methyl alcohol?",\n'
            '        "How many hydrogen atoms are there in ethyl alcohol?",\n'
            '        "What is the chemical composition of alcohol?"\n'
            "    ],\n"
            '    "satisfied": false\n'
            "}\n"
            "Note that the third sub-question is unnecessary and should not be included. Hence, the value of the satisfied flag is set to false."
            "\n\nAlways, respond in pure JSON without any Markdown, like this:\n"
            "{\n"
            '    "sub_questions": [\n'
            '        "sub question 1",\n'
            '        "sub question 2",\n'
            '        "sub question 3"\n'
            "    ],\n"
            '    "satisfied": true or false\n'
            "}"
            f"\n\nHere is the user question: {ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}"
        )
        generator = await self.llm.astream_complete(prompt)

        async for response in generator:
            pass

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]
        satisfied = response_obj["satisfied"]

        ctx.write_event_to_stream(
            DQAStatusEvent(
                msg=f"{'Satisfactory' if satisfied else 'Unsatisfactory'} sub-questions:\n\n{str(sub_questions)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        ctx.data["sub_question_count"] = len(sub_questions)

        if len(sub_questions) == 1 and satisfied:
            return DQAQueryEvent(question=sub_questions[0])
        else:
            if satisfied:
                for question in sub_questions:
                    ctx.send_event(DQAQueryEvent(question=question))
            else:
                return DQAReviewSubQuestionEvent(
                    questions=sub_questions, satisfied=satisfied
                )

    @step
    async def review_sub_questions(
        self, ctx: Context, ev: DQAReviewSubQuestionEvent
    ) -> DQAQueryEvent | DQAReviewSubQuestionEvent:
        """
        This step receives the sub-questions and asks the LLM to review them. If the LLM is satisfied with the
        sub-questions, they can be used to answer the original question. Otherwise, the LLM can provide updated
        sub-questions.

        Args:
            ctx (Context): The context object.
            ev (DQAQueryEvent): The event containing the sub-question.

        Returns:
            DQAQueryEvent | DQAReviewSubQuestionEvent: The event containing the sub-question or the event to review the
            sub-questions.
        """

        if ev.satisfied:
            # Already satisfied, no need to review anymore.
            for question in ev.questions:
                ctx.send_event(DQAQueryEvent(question=question))

        self._total_steps += 1
        self._finished_steps += 1
        ctx.write_event_to_stream(
            DQAStatusEvent(
                msg=f"Reviewing sub-questions:\n\n{str(ev.questions)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are a linguistic expert who performs a review of query decomposition."
            "\nYou are given an overall question that has been decomposed into sub-questions, which are also given below. "
            "Review each sub-questions and improve it, if necessary. Remove any sub-questions that is not required to answer the original query."
            "Do not add new sub-questions, unless necessary. Remember that the sub-questions represent a concise decomposition of the original question. "
            "\nEnsure that sub-questions explicitly mention the subject by name, avoiding pronouns like 'these,' 'they,' 'he,' 'she,' 'it,', and so on. "
            "Each sub-question should clearly state the subject to ensure no ambiguity. "
            "Do not output sub-questions that are not required to answer the original question. "
            "\n\nLastly, reflect on the amended generate a binary response indicating whether you are satisfied with the amended sub-questions or not."
            "\n\nAlways, respond in pure JSON without any Markdown, like this:"
            "{\n"
            '    "sub_questions": [\n'
            '        "sub question 1",\n'
            '        "sub question 2",\n'
            '        "sub question 3"\n'
            "    ],\n"
            '    "satisfied": true or false\n'
            "}"
            f"\n\nHere is the user question: {ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}"
            f"\n\nHere are the sub-questions for you to review:\n{ev.questions}"
        )
        generator = await self.llm.astream_complete(prompt)

        async for response in generator:
            pass
        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]
        satisfied = response_obj["satisfied"]

        ctx.write_event_to_stream(
            DQAStatusEvent(
                msg=f"{'Satisfactory' if satisfied else 'Unsatisfactory'} refined sub-questions:\n\n{str(sub_questions)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        ctx.data["sub_question_count"] = len(sub_questions)

        if len(sub_questions) == 1 and satisfied:
            return DQAQueryEvent(question=sub_questions[0])

        if satisfied:
            for question in sub_questions:
                ctx.send_event(DQAQueryEvent(question=question))
        else:
            # Not satisfied, so ask for review again.
            return DQAReviewSubQuestionEvent(
                questions=sub_questions, satisfied=satisfied
            )

        return None

    @step(num_workers=4)
    async def answer_sub_question(
        self, ctx: Context, ev: DQAQueryEvent
    ) -> DQAAnswerEvent:
        """
        This step receives a sub-question and attempts to answer it using the tools provided in the context.

        Args:
            ctx (Context): The context object.
            ev (DQAQueryEvent): The event containing the sub-question.

        Returns:
            DQAAnswerEvent: The event containing the sub-question and the answer.
        """

        self._total_steps += 1
        self._finished_steps += 1
        ctx.write_event_to_stream(
            DQAStatusEvent(
                msg=f"Starting a {ReActWorkflow.__name__} to answer question:\n\n{ev.question}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        react_workflow = ReActWorkflow(
            llm=self.llm,
            tools=self.tools,
            # Let's set the timeout of the ReAct workflow to half of the DQA workflow's timeout.
            timeout=self._timeout / 2,
            verbose=self._verbose,
            # Let's set the maximum iterations of the ReAct workflow to its default value.
        )
        react_task = asyncio.create_task(react_workflow.run(input=ev.question))

        async for nested_ev in react_workflow.stream_events():
            self._total_steps += 1
            self._finished_steps += 1
            ctx.write_event_to_stream(
                DQAStatusEvent(
                    msg=f"[{ReActWorkflow.__name__}] {nested_ev.msg}",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )

        response = await react_task

        return DQAAnswerEvent(
            question=ev.question,
            answer=response[ReActWorkflow.KEY_RESPONSE],
            sources=[
                tool_output.content
                for tool_output in response[ReActWorkflow.KEY_SOURCES]
            ],
        )

    @step
    async def combine_refine_answers(
        self, ctx: Context, ev: DQAAnswerEvent
    ) -> StopEvent | None:
        """
        This step receives the answers to the sub-questions and combines them into a single answer to the original
        question so long as all the sub-questions have been answered. If there was only one sub-question, then the
        answer to that sub-question is refined before it is finally returned.

        Args:
            ctx (Context): The context object.
            ev (DQAAnswerEvent): The event containing the sub-question and the answer.

        Returns:
            StopEvent | None: The event containing the final answer to the original question, or None if the sub-questions
            have not all been answered.
        """
        ready = ctx.collect_events(
            ev, [DQAAnswerEvent] * ctx.data["sub_question_count"]
        )
        if ready is None:
            return None

        answers = "\n\n".join(
            [
                (
                    f"Question: {event.question}"
                    f"\nAnswer: {event.answer}"
                    f"\nSources: {', '.join(event.sources)}"
                )
                for event in ready
            ]
        )

        self._finished_steps += 1
        ctx.write_event_to_stream(
            DQAStatusEvent(
                msg=f"Generating the final response to the original query:\n\n{ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are given an overall question that has been split into sub-questions, each of which has been answered."
            "\nCombine the answers to all the sub-questions into a single and coherent response to the original question. "
            "Ensure that your final answer includes all the relevant details and nuances from the answers to the sub-questions. "
            "In your final answer, cite the sources and their corresponding URLs, if source URLs are available are in the answers to the sub-questions."
            "\nDo not make up sources or URLs if they are not present in the answers to the sub-questions."
            "\nYour final answer must be correctly formatted as HTML in a concise, readable and visually pleasing way. "
            "Enclose the HTML with a <div> tag that has an attribute `id` set to the value 'dqa_workflow_response'."
            f"\n\nOriginal question: {ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}"
            f"\n\nSub-questions and answers:\n{answers}"
        )

        generator = await self.llm.astream_complete(prompt)
        async for response in generator:
            pass

        return StopEvent(result=str(response))


class DQAEngine:
    def __init__(self, llm: LLM | None = None):
        """
        Initialize the Difficult Questions Attempted engine.

        Args:
            llm (LLM): The function calling LLM instance to use.
        """
        self.llm = llm
        # Add tool specs
        self.tools: List[FunctionTool] = []
        # Mandatory tools
        self.tools.extend(StringFunctionsToolSpec().to_tool_list())
        self.tools.extend(BasicArithmeticCalculatorSpec().to_tool_list())

        # TODO: Populate the tools based on toolset names specified in the environment variables?
        self.tools.extend(ArxivToolSpec().to_tool_list())
        self.tools.extend(DuckDuckGoSearchToolSpec().to_tool_list())

    def _are_tools_present(self, tool_names: list[str]) -> bool:
        """
        Check if the tools with the given names are present in the current set of tools.

        Args:
            tool_names (list[str]): The names of the tools to check.

        Returns:
            bool: True if all the tools are present, False otherwise.
        """
        return all(
            tool_name in [tool.metadata.name for tool in self.tools]
            for tool_name in tool_names
        )

    def _remove_tools_by_names(self, tool_names: list[str]):
        """
        Remove the tools with the given names from the current set of tools.

        Args:
            tool_names (list[str]): The names of the tools to remove.
        """
        self.tools = [
            tool for tool in self.tools if tool.metadata.name not in tool_names
        ]

    def is_toolset_present(self, toolset_name: str) -> bool:
        """
        Check if the tools for the given toolset are present in the current set of tools.

        Args:
            toolset_name (str): The name of the toolset to check.

        Returns:
            bool: True if the tools are present, False otherwise.
        """
        if toolset_name == ToolNames.TOOL_NAME_ARXIV:
            return self._are_tools_present(
                [tool.metadata.name for tool in ArxivToolSpec().to_tool_list()]
            )
        elif toolset_name == ToolNames.TOOL_NAME_BASIC_ARITHMETIC_CALCULATOR:
            return self._are_tools_present(
                [
                    tool.metadata.name
                    for tool in BasicArithmeticCalculatorSpec().to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_DUCKDUCKGO:
            return self._are_tools_present(
                [
                    tool.metadata.name
                    for tool in DuckDuckGoSearchToolSpec().to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_STRING_FUNCTIONS:
            return self._are_tools_present(
                [
                    tool.metadata.name
                    for tool in StringFunctionsToolSpec().to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_TAVILY:
            return self._are_tools_present(
                [
                    tool.metadata.name
                    for tool in TavilyToolSpec(api_key=FAKE_STRING).to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_WIKIPEDIA:
            return self._are_tools_present(
                [tool.metadata.name for tool in WikipediaToolSpec().to_tool_list()]
            )
        elif toolset_name == ToolNames.TOOL_NAME_YAHOO_FINANCE:
            return self._are_tools_present(
                [tool.metadata.name for tool in YahooFinanceToolSpec().to_tool_list()]
            )

    def get_selected_web_search_toolset(self) -> str:
        """
        Get the name of the web search toolset currently selected.

        Returns:
            str: The name of the web search toolset.
        """
        if self.is_toolset_present(ToolNames.TOOL_NAME_DUCKDUCKGO):
            return ToolNames.TOOL_NAME_DUCKDUCKGO
        elif self.is_toolset_present(ToolNames.TOOL_NAME_TAVILY):
            return ToolNames.TOOL_NAME_TAVILY
        else:
            # Unknown or no toolset selected.
            return ToolNames.TOOL_NAME_SELECTION_DISABLE

    def remove_toolset(self, toolset_name: str):
        """
        Remove the tools for the given toolset from the current set of tools.

        Args:
            toolset_name (str): The name of the toolset to remove.
        """
        if toolset_name == ToolNames.TOOL_NAME_ARXIV:
            self._remove_tools_by_names(
                [tool.metadata.name for tool in ArxivToolSpec().to_tool_list()]
            )
        elif toolset_name == ToolNames.TOOL_NAME_BASIC_ARITHMETIC_CALCULATOR:
            self._remove_tools_by_names(
                [
                    tool.metadata.name
                    for tool in BasicArithmeticCalculatorSpec().to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_DUCKDUCKGO:
            self._remove_tools_by_names(
                [
                    tool.metadata.name
                    for tool in DuckDuckGoSearchToolSpec().to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_STRING_FUNCTIONS:
            self._remove_tools_by_names(
                [
                    tool.metadata.name
                    for tool in StringFunctionsToolSpec().to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_TAVILY:
            self._remove_tools_by_names(
                [
                    tool.metadata.name
                    for tool in TavilyToolSpec(api_key=FAKE_STRING).to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_WIKIPEDIA:
            self._remove_tools_by_names(
                [tool.metadata.name for tool in WikipediaToolSpec().to_tool_list()]
            )
        elif toolset_name == ToolNames.TOOL_NAME_YAHOO_FINANCE:
            self._remove_tools_by_names(
                [tool.metadata.name for tool in YahooFinanceToolSpec().to_tool_list()]
            )

    def add_or_set_toolset(
        self,
        toolset_name: str,
        api_key: str | None = None,
        remove_existing: bool = True,
    ):
        """
        Add or set the tools for the given toolset.

        Args:
            toolset_name (str): The name of the toolset to add or set.
            api_key (str): The API key to use with the toolset.
            remove_existing (bool): Whether to remove the existing tools for the toolset before adding the new ones.
        """

        # Remove the existing tools for the toolset to avoid duplicates
        if remove_existing:
            self.remove_toolset(toolset_name)

        if toolset_name == ToolNames.TOOL_NAME_ARXIV:
            self.tools.extend(ArxivToolSpec().to_tool_list())
        elif toolset_name == ToolNames.TOOL_NAME_BASIC_ARITHMETIC_CALCULATOR:
            self.tools.extend(BasicArithmeticCalculatorSpec().to_tool_list())
        elif toolset_name == ToolNames.TOOL_NAME_DUCKDUCKGO:
            self.tools.extend(DuckDuckGoSearchToolSpec().to_tool_list())
        elif toolset_name == ToolNames.TOOL_NAME_STRING_FUNCTIONS:
            self.tools.extend(StringFunctionsToolSpec().to_tool_list())
        elif toolset_name == ToolNames.TOOL_NAME_TAVILY:
            self.tools.extend(TavilyToolSpec(api_key=api_key).to_tool_list())
        elif toolset_name == ToolNames.TOOL_NAME_WIKIPEDIA:
            self.tools.extend(WikipediaToolSpec().to_tool_list())
        elif toolset_name == ToolNames.TOOL_NAME_YAHOO_FINANCE:
            self.tools.extend(YahooFinanceToolSpec().to_tool_list())

    def set_web_search_tool(
        self, search_tool: str, search_tool_api_key: str | None = None
    ):
        """
        Set the web search tool to use for the Difficult Questions Attempted engine.

        Args:
            search_tool (str): The web search tool to use.
            api_key (str): The API key to use with the web search tool.
        """

        self.remove_toolset(ToolNames.TOOL_NAME_DUCKDUCKGO)
        self.remove_toolset(ToolNames.TOOL_NAME_TAVILY)

        if search_tool != ToolNames.TOOL_NAME_SELECTION_DISABLE:
            self.add_or_set_toolset(
                search_tool, api_key=search_tool_api_key, remove_existing=False
            )

    def get_descriptive_tools_dataframe(self):
        """
        Get a dataframe consisting of the names and descriptions of the tools currently available.
        """
        return [
            [
                f"`{tool.metadata.name}`",
                tool.metadata.description.split("\n\n")[1].strip(),
            ]
            for tool in self.tools
        ]

    async def run(self, query: str):
        """
        Run the Difficult Questions Attempted engine with the given query.

        Args:
            query (str): The query to process.

        Yields:
            bool: A flag indicating whether the workflow is done.
            Any: The output of the workflow when the status is done. Otherwise, the events streamed by the workflow.
        """
        self.workflow = DQAWorkflow(
            llm=self.llm, tools=self.tools, timeout=180, verbose=False
        )
        # No need for this, see: https://github.com/run-llama/llama_index/discussions/15838#discussioncomment-10553154
        # self.workflow.add_workflows(
        #     react_workflow=ReActWorkflow(
        #         llm=self.llm, tools=self.tools, timeout=60, verbose=True
        #     )
        # )
        task = asyncio.create_task(
            self.workflow.run(
                query=query,
            )
        )
        done: bool = False
        total_steps: int = 0
        finished_steps: int = 0
        async for ev in self.workflow.stream_events():
            total_steps = ev.total_steps
            finished_steps = ev.finished_steps
            print(ev.msg)
            yield done, finished_steps, total_steps, ev.msg
        result = await task
        done = self.workflow.is_done()
        yield done, finished_steps, total_steps, result
