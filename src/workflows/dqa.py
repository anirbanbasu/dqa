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

"""Difficult Questions Attempted module containing various workflows."""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import sys
from tqdm import tqdm
import asyncio

# Weaker LLMs may generate horrible JSON strings.
# `dirtyjson` is more lenient than `json` in parsing JSON strings.
import dirtyjson as json
from typing import Any, List
from llama_index.tools.arxiv import ArxivToolSpec

from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec

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

from tools import (
    DuckDuckGoFullSearchOnlyToolSpec,
    StringFunctionsToolSpec,
    BasicArithmeticCalculatorSpec,
    MathematicalFunctionsSpec,
)
from utils import (
    APP_TITLE_SHORT,
    FAKE_STRING,
    ToolNames,
    get_terminal_size,
)  # , parse_env, EnvironmentVariables


from llama_index.core.llms.llm import LLM
from llama_index.core.tools.types import BaseTool

from workflows.common import WorkflowStatusEvent
from workflows.react import ReActWorkflow
from workflows.self_discover import SelfDiscoverWorkflow


class DQAReasoningStructureEvent(Event):
    """
    Event to handle reasoning structure for DQA.

    Fields:
        question (str): The question.
        reasoning_structure (str): The reasoning structure.
    """

    reasoning_structure: str


class DQASequentialQueryEvent(Event):
    """
    Event to handle a DQA query from a list of questions. This event is used to handle questions sequentially.
    The question index is used to keep track of the current question being handled. The list of questions are
    expected to be stored in the context.

    Fields:
        query_index (int): The index of the query in the list of queries.
    """

    # TODO: Always send the first sub-question and let the answer step loop through the rest.
    # This is to ensure that all sub-questions are answered even if they are dependent on each other.
    # In contrast, the current mechanism of sending all sub-questions in parallel may lead incomplete answers, if
    # the LLM is unable to answer a sub-question due to dependencies.
    # Make sure that the first sub-question is not dependent on any of the rest -- prompt engineering.

    question_index: int = 0


class DQAAnswerEvent(Event):
    """
    Event to handle a DQA answer.

    Fields:
        question (str): The question.
        answer (str): The answer.
        sources (List[Any]): The sources.
    """

    question: str
    answer: str
    sources: List[Any] = []


class DQAReviewSubQuestionEvent(Event):
    """
    Event to review the sub-questions.

    Fields:
        questions (List[str]): The sub-questions.
        satisfied (bool): Whether the sub-questions are satisfied.
    """

    questions: List[str]
    satisfied: bool = False


class DQAWorkflow(Workflow):
    """A workflow implementation for DQA: Difficult Questions Attempted."""

    KEY_ORIGINAL_QUERY = "original_query"
    KEY_REASONING_STRUCTURE = "reasoning_structure"
    KEY_SUB_QUESTIONS = "sub_questions"
    KEY_SUB_QUESTIONS_COUNT = "sub_questions_count"
    KEY_SATISFIED = "satisfied"
    KEY_REACT_CONTEXT = "react_context"

    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        max_refinement_iterations: int = 3,
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

        self._max_refinement_iterations: int = max_refinement_iterations
        self._refinement_iterations: int = 0

    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> DQAReasoningStructureEvent | StopEvent:
        """
        As a start event of the workflow, this step receives the original query and stores it in the context.

        Args:
            ctx (Context): The context object.
            ev (StartEvent): The start event.

        Returns:
            DQAReasoningStructureEvent | StopEvent: The event containing the reasoning structure or the event to stop the workflow.
        """
        if hasattr(ev, "query"):
            await ctx.set(DQAWorkflow.KEY_ORIGINAL_QUERY, ev.query)
            await ctx.set(
                DQAWorkflow.KEY_REACT_CONTEXT,
                (
                    "\nPAY ATTENTION since the given question may make implicit references to information in the context. "
                    "If you find or can deduce the answer to the given question from the context below, PLEASE refrain from calling any further tools. "
                    "Instead, formulate the answer to the given question from the context. "
                    "PLEASE answer the given question only. Do NOT answer the original question below. It is provided for context only."
                    f"\nOriginal question: {ev.query}"
                ),
            )
        else:
            return StopEvent(result="No query provided. Try again!")

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Generating structured reasoning for the query:\n\t{await ctx.get(DQAWorkflow.KEY_ORIGINAL_QUERY)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        self_discover_workflow = SelfDiscoverWorkflow(
            llm=self.llm,
            # Let's set the timeout of the ReAct workflow to half of the DQA workflow's timeout.
            timeout=self._timeout / 2,
            verbose=self._verbose,
            plan_only=True,
        )
        self_discover_task = asyncio.create_task(
            self_discover_workflow.run(task=ev.query)
        )

        async for nested_ev in self_discover_workflow.stream_events():
            self._total_steps += 1
            self._finished_steps += 1
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=f"[{SelfDiscoverWorkflow.__name__}]\n{nested_ev.msg}",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )

        response = await self_discover_task
        self._finished_steps += 1

        return DQAReasoningStructureEvent(reasoning_structure=response)

    @step
    async def query(
        self, ctx: Context, ev: DQAReasoningStructureEvent
    ) -> DQASequentialQueryEvent | DQAReviewSubQuestionEvent | StopEvent:
        """
        This step receives the structured reasoning for the query.
        It then asks the LLM to decompose the query into sub-questions. Upon decomposition, it emits every
        sub-question as a query event. Alternatively, if the LLM is not satisfied with the sub-questions, it
        emits a sub question review event to review the sub-questions.

        Args:
            ctx (Context): The context object.
            ev (DQAStructuredReasoningEvent): The event containing the structured reasoning.

        Returns:
            DQASequentialQueryEvent | DQAReviewSubQuestionEvent | StopEvent: The event containing the sub-question index to process or the event
            to review the sub-questions or the event to stop the workflow.
        """

        await ctx.set(DQAWorkflow.KEY_REASONING_STRUCTURE, ev.reasoning_structure)

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Assessing query and plan:\n\t{await ctx.get(DQAWorkflow.KEY_ORIGINAL_QUERY)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are a linguistic expert who performs efficient query decomposition."
            "\nBelow, you are given a question and a corresponding reasoning structure to answer it. "
            "Generate a minimalist list of distinct and absolutely essential sub-questions, each of which must be answered in order to answer the original question according to the suggested structured reasoning. "
            "If a sub-question is already implicitly answered through the reasoning structure then do not include it in the list of sub-questions. "
            "If the original question cannot be or need not be decomposed then output a list of sub-questions that contain the original question as the only sub-question. "
            "Otherwise, do not include the original question in the list of sub-questions. "
            "In the sub-questions, explicitly mention the subject by name, avoiding pronouns like 'these,' 'they,' 'he,' 'she,' 'it,', and so on. "
            "Each sub-question should clearly state the subject to ensure no ambiguity. "
            "Do not generate sub-questions that are not required to answer the original question. "
            "\n\nLastly, reflect on the generated sub-questions and output a binary response indicating whether you are satisfied with the generated sub-questions or not. "
            "\n\nExample 1:\n"
            "Question: Is Hamlet more common on IMDB than Comedy of Errors?\n"
            "Decompositions:\n"
            "{\n"
            f'    "{DQAWorkflow.KEY_SUB_QUESTIONS}": [\n'
            '        "How many listings of Hamlet are there on IMDB?",\n'
            '        "How many listings of Comedy of Errors is there on IMDB?"\n'
            "    ],\n"
            f'    "{DQAWorkflow.KEY_SATISFIED}": true\n'
            "}"
            "\n\nExample 2:\n"
            "Question: What is the capital city of Japan?\n"
            "Decompositions:\n"
            "{\n"
            f'    "{DQAWorkflow.KEY_SUB_QUESTIONS}": ["What is the capital city of Japan?"],\n'
            f'    "{DQAWorkflow.KEY_SATISFIED}": true\n'
            "}\n"
            "Note that this question above needs no decomposition. Hence, the original question is output as the only sub-question."
            "\n\nExample 3:\n"
            "Question: Are there more hydrogen atoms in methyl alcohol than in ethyl alcohol?\n"
            "Decompositions:\n"
            "{\n"
            f'    "{DQAWorkflow.KEY_SUB_QUESTIONS}": [\n'
            '        "How many hydrogen atoms are there in methyl alcohol?",\n'
            '        "How many hydrogen atoms are there in ethyl alcohol?",\n'
            '        "What is the chemical composition of alcohol?"\n'
            "    ],\n"
            f'    "{DQAWorkflow.KEY_SATISFIED}": false\n'
            "}\n"
            "Note that the third sub-question is unnecessary and should not be included. Hence, the value of the satisfied flag is set to false."
            "\n\nAlways, respond in pure JSON without any Markdown, like this:\n"
            "{\n"
            f'    "{DQAWorkflow.KEY_SUB_QUESTIONS}": [\n'
            '        "sub question 1",\n'
            '        "sub question 2",\n'
            '        "sub question 3"\n'
            "    ],\n"
            f'    "{DQAWorkflow.KEY_SATISFIED}": true or false\n'
            "}"
            "\nDO NOT hallucinate!"
            f"\n\nHere is the user question: {await ctx.get(DQAWorkflow.KEY_ORIGINAL_QUERY)}"
            f"\n\nAnd, here is the corresponding reasoning structure:\n{await ctx.get(DQAWorkflow.KEY_REASONING_STRUCTURE)}"
        )
        response = await self.llm.acomplete(prompt)
        self._finished_steps += 1

        response_obj = json.loads(str(response))
        sub_questions = response_obj[DQAWorkflow.KEY_SUB_QUESTIONS]
        satisfied = response_obj[DQAWorkflow.KEY_SATISFIED]

        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"{'Satisfactory' if satisfied else 'Unsatisfactory'} sub-questions:\n\t{str(sub_questions)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        await ctx.set(DQAWorkflow.KEY_SUB_QUESTIONS, sub_questions)
        await ctx.set(DQAWorkflow.KEY_SUB_QUESTIONS_COUNT, len(sub_questions))

        # Ignore the satisfied flag if there is only one sub-question.
        if len(sub_questions) == 1:
            return DQASequentialQueryEvent()
        else:
            if satisfied:
                return DQASequentialQueryEvent()
            else:
                return DQAReviewSubQuestionEvent(
                    questions=sub_questions, satisfied=satisfied
                )

    @step
    async def review_sub_questions(
        self, ctx: Context, ev: DQAReviewSubQuestionEvent
    ) -> DQASequentialQueryEvent | DQAReviewSubQuestionEvent:
        """
        This step receives the sub-questions and asks the LLM to review them. If the LLM is satisfied with the
        sub-questions, they can be used to answer the original question. Otherwise, the LLM can provide updated
        sub-questions.

        Args:
            ctx (Context): The context object.
            ev (DQAReviewSubQuestionEvent): The event containing the sub-questions.

        Returns:
            DQASequentialQueryEvent | DQAReviewSubQuestionEvent: The event containing the sub-question index to process or the event to review the
            sub-questions.
        """

        if ev.satisfied:
            # Already satisfied, no need to review anymore.
            return DQASequentialQueryEvent()

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Reviewing sub-questions:\n\t{str(ev.questions)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are a linguistic expert who performs query decomposition and its systematic review."
            "\nYou are given a question that has been decomposed into sub-questions, which are also given below. "
            "Furthermore, you are provided with the reasoning structure to answer the original question. The sub-questions are generated in accordance with the reasoning structure. "
            "Review each sub-question and improve it, if necessary. Minimise the number of sub-questions. "
            # "Each sub-question must be possible to answer without depending on the answer from another sub-question. "
            # "A sub-question should not be a subset of another sub-question. "
            "If the original question cannot be or need not be decomposed then output a list of sub-questions that contain the original question as the only sub-question. "
            "Otherwise, do not include the original question in the list of sub-questions. "
            "Remove any sub-question that is not absolutely required to answer the original query. "
            "Remove any sub-question that is already implicitly answered through the reasoning structure. "
            "Do not add new sub-questions, unless necessary. Remember that the sub-questions represent a concise decomposition of the original question. "
            "\nEnsure that sub-questions explicitly mention the subject by name, avoiding pronouns like 'these,' 'they,' 'he,' 'she,' 'it,', and so on. "
            "Each sub-question should clearly state the subject to ensure no ambiguity. "
            "\n\nLastly, reflect on the amended sub-questions and generate a binary response indicating whether you are satisfied with the amended sub-questions or not."
            "\n\nAlways, respond in pure JSON without any Markdown, like this:"
            "{\n"
            f'    "{DQAWorkflow.KEY_SUB_QUESTIONS}": [\n'
            '        "sub question 1",\n'
            '        "sub question 2",\n'
            '        "sub question 3"\n'
            "    ],\n"
            f'    "{DQAWorkflow.KEY_SATISFIED}": true or false\n'
            "}"
            "\nDO NOT hallucinate!"
            f"\n\nHere is the user question: {await ctx.get(DQAWorkflow.KEY_ORIGINAL_QUERY)}"
            f"\n\nHere are the sub-questions for you to review:\n{ev.questions}"
            f"\n\nAnd, here is the corresponding reasoning structure:\n{await ctx.get(DQAWorkflow.KEY_REASONING_STRUCTURE)}"
        )
        response = await self.llm.acomplete(prompt)
        self._finished_steps += 1
        self._refinement_iterations += 1

        response_obj = json.loads(str(response))
        sub_questions = response_obj[DQAWorkflow.KEY_SUB_QUESTIONS]
        satisfied = response_obj[DQAWorkflow.KEY_SATISFIED]

        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"{'Satisfactory' if satisfied else 'Unsatisfactory'} refined sub-questions:\n\t{str(sub_questions)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        await ctx.set(DQAWorkflow.KEY_SUB_QUESTIONS, sub_questions)
        await ctx.set(DQAWorkflow.KEY_SUB_QUESTIONS_COUNT, len(sub_questions))

        # Ignore the satisfied flag if there is only one sub-question.
        if len(sub_questions) == 1:
            return DQASequentialQueryEvent()

        if satisfied or self._refinement_iterations >= self._max_refinement_iterations:
            return DQASequentialQueryEvent()
        else:
            return DQAReviewSubQuestionEvent(
                questions=sub_questions, satisfied=satisfied
            )

    @step
    async def answer_sub_question(
        self, ctx: Context, ev: DQASequentialQueryEvent
    ) -> DQAAnswerEvent | DQASequentialQueryEvent:
        """
        This step receives a sub-question and attempts to answer it using the tools provided in the context.

        Args:
            ctx (Context): The context object.
            ev (DQASequentialQueryEvent): The event containing the sub-question index to process.

        Returns:
            DQAAnswerEvent: The event containing the sub-question and the answer.
        """

        if isinstance(ev, DQASequentialQueryEvent):
            sub_questions = await ctx.get(DQAWorkflow.KEY_SUB_QUESTIONS)
            if not sub_questions or len(sub_questions) == 0:
                raise ValueError("No questions to answer.")
            question = sub_questions[ev.question_index]
        else:
            question = ev.question

        react_context = await ctx.get(DQAWorkflow.KEY_REACT_CONTEXT)

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=(
                    f"Starting a {ReActWorkflow.__name__} to answer question:\n\t{question}"
                    f"\n\nAdditional context:{react_context}"
                ),
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
            # Let's keep the maximum iterations of the ReAct workflow to its default value.
            extra_context=react_context,
        )

        react_task = asyncio.create_task(react_workflow.run(input=question))

        async for nested_ev in react_workflow.stream_events():
            self._total_steps += 1
            self._finished_steps += 1
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=f"[{ReActWorkflow.__name__}]\n{nested_ev.msg}",
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )

        response = await react_task
        self._finished_steps += 1

        react_answer_event = DQAAnswerEvent(
            question=question,
            answer=response[ReActWorkflow.KEY_RESPONSE],
            # TODO: Should we format the sources nicely here so that the LLM does not have to deal with it later?
            sources=[
                tool_output.content
                for tool_output in response[ReActWorkflow.KEY_SOURCES]
            ],
        )

        react_context += (
            f"\n\nRelated question: {question}\n"
            f"> Answer: {react_answer_event.answer}\n"
            # TODO: Do we need the sources or is that too much information?
            # f"> Sources: {', '.join(react_answer_event.sources)}"
        )

        await ctx.set(DQAWorkflow.KEY_REACT_CONTEXT, react_context)

        ctx.send_event(react_answer_event)

        if isinstance(ev, DQASequentialQueryEvent):
            if ev.question_index + 1 < len(sub_questions):
                # Let's move to the next sub-question.
                return DQASequentialQueryEvent(question_index=ev.question_index + 1)

        return None

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
            ev, [DQAAnswerEvent] * await ctx.get(DQAWorkflow.KEY_SUB_QUESTIONS_COUNT)
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

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Generating the final response to the original query:\n\t{await ctx.get(DQAWorkflow.KEY_ORIGINAL_QUERY)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are a linguistic expert who generates a coherent summary of the information provided to you."
            "\nYou are given a question that has been split into sub-questions, each of which has been answered."
            "\nYou are also given a reasoning structure that was used to generate the sub-questions. "
            "\nCombine the answers to all the sub-questions into a single and coherent response to the original question. "
            "Your response should match the reasoning structure. "
            "Ensure that your final answer includes all the relevant details and nuances from the answers to the sub-questions. "
            "If the original question has been answered by a single sub-question, refine the answer to make it concise and coherent. "
            "Likewise, if there are ambiguities and/or conflicting information in the answers to the sub-questions, resolve them to generate the final answer. "
            "However, state the ambiguities and conflicting information that you encountered. "
            "If the answer to any of the sub-questions contain errors, state those errors in the final answer without correcting them. "
            "In your final answer, cite each source and its corresponding URLs, only if such source URLs are available are in the answers to the sub-questions."
            "\nDo not make up sources or URLs if they are not present in the answers to the sub-questions. "
            "\nYour final answer must be correctly formatted as pure HTML (with no Javascript and Markdown) in a concise, readable and visually pleasing way. "
            "Enclose your HTML response with a <div> tag that has an attribute `id` set to the value 'dqa_workflow_response'."
            "\nDO NOT hallucinate!"
            f"\n\nOriginal question: {await ctx.get(DQAWorkflow.KEY_ORIGINAL_QUERY)}"
            f"\n\nSub-questions, answers and relevant sources:\n{answers}"
        )

        response = await self.llm.acomplete(prompt)
        self._finished_steps += 1

        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=("Done, final response generated.\n" f"{response}" "\n"),
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        return StopEvent(result=str(response))


class DQAEngine:
    """The Difficult Questions Attempted engine."""

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
        self.tools.extend(DuckDuckGoFullSearchOnlyToolSpec().to_tool_list())

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
        elif toolset_name == ToolNames.TOOL_NAME_MATHEMATICAL_FUNCTIONS:
            return self._are_tools_present(
                [
                    tool.metadata.name
                    for tool in MathematicalFunctionsSpec().to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_DUCKDUCKGO:
            return self._are_tools_present(
                [
                    tool.metadata.name
                    for tool in DuckDuckGoFullSearchOnlyToolSpec().to_tool_list()
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
        elif toolset_name == ToolNames.TOOL_NAME_MATHEMATICAL_FUNCTIONS:
            self._remove_tools_by_names(
                [
                    tool.metadata.name
                    for tool in MathematicalFunctionsSpec().to_tool_list()
                ]
            )
        elif toolset_name == ToolNames.TOOL_NAME_DUCKDUCKGO:
            self._remove_tools_by_names(
                [
                    tool.metadata.name
                    for tool in DuckDuckGoFullSearchOnlyToolSpec().to_tool_list()
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
        elif toolset_name == ToolNames.TOOL_NAME_MATHEMATICAL_FUNCTIONS:
            self.tools.extend(MathematicalFunctionsSpec().to_tool_list())
        elif toolset_name == ToolNames.TOOL_NAME_DUCKDUCKGO:
            self.tools.extend(DuckDuckGoFullSearchOnlyToolSpec().to_tool_list())
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
            tuple: A tuple containing the done status, the number of finished steps, the total number of steps, and the message
            for each step of the workflow. The message is the response to the query when the workflow is done.
        """
        # Instantiating the ReAct workflow instead may not be always enough to get the desired responses to certain questions.
        self.workflow = DQAWorkflow(
            llm=self.llm,
            tools=self.tools,
            timeout=180,
            verbose=False,
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
                # input=query,
                # task=query,
            )
        )
        done: bool = False
        total_steps: int = 0
        finished_steps: int = 0
        terminal_columns, _ = get_terminal_size()
        progress_bar = tqdm(
            total=total_steps,
            leave=False,
            unit="step",
            ncols=int(terminal_columns / 2),
            desc=APP_TITLE_SHORT,
            colour="yellow",
        )
        async for ev in self.workflow.stream_events():
            total_steps = ev.total_steps
            finished_steps = ev.finished_steps
            print(f"\n{str(ev.msg)}", flush=True)
            # TODO: Is tqdm.write better than printf?
            # tqdm.write(f"\n{str(ev.msg)}")
            progress_bar.reset(total=total_steps)
            progress_bar.update(finished_steps)
            progress_bar.refresh()
            yield done, finished_steps, total_steps, ev.msg
        try:
            result = await task
            done = self.workflow.is_done()
        except Exception as e:
            result = f"\nException in running the workflow(s). Type: {type(e).__name__}. Message: '{str(e)}'"
            # Set this to done, otherwise another workflow call cannot be made.
            done = True
            print(result, file=sys.stderr)
        finally:
            progress_bar.close()
        yield done, finished_steps, total_steps, result
