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
import uuid

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

from llama_index.core.prompts import PromptTemplate

# from llama_index.core.agent import ReActAgent

from tools import (
    DuckDuckGoFullSearchOnlyToolSpec,
    StringFunctionsToolSpec,
    BasicArithmeticCalculatorSpec,
    MathematicalFunctionsSpec,
)
from utils import (
    APP_TITLE_SHORT,
    EMPTY_STRING,
    FAKE_STRING,
    ToolNames,
    get_terminal_size,
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


# Generic Events
class WorkflowStatusEvent(Event):
    """
    Event to update the status of the workflow.

    Fields:
        msg (str): The message to display.
        total_steps (int): Optional total number of steps, defaults to zero.
        finished_steps (int): Optional number of steps finished, defaults to zero.
    """

    msg: str
    total_steps: int = 0
    finished_steps: int = 0


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
    """A workflow implementation for a ReAct agent: https://arxiv.org/abs/2210.03629"""

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

        response = await self.llm.achat(chat_history)

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
            ctx.write_event_to_stream(WorkflowStatusEvent(msg=streaming_message))
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
                WorkflowStatusEvent(msg=f"{ReActWorkflow.KEY_ERROR.capitalize()}: {e}")
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
                        observation=f"Tool {tool_call.tool_name} does not exist."
                    )
                )
                ctx.write_event_to_stream(
                    WorkflowStatusEvent(
                        msg=f"{ReActWorkflow.KEY_WARNING.capitalize()}: Tool {tool_call.tool_name} does not exist."
                    )
                )
                continue

            try:
                tool_output = tool(**tool_call.tool_kwargs)
                self.sources.append(tool_output)
                (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                    ObservationReasoningStep(observation=tool_output.content)
                )
                ctx.write_event_to_stream(
                    WorkflowStatusEvent(
                        msg=f"{ReActWorkflow.KEY_OBSERVATION.capitalize()}: {tool_output.content}",
                    )
                )
            except Exception as e:
                (await ctx.get(ReActWorkflow.KEY_CURRENT_REASONING, default=[])).append(
                    ObservationReasoningStep(
                        observation="Error calling tool {tool.metadata.get_name()}: {e}"
                    )
                )
                ctx.write_event_to_stream(
                    WorkflowStatusEvent(
                        msg=f"{ReActWorkflow.KEY_ERROR.capitalize()}: Failed calling tool {tool.metadata.get_name()}: {e}"
                    )
                )

        # prep the next iteraiton
        return ReActPrepEvent()


class SelfDiscoverGetModulesEvent(Event):
    """
    Event to get modules.

    Fields:
        task (str): The task.
        modules (str): The modules.
    """

    task: str
    modules: str


class SelfDiscoverRefineModulesEvent(Event):
    """
    Event to refine modules.

    Fields:
        task (str): The task.
        refined_modules (str): The refined modules
    """

    task: str
    refined_modules: str


class SelfDiscoverReasoningStructureEvent(Event):
    """
    Event to create reasoning structure.

    Fields:
        task (str): The task.
        reasoning_structure (str): The reasoning structure.
    """

    task: str
    reasoning_structure: str


class SelfDiscoverWorkflow(Workflow):
    """Self discover workflow: https://arxiv.org/abs/2402.03620 with a plan-only option"""

    REASONING_OUTPUT_BYPASS_NONE = "None"

    _REASONING_MODULES = [
        "1. How could I devise an experiment to help solve that problem?",
        "2. Make a list of ideas for solving this problem, and apply them one by one to the problem to see if any progress can be made.",
        "3. How could I measure progress on this problem?",
        "4. How can I simplify the problem so that it is easier to solve?",
        "5. What are the key assumptions underlying this problem?",
        "6. What are the potential risks and drawbacks of each solution?",
        "7. What are the alternative perspectives or viewpoints on this problem?",
        "8. What are the long-term implications of this problem and its solutions?",
        "9. How can I break down this problem into smaller, more manageable parts?",
        "10. Critical Thinking: This style involves analyzing the problem from different perspectives, questioning assumptions, and evaluating the evidence or information available. It focuses on logical reasoning, evidence-based decision-making, and identifying potential biases or flaws in thinking.",
        "11. Try creative thinking, generate innovative and out-of-the-box ideas to solve the problem. Explore unconventional solutions, thinking beyond traditional boundaries, and encouraging imagination and originality.",
        "12. Seek input and collaboration from others to solve the problem. Emphasize teamwork, open communication, and leveraging the diverse perspectives and expertise of a group to come up with effective solutions.",
        "13. Use systems thinking: Consider the problem as part of a larger system and understanding the interconnectedness of various elements. Focuses on identifying the underlying causes, feedback loops, and interdependencies that influence the problem, and developing holistic solutions that address the system as a whole.",
        "14. Use Risk Analysis: Evaluate potential risks, uncertainties, and tradeoffs associated with different solutions or approaches to a problem. Emphasize assessing the potential consequences and likelihood of success or failure, and making informed decisions based on a balanced analysis of risks and benefits.",
        "15. Use Reflective Thinking: Step back from the problem, take the time for introspection and self-reflection. Examine personal biases, assumptions, and mental models that may influence problem-solving, and being open to learning from past experiences to improve future approaches.",
        "16. What is the core issue or problem that needs to be addressed?",
        "17. What are the underlying causes or factors contributing to the problem?",
        "18. Are there any potential solutions or strategies that have been tried before? If yes, what were the outcomes and lessons learned?",
        "19. What are the potential obstacles or challenges that might arise in solving this problem?",
        "20. Are there any relevant data or information that can provide insights into the problem? If yes, what data sources are available, and how can they be analyzed?",
        "21. Are there any stakeholders or individuals who are directly affected by the problem? What are their perspectives and needs?",
        "22. What resources (financial, human, technological, etc.) are needed to tackle the problem effectively?",
        "23. How can progress or success in solving the problem be measured or evaluated?",
        "24. What indicators or metrics can be used?",
        "25. Is the problem a technical or practical one that requires a specific expertise or skill set? Or is it more of a conceptual or theoretical problem?",
        "26. Does the problem involve a physical constraint, such as limited resources, infrastructure, or space?",
        "27. Is the problem related to human behavior, such as a social, cultural, or psychological issue?",
        "28. Does the problem involve decision-making or planning, where choices need to be made under uncertainty or with competing objectives?",
        "29. Is the problem an analytical one that requires data analysis, modeling, or optimization techniques?",
        "30. Is the problem a design challenge that requires creative solutions and innovation?",
        "31. Does the problem require addressing systemic or structural issues rather than just individual instances?",
        "32. Is the problem time-sensitive or urgent, requiring immediate attention and action?",
        "33. What kinds of solution typically are produced for this kind of problem specification?",
        "34. Given the problem specification and the current best solution, have a guess about other possible solutions."
        "35. Let's imagine the current best solution is totally wrong, what other ways are there to think about the problem specification?"
        "36. What is the best way to modify this current best solution, given what you know about these kinds of problem specification?"
        "37. Ignoring the current best solution, create an entirely new solution to the problem."
        "38. Let's think step by step."
        "39. Let's make a step by step plan and implement it with good notation and explanation.",
    ]

    _REASONING_MODULES = "\n".join(_REASONING_MODULES)

    SELECT_PROMPT_TEMPLATE = PromptTemplate(
        "Given the task: {task}, assess if it requires a reasoning structure to solve. "
        "If a reasoning structure is necessary to solve the task, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n {reasoning_modules}"
        f"If a reasoning structure is unnecessary, please output '{REASONING_OUTPUT_BYPASS_NONE}' only without selecting any reasoning module."
    )

    ADAPT_PROMPT_TEMPLATE = PromptTemplate(
        "Without working out the full solution, adapt the following reasoning modules to be specific to our task:\n{selected_modules}\n\nOur task:\n{task}"
    )

    IMPLEMENT_PROMPT_TEMPLATE = PromptTemplate(
        "Without working out the full solution, create an actionable reasoning structure for the task using these adapted reasoning modules:\n{adapted_modules}\n\nTask Description:\n{task}"
    )

    REASONING_PROMPT_TEMPLATE = PromptTemplate(
        "Using the following reasoning structure: {reasoning_structure}\n\nSolve this task, providing your final answer: {task}"
    )

    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        plan_only: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SelfDiscover workflow.

        Args:
            llm (LLM): The LLM instance to use.
            plan_only (bool): Whether to plan only or output a final result.
        """
        super().__init__(*args, **kwargs)

        self.llm = llm
        self.plan_only = plan_only

    @step
    async def get_modules(
        self, ctx: Context, ev: StartEvent
    ) -> SelfDiscoverGetModulesEvent | StopEvent:
        """Get modules step."""
        # get input data, store llm into ctx
        task = ev.get("task")

        if task is None:
            raise ValueError("'task' argument is required.")

        if self.llm is None:
            raise ValueError("LLM is required for this workflow.")

        # format prompt and get result from LLM
        prompt = SelfDiscoverWorkflow.SELECT_PROMPT_TEMPLATE.format(
            task=task, reasoning_modules=SelfDiscoverWorkflow._REASONING_MODULES
        )
        result = await self.llm.acomplete(prompt)

        if str(result) == SelfDiscoverWorkflow.REASONING_OUTPUT_BYPASS_NONE:
            # Too simple, bypass self-discovery
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg="Task is too simple to need a reasoning structure, bypassing self-discovery."
                )
            )
            return StopEvent(result=str(result))
        else:
            ctx.write_event_to_stream(
                WorkflowStatusEvent(msg=f"Selected modules: {result}")
            )
            return SelfDiscoverGetModulesEvent(task=task, modules=str(result))

    @step
    async def refine_modules(
        self, ctx: Context, ev: SelfDiscoverGetModulesEvent
    ) -> SelfDiscoverRefineModulesEvent:
        """Refine modules step."""
        task = ev.task
        modules = ev.modules

        # format prompt and get result
        prompt = SelfDiscoverWorkflow.ADAPT_PROMPT_TEMPLATE.format(
            task=task, selected_modules=modules
        )
        result = await self.llm.acomplete(prompt)

        ctx.write_event_to_stream(WorkflowStatusEvent(msg=f"Refined modules: {result}"))

        return SelfDiscoverRefineModulesEvent(task=task, refined_modules=str(result))

    @step
    async def create_reasoning_structure(
        self, ctx: Context, ev: SelfDiscoverRefineModulesEvent
    ) -> SelfDiscoverReasoningStructureEvent:
        """Create reasoning structures step."""
        task = ev.task
        refined_modules = ev.refined_modules

        # format prompt, get result
        prompt = SelfDiscoverWorkflow.IMPLEMENT_PROMPT_TEMPLATE.format(
            task=task, adapted_modules=refined_modules
        )
        result = await self.llm.acomplete(prompt)

        ctx.write_event_to_stream(
            WorkflowStatusEvent(msg=f"Reasoning structure: {result}")
        )

        if self.plan_only:
            return StopEvent(result=str(result))
        else:
            return SelfDiscoverReasoningStructureEvent(
                task=task, reasoning_structure=str(result)
            )

    @step
    async def get_final_result(
        self, ctx: Context, ev: SelfDiscoverReasoningStructureEvent
    ) -> StopEvent:
        """Gets final result from reasoning structure event."""
        task = ev.task
        reasoning_structure = ev.reasoning_structure

        # format prompt, get res
        prompt = SelfDiscoverWorkflow.REASONING_PROMPT_TEMPLATE.format(
            task=task, reasoning_structure=reasoning_structure
        )
        result = await self.llm.acomplete(prompt)

        ctx.write_event_to_stream(WorkflowStatusEvent(msg=f"Final result: {result}"))

        return StopEvent(result=result)


class DQAReasoningStructureEvent(Event):
    """
    Event to handle reasoning structure for DQA.

    Fields:
        question (str): The question.
        reasoning_structure (str): The reasoning structure.
    """

    reasoning_structure: str


class DQAQueryEvent(Event):
    """
    Event to handle an individual DQA query.

    Fields:
        question (str): The question to handle.
    """

    question: str


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
            ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY] = ev.query
        else:
            return StopEvent(result="No query provided. Try again!")

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Generating structured reasoning for the query:\n\t{ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}",
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

        return DQAReasoningStructureEvent(reasoning_structure=response)

    @step
    async def query(
        self, ctx: Context, ev: DQAReasoningStructureEvent
    ) -> DQAQueryEvent | DQAReviewSubQuestionEvent | StopEvent:
        """
        This step receives the structured reasoning for the query.
        It then asks the LLM to decompose the query into sub-questions. Upon decomposition, it emits every
        sub-question as a `DQAQueryEvent`. Alternatively, if the LLM is not satisfied with the sub-questions, it
        emits a `DQAReviewSubQuestionEvent` to review the sub-questions.

        Args:
            ctx (Context): The context object.
            ev (DQAStructuredReasoningEvent): The event containing the structured reasoning.

        Returns:
            DQAQueryEvent | DQAReviewSubQuestionEvent | StopEvent: The event containing the sub-question or the event
            to review the sub-questions or the event to stop the workflow.
        """

        ctx.data[DQAWorkflow.KEY_REASONING_STRUCTURE] = ev.reasoning_structure

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Assessing query and plan:\n\t{ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are a linguistic expert who performs efficient query decomposition."
            "\nBelow, you are given a question and a corresponding reasoning structure to answer it. "
            "Generate a minimalist list of distinct, independent and absolutely essential sub-questions, each of which must be answered in order to answer the original question according to the suggested structured reasoning. "
            "If a sub-question is already implicitly answered through the reasoning structure then do not include it in the list of sub-questions. "
            "Each sub-question must be possible to answer without depending on the answer from another sub-question. "
            "A sub-question should not be a subset of another sub-question. "
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
            "Note that this question above needs no decomposition. Hence, the original question is output as the only sub-question."
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
            "\nDO NOT hallucinate!"
            f"\n\nHere is the user question: {ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}"
            f"\n\nAnd, here is the corresponding reasoning structure:\n{ctx.data[DQAWorkflow.KEY_REASONING_STRUCTURE]}"
        )
        response = await self.llm.acomplete(prompt)

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]
        satisfied = response_obj["satisfied"]

        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"{'Satisfactory' if satisfied else 'Unsatisfactory'} sub-questions:\n\t{str(sub_questions)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        ctx.data["sub_question_count"] = len(sub_questions)

        # Ignore the satisfied flag if there is only one sub-question.
        if len(sub_questions) == 1:
            return DQAQueryEvent(question=sub_questions[0])
        else:
            if satisfied:
                for question in sub_questions:
                    ctx.send_event(DQAQueryEvent(question=question))
            else:
                return DQAReviewSubQuestionEvent(
                    questions=sub_questions, satisfied=satisfied
                )

        return None

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
            "Each sub-question must be possible to answer without depending on the answer from another sub-question. "
            "A sub-question should not be a subset of another sub-question. "
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
            '    "sub_questions": [\n'
            '        "sub question 1",\n'
            '        "sub question 2",\n'
            '        "sub question 3"\n'
            "    ],\n"
            '    "satisfied": true or false\n'
            "}"
            "\nDO NOT hallucinate!"
            f"\n\nHere is the user question: {ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}"
            f"\n\nHere are the sub-questions for you to review:\n{ev.questions}"
            f"\n\nAnd, here is the corresponding reasoning structure:\n{ctx.data[DQAWorkflow.KEY_REASONING_STRUCTURE]}"
        )
        response = await self.llm.acomplete(prompt)
        self._refinement_iterations += 1

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]
        satisfied = response_obj["satisfied"]

        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"{'Satisfactory' if satisfied else 'Unsatisfactory'} refined sub-questions:\n\t{str(sub_questions)}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        ctx.data["sub_question_count"] = len(sub_questions)

        # Ignore the satisfied flag if there is only one sub-question.
        if len(sub_questions) == 1:
            return DQAQueryEvent(question=sub_questions[0])

        if satisfied or self._refinement_iterations >= self._max_refinement_iterations:
            for question in sub_questions:
                ctx.send_event(DQAQueryEvent(question=question))
        else:
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
            WorkflowStatusEvent(
                msg=f"Starting a {ReActWorkflow.__name__} to answer question:\n\t{ev.question}",
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
        )
        react_task = asyncio.create_task(react_workflow.run(input=ev.question))

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

        self._total_steps += 1
        self._finished_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Generating the final response to the original query:\n\t{ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are a linguistic expert who generates a coherent summary of the information provided to you."
            "\nYou are given a question that has been split into sub-questions, each of which has been answered."
            "\nYou are also given a reasoning structure that was used to generate the sub-questions. "
            "\nCombine the answers to all the sub-questions into a single and coherent response to the original question. "
            "Your response should be in accordance with the reasoning structure. "
            "Ensure that your final answer includes all the relevant details and nuances from the answers to the sub-questions. "
            "If the original question has been answered by a single sub-question, refine the answer to make it more concise and coherent. "
            "If the answer to any of the sub-questions contain errors, state those errors in the final answer without correcting them. "
            "In your final answer, cite the sources and their corresponding URLs, if source URLs are available are in the answers to the sub-questions."
            "\nDo not make up sources or URLs if they are not present in the answers to the sub-questions. "
            "\nYour final answer must be correctly formatted as pure HTML (with no Javascript and Markdown) in a concise, readable and visually pleasing way. "
            "Enclose your HTML response with a <div> tag that has an attribute `id` set to the value 'dqa_workflow_response'."
            "\nDO NOT hallucinate!"
            f"\n\nOriginal question: {ctx.data[DQAWorkflow.KEY_ORIGINAL_QUERY]}"
            f"\n\nSub-questions, answers and relevant sources:\n{answers}"
        )

        response = await self.llm.acomplete(prompt)

        self._finished_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Done, final response generated.\n\nFinal response: {response}",
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
            progress_bar.close()
        except Exception as e:
            result = f"Exception in running the workflow(s): {str(e)}"
            # Set this to done, otherwise another workflow call cannot be made.
            done = True
            progress_bar.close()
            print(result, file=sys.stderr)
        yield done, finished_steps, total_steps, result
