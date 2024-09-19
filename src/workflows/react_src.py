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

"""ReAct with Structured Reasoning in Context workflow."""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

import asyncio

# Weaker LLMs may generate horrible JSON strings.
# `dirtyjson` is more lenient than `json` in parsing JSON strings.
from typing import Any, List

from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)


from llama_index.core.llms.llm import LLM
from llama_index.core.tools.types import BaseTool

from workflows.common import WorkflowStatusEvent
from workflows.react import ReActWorkflow
from workflows.self_discover import SelfDiscoverWorkflow


class ReActSRCReasoningStructureEvent(Event):
    """
    Event to handle reasoning structure for ReActSRC.

    Fields:
        reasoning_structure (str): The reasoning structure.
    """

    reasoning_structure: str


class ReActSRCAnswerEvent(Event):
    """
    Event to handle a ReActSRC answer.

    Fields:
        question (str): The question.
        reasoning_structure (str): The reasoning structure.
        answer (str): The answer.
        sources (List[Any]): The sources.
    """

    question: str
    reasoning_structure: str
    answer: str
    sources: List[Any] = []


class ReActWithStructuredReasoningInContextWorkflow(Workflow):
    """A workflow implementation for ReActSRC: ReAct with Strucutred Reasoning In Context."""

    KEY_ORIGINAL_QUERY = "original_query"

    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the ReActSRC workflow.

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
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> ReActSRCReasoningStructureEvent | StopEvent:
        """
        As a start event of the workflow, this step receives the original query and stores it in the context.
        It then invokes a self-discovery workflow to generate a reasoning structure for the query.

        Args:
            ctx (Context): The context object.
            ev (StartEvent): The start event.

        Returns:
            ReActSRCReasoningStructureEvent | StopEvent: The event containing the reasoning structure or the event to stop the workflow.
        """
        if hasattr(ev, "query"):
            await ctx.set(
                ReActWithStructuredReasoningInContextWorkflow.KEY_ORIGINAL_QUERY,
                ev.query,
            )
        else:
            return StopEvent(result="No query provided. Try again!")

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Generating a reasoning structure for the query:\n\t{ev.query}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        self_discover_workflow = SelfDiscoverWorkflow(
            llm=self.llm,
            # Let's set the timeout of the self-discover workflow to half of the ReActSRC workflow's timeout.
            timeout=self._timeout / 2,
            verbose=self._verbose,
            plan_only=True,
        )
        self_discover_task: asyncio.Future = self_discover_workflow.run(task=ev.query)

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

        done, _ = await asyncio.wait([self_discover_task])
        if done:
            response = self_discover_task.result()
        self._finished_steps += 1

        return ReActSRCReasoningStructureEvent(reasoning_structure=response)

    @step
    async def answer_with_context(
        self, ctx: Context, ev: ReActSRCReasoningStructureEvent
    ) -> ReActSRCAnswerEvent:
        """
        This step receives the reasoning structure and the original query, and then invokes the ReAct workflow to
        answer the original query using the reasoning structure.

        Args:
            ctx (Context): The context object.
            ev (ReActSRCReasoningStructureEvent): The event containing the reasoning structure.

        Returns:
            ReActSRCAnswerEvent: The event containing the answer to the original query.
        """

        question = await ctx.get(
            ReActWithStructuredReasoningInContextWorkflow.KEY_ORIGINAL_QUERY
        )

        react_context = (
            "\nPlease use the following reasoning structure in your thinking process while answering the question given to you. "
            "The reasoning structure can help you decompose the given question into relevant sub-questions. "
            "Please ignore any partial solution present in the reasoning structure. "
            f"\nReasoning structure:\n{ev.reasoning_structure}"
        )

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=(
                    f"Starting the {ReActWorkflow.__name__} to answer the question:\n\t{question}"
                    f"\n\nReAct context:{react_context}"
                ),
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        react_workflow = ReActWorkflow(
            llm=self.llm,
            tools=self.tools,
            # Let's set the timeout of the ReAct workflow to half of the ReActSRC workflow's timeout.
            timeout=self._timeout / 2,
            verbose=self._verbose,
            # Let's keep the maximum iterations of the ReAct workflow to its default value.
            extra_context=react_context,
        )

        react_task: asyncio.Future = react_workflow.run(input=question)

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

        done, _ = await asyncio.wait([react_task])
        if done:
            response = react_task.result()
        self._finished_steps += 1

        return ReActSRCAnswerEvent(
            question=question,
            answer=response[ReActWorkflow.KEY_RESPONSE],
            # TODO: Should we format the sources nicely here so that the LLM does not have to deal with it later?
            sources=[
                tool_output.content
                for tool_output in response[ReActWorkflow.KEY_SOURCES]
            ],
            reasoning_structure=ev.reasoning_structure,
        )

    @step
    async def combine_refine_answers(
        self, ctx: Context, ev: ReActSRCAnswerEvent
    ) -> StopEvent | None:
        """
        This step receives the answer to the original query and the reasoning structure, and then generates the final response.

        Args:
            ctx (Context): The context object.
            ev (ReActSRCAnswerEvent): The event containing the answer to the original query.

        Returns:
            StopEvent | None: The event to stop the workflow.
        """

        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=f"Generating the final response to the original query:\n\t{ev.question}",
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )

        prompt = (
            "You are a linguistic expert who generates a coherent and structured response from the information provided to you."
            "\nYou are given a question that has been answered. You are given the answer and relevant sources, if any. "
            "You are also given a reasoning structure that was used to answer the question. "
            "\nAdhering to the reasoning structure, rephrase the answer to the question. "
            "Ensure that your final answer includes all the relevant details and nuances from the answer. "
            "State the ambiguities and conflicting information that you encounter. "
            "If the answer contain errors, state those errors in your response without correcting them. "
            "In your final answer, cite each source and its corresponding URLs, only if such source URLs are available."
            "\nDo not make up sources or URLs if they have not been given to you. "
            "\nYour final answer must be correctly formatted as pure HTML (with no Javascript and Markdown) in a concise, readable and visually pleasing way. "
            "Enclose your HTML response with a <div> tag that has an attribute `id` set to the value 'workflow_response'."
            "\nDO NOT hallucinate!"
            f"\n\nOriginal question: {ev.question}"
            f"\n\nAnswer:\n{ev.answer}"
            f"\n\nSources:\n{', '.join(ev.sources)}"
            f"\n\nReasoning structure:\n{ev.reasoning_structure}"
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
