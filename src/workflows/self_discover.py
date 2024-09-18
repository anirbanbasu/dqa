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

"""Variations of the self-discover agent."""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


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

from llama_index.core.prompts import PromptTemplate

from llama_index.core.llms.llm import LLM

from utils import EMPTY_STRING
from workflows.common import WorkflowStatusEvent


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
    """Self discover workflow: https://arxiv.org/abs/2402.03620 with a plan-only option and a bypass option."""

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
        "Given the task: {task}, which of the following reasoning modules are relevant? Do not elaborate on why.\n\n{reasoning_modules}"
        "{bypass_information}"
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
        # Disable it by default because the LLMs have a tendency to bypass the reasoning structure even when it is necessary.
        allow_bypass: bool = False,
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
        self.allow_bypass = allow_bypass

    @step
    async def get_modules(
        self, ctx: Context, ev: StartEvent
    ) -> SelfDiscoverGetModulesEvent | StopEvent:
        """Get modules step."""
        # get input data, store llm into ctx
        task = ev.get("task")

        if task is None:
            raise ValueError("The 'task' argument is required.")

        if self.llm is None:
            raise ValueError("LLM is required for this workflow.")

        # format prompt and get result from LLM
        prompt = SelfDiscoverWorkflow.SELECT_PROMPT_TEMPLATE.format(
            task=task,
            reasoning_modules=SelfDiscoverWorkflow._REASONING_MODULES,
            bypass_information=(
                "If the given task can be solved without a reasoning structure, please output "
                f"'{SelfDiscoverWorkflow.REASONING_OUTPUT_BYPASS_NONE}' only without selecting any reasoning module."
                if self.allow_bypass
                else EMPTY_STRING
            ),
        )
        result = await self.llm.acomplete(prompt)

        if str(result) == SelfDiscoverWorkflow.REASONING_OUTPUT_BYPASS_NONE:
            # Too simple, bypass self-discovery
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg="Task is too simple to require a reasoning structure, bypassing self-discovery."
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
