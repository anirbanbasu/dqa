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

"""
Implementation of the Language Agent Tree Search (LATS).

A wrapper around the llama-index-packs-agents-lats implementation.
"""

try:
    from icecream import ic
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from typing import Any


from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    StartEvent,
    StopEvent,
)

from workflows.common import WorkflowStatusEvent

from llama_index.core.llms.llm import LLM
from llama_index.core.tools import BaseTool

from llama_index.packs.agents_lats import LATSAgentWorker


class LATSWorkflow(Workflow):
    """
    ## A Language Agent Tree Search (LATS) agent

    This workflow is a wrapper around the implementation in llama-index-packs-agents-lats.

    See the LATS agent described in the paper: https://arxiv.org/pdf/2310.04406v2.pdf.

    **Caveats**
     - May not be able to answer complex queries without a structure. May exceed the total number of rollouts for complex queries, and hence stop.
     - May fail to call tools correctly with low-parameter models.
     - May spend unnecessary time on reflection and candidate expansion.
     - Does not generate a nicely formatted response with a structure.

    """

    def __init__(
        self,
        *args: Any,
        llm: LLM | None = None,
        tools: list[BaseTool] | None = None,
        num_expansions: int = 3,
        max_rollouts: int = 6,
        **kwargs: Any,
    ) -> None:
        """Initialise the LATS workflow."""
        super().__init__(
            *args,
            **kwargs,
        )
        self.llm = llm
        self.tools = tools
        self.num_expansions = num_expansions
        self.max_rollouts = max_rollouts
        self._total_steps: int = 0
        self._finished_steps: int = 0

    @step
    async def run_agent(self, ctx: Context, ev: StartEvent) -> StopEvent:
        """Run the agent."""
        agent_worker = LATSAgentWorker.from_tools(
            tools=self.tools,
            llm=self.llm,
            num_expansions=self.num_expansions,
            max_rollouts=self.max_rollouts,
            verbose=True,
        )
        agent = agent_worker.as_agent()
        task = agent.create_task(ev.input)
        self._total_steps += 1
        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=(
                    f"Starting the LATS agent to answer the question:\n\t{ev.input}\n"
                ),
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        step_output = await agent.arun_step(task.task_id)
        self._finished_steps += 1
        while not step_output.is_last:
            self._total_steps += 1
            ctx.write_event_to_stream(
                WorkflowStatusEvent(
                    msg=(
                        f"Obtained response: {step_output.output}\n"
                        "Running the next step."
                    ),
                    total_steps=self._total_steps,
                    finished_steps=self._finished_steps,
                )
            )
            step_output = await agent.arun_step(task.task_id)
            self._finished_steps += 1

        response = agent.finalize_response(task.task_id)

        ctx.write_event_to_stream(
            WorkflowStatusEvent(
                msg=("Done, final response generated.\n" f"{response}" "\n"),
                total_steps=self._total_steps,
                finished_steps=self._finished_steps,
            )
        )
        return StopEvent(result=str(response))
