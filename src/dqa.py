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
from typing import List
from llama_index.tools.arxiv import ArxivToolSpec

from llama_index.tools.wikipedia import WikipediaToolSpec
from llama_index.tools.tavily_research import TavilyToolSpec

from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
from llama_index.core.tools import FunctionTool


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

from workflows.ssq_react import StructuredSubQuestionReActWorkflow


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
        self.workflow = StructuredSubQuestionReActWorkflow(
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
        # No longer usable in this way, due to breaking changes in LlamaIndex Workflows.
        # task = asyncio.create_task(
        #     self.workflow.run(
        #         query=query,
        #     )
        # )
        task: asyncio.Future = self.workflow.run(
            query=query,
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
            done, _ = await asyncio.wait([task])
            if done:
                result = task.result()
        except Exception as e:
            result = f"\nException in running the workflow(s). Type: {type(e).__name__}. Message: '{str(e)}'"
            # Set this to done, otherwise another workflow call cannot be made.
            done = True
            print(result, file=sys.stderr)
        finally:
            progress_bar.close()
        yield done, finished_steps, total_steps, result
