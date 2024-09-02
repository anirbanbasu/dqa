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

import json
from typing import List
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
from llama_index.core.agent import ReActAgent

from tools import StringFunctionsToolSpec
from utils import parse_env, EnvironmentVariables


class QueryEvent(Event):
    question: str


class AnswerEvent(Event):
    question: str
    answer: str


class DQAWorkflow(Workflow):
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
        """
        As a start event of the workflow, this step receives the original query and stores it in the context.
        It then asks the LLM to decompose the query into sub-questions. Upon decomposition, it emits every
        sub-question as a `QueryEvent`.

        Args:
            ctx (Context): The context object.
            ev (StartEvent): The start event.

        Returns:
            QueryEvent: The event containing the original query.
        """
        if hasattr(ev, "query"):
            ctx.data["original_query"] = ev.query
            print(f"Query is {ctx.data['original_query']}")

        if hasattr(ev, "llm"):
            ctx.data["llm"] = ev.llm

        if hasattr(ev, "tools"):
            ctx.data["tools"] = ev.tools

        response = ctx.data["llm"].complete(
            f"""
You are an assistant for question-answering tasks who performs query decomposition.
Given a user question, generate a list of distinct sub-questions that you need to answer in order to answer the original question.
Respond with a list containing only the unmodified original question when no decomposition is needed.
Generate sub-questions that explicitly mention the subject by name, avoiding pronouns like 'these,' 'they,' 'he,' 'she,' 'it,', and so on.
Each sub-question should clearly state the subject to ensure no ambiguity.
Do not generate unnecessary sub-questions that do not contribute to answering the original question.

Example 1:
Question: Is Hamlet more common on IMDB than Comedy of Errors?
Decompositions:
{{
    "sub_questions": [
        "How many listings of Hamlet are there on IMDB?"
        "How many listings of Comedy of Errors is there on IMDB?"
    ]
}}

Example 2:
Question: What is the capital city of Japan?
Decompositions:
{{
    "sub_questions": ["What is the capital city of Japan?"]
}}
The question needs no decomposition

Always, respond in pure JSON without any Markdown, like this:
{{
    "sub_questions": [
        "sub question 1",
        "sub question 2",
        "sub question 3",
    ]
}}

Here is the user question: {ctx.data['original_query']}

And here is the list of tools: {ctx.data['tools']}
        """
        )

        ic(response)

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]

        ctx.data["sub_question_count"] = len(sub_questions)

        for question in sub_questions:
            self.send_event(QueryEvent(question=question))

        return None

    @step
    async def answer_sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        """
        This step receives a sub-question and attempts to answer it using the tools provided in the context.

        Args:
            ctx (Context): The context object.
            ev (QueryEvent): The event containing the sub-question.

        Returns:
            AnswerEvent: The event containing the sub-question and the answer.
        """
        ic(f"Attempting sub-question: {ev.question}")

        agent = ReActAgent.from_tools(
            ctx.data["tools"],
            llm=ctx.data["llm"],
            verbose=True,
            max_iterations=25,
        )
        response = agent.chat(ev.question)

        return AnswerEvent(question=ev.question, answer=str(response))

    @step
    async def combine_answers(self, ctx: Context, ev: AnswerEvent) -> StopEvent | None:
        """
        This step receives the answers to the sub-questions and combines them into a single answer to the original
        question so long as all the sub-questions have been answered.

        Args:
            ctx (Context): The context object.
            ev (AnswerEvent): The event containing the sub-question and the answer.

        Returns:
            StopEvent | None: The event containing the final answer to the original
        """
        ready = ctx.collect_events(ev, [AnswerEvent] * ctx.data["sub_question_count"])
        if ready is None:
            return None

        answers = "\n\n".join(
            [
                f"Question: {event.question}: \n Answer: {event.answer}"
                for event in ready
            ]
        )

        for event in ready:
            ic(event)

        prompt = f"""
You are given an overall question that has been split into sub-questions, each of which has been answered.
Combine the answers to all the sub-questions into a single and succinct answer to the original question.

Original question: {ctx.data['original_query']}

Sub-questions and answers:
{answers}
        """

        ic(f"Final prompt is {prompt}")

        response = ctx.data["llm"].complete(prompt)

        ic("Final response is", response)

        return StopEvent(result=str(response))


class DQAEngine:
    def __init__(self, llm):
        """
        Initialize the Difficult Questions Attempted engine.

        Args:
            llm (FunctionCallingLLM): The function calling LLM instance to use.
        """
        self.llm = llm
        # Add tool specs
        self.tools: List[FunctionTool] = []
        self.tools.extend(ArxivToolSpec().to_tool_list())
        # DuckDuckGo search tool can end up being used even when other better tools are available, so it is commented out.
        # self.tools.extend(DuckDuckGoSearchToolSpec().to_tool_list())
        self.tools.extend(StringFunctionsToolSpec().to_tool_list())
        self.tools.extend(
            TavilyToolSpec(
                api_key=parse_env(EnvironmentVariables.KEY__TAVILY_API_KEY)
            ).to_tool_list()
        )
        self.tools.extend(WikipediaToolSpec().to_tool_list())
        self.tools.extend(YahooFinanceToolSpec().to_tool_list())

        self.workflow = DQAWorkflow(timeout=120, verbose=True)

    async def run(self, query: str) -> str:
        """
        Run the Difficult Questions Attempted engine with the given query.

        Args:
            query (str): The query to process.

        Returns:
            str: The response from the engine.
        """
        result = await self.workflow.run(
            llm=self.llm,
            tools=self.tools,
            query=query,
        )
        ic(result)
        return result
