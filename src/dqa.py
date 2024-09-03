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

from tools import StringFunctionsToolSpec, BasicArithmeticCalculatorSpec
from utils import parse_env, EnvironmentVariables


class QueryEvent(Event):
    question: str


class AnswerEvent(Event):
    question: str
    answer: str


class ReviewSubQuestionEvent(Event):
    questions: List[str]
    satisfied: bool = False


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
Do not generate unnecessary sub-questions that are not required to answer the original question.

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
Note that this question above needs no decomposition. Hence, the original question is repeated as the only sub-question.

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

        ic(self.query.__name__)

        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]

        ctx.data["sub_question_count"] = len(sub_questions)

        if len(sub_questions) == 1:
            return QueryEvent(question=sub_questions[0])
        else:
            return ReviewSubQuestionEvent(questions=sub_questions)

    @step
    async def review_sub_questions(
        self, ctx: Context, ev: ReviewSubQuestionEvent
    ) -> QueryEvent | ReviewSubQuestionEvent:
        """
        This step receives the sub-questions and asks the LLM to review them. If the LLM is satisfied with the
        sub-questions, they can be used to answer the original question. Otherwise, the LLM can provide updated
        sub-questions.

        Args:
            ctx (Context): The context object.
            ev (QueryEvent): The event containing the sub-question.

        Returns:
            QueryEvent | ReviewSubQuestionEvent: The event containing the sub-question or the event to review the
            sub-questions.
        """

        if ev.satisfied:
            # Already satisfied, no need to review.
            for question in ev.questions:
                ctx.send_event(QueryEvent(question=question))

        response = ctx.data["llm"].complete(
            f"""
You are given an overall question that has been decomposed into sub-questions.
Review each sub-questions and improve it, if necessary. Remove any sub-questions that is not required to answer the original query..
Do not add new sub-questions, unless necessary. Remember that the sub-questions represent a concise decomposition of the original question.
Ensure that sub-questions explicitly mention the subject by name, avoiding pronouns like 'these,' 'they,' 'he,' 'she,' 'it,', and so on.
Each sub-question should clearly state the subject to ensure no ambiguity.
Do not output sub-questions that are not required to answer the original question.

Lastly, generate a binary response indicating whether you are satisfied with the amended sub-questions or not.

Always, respond in pure JSON without any Markdown, like this:
{{
    "sub_questions": [
        "sub question 1",
        "sub question 2",
        "sub question 3",
    ],
    "satisfied": true or false
}}

Here is the user question: {ctx.data['original_query']}

And here is the list of tools: {ctx.data['tools']}

Sub-questions and answers:
{ev.questions}
            """
        )
        ic(self.review_sub_questions.__name__)
        response_obj = json.loads(str(response))
        sub_questions = response_obj["sub_questions"]
        ctx.data["sub_question_count"] = len(sub_questions)

        if len(sub_questions) == 1:
            return QueryEvent(question=sub_questions[0])

        if response_obj["satisfied"]:
            for question in sub_questions:
                ctx.send_event(QueryEvent(question=question))
        else:
            # Not satisfied, so ask for review again.
            return ReviewSubQuestionEvent(
                questions=sub_questions, satisfied=response_obj["satisfied"]
            )

        return None

    @step(num_workers=4)
    async def answer_sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        """
        This step receives a sub-question and attempts to answer it using the tools provided in the context.

        Args:
            ctx (Context): The context object.
            ev (QueryEvent): The event containing the sub-question.

        Returns:
            AnswerEvent: The event containing the sub-question and the answer.
        """

        agent = ReActAgent.from_tools(
            ctx.data["tools"],
            llm=ctx.data["llm"],
            verbose=True,
            max_iterations=25,
        )
        response = agent.chat(ev.question)
        ic(self.answer_sub_question.__name__)

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

        if len(ready) == 1:
            # Nothing to combine if there was only ever one sub-question.
            return StopEvent(result=ready[0].answer)

        answers = "\n\n".join(
            [
                f"Question: {event.question}: \n Answer: {event.answer}"
                for event in ready
            ]
        )

        prompt = f"""
You are given an overall question that has been split into sub-questions, each of which has been answered.
Combine the answers to all the sub-questions into a single answer to the original question. Your answer should be a coherent response to the original question.
Ensure that your final answer includes all the relevant information from the answers to the sub-questions. Do not miss out on any important details and nuances.

Original question: {ctx.data['original_query']}

Sub-questions and answers:
{answers}
        """

        response = ctx.data["llm"].complete(prompt)
        ic(self.combine_answers.__name__)
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
        self.tools.extend(
            TavilyToolSpec(
                api_key=parse_env(EnvironmentVariables.KEY__TAVILY_API_KEY)
            ).to_tool_list()
        )
        self.tools.extend(WikipediaToolSpec().to_tool_list())
        self.tools.extend(YahooFinanceToolSpec().to_tool_list())

        # Custom tools
        self.tools.extend(StringFunctionsToolSpec().to_tool_list())
        self.tools.extend(BasicArithmeticCalculatorSpec().to_tool_list())

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
