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
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.workflow import (
    step,
    Context,
    Workflow,
    Event,
    StartEvent,
    StopEvent,
)
from llama_index.core.agent import ReActAgent

from utils import parse_env


class QueryEvent(Event):
    question: str


class AnswerEvent(Event):
    question: str
    answer: str


class DQAWorkflow(Workflow):
    @step
    async def query(self, ctx: Context, ev: StartEvent) -> QueryEvent:
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
Given a user question, generate a list of distinct sub questions that you need to answer in order to answer the original question.
Respond with a list only the original question when no decomposition is needed.
Generate questions that explicitly mention the subject by name, avoiding pronouns like 'these,' 'they,' 'he,' 'she,' 'it,' etc.
Each question should clearly state the subject to ensure no ambiguity.

Example 1:
Question: Is Hamlet more common on IMDB than Comedy of Errors?
Decompositions:
{{
    "sub_questions": [
        "How many listings of Hamlet are there on IMDB?"
        "How many listing of Comedy of Errors is there on IMDB?"
    ]
}}

Example 2:
Question: What is the Capital city of Japan?
Decompositions:
{{
    "sub_questions": ["What is the Capital city of Japan?"]
}}
The question needs no decomposition

Always, respond in pure JSON without any Markdown, like this:
{{
    "sub_questions": [
        "sub question 1",
        "sub question 2",
        "sub question 3"
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
    async def sub_question(self, ctx: Context, ev: QueryEvent) -> AnswerEvent:
        ic(f"Attempting sub-question: {ev.question}")

        agent = ReActAgent.from_tools(
            ctx.data["tools"], llm=ctx.data["llm"], verbose=True
        )
        response = agent.chat(ev.question)

        return AnswerEvent(question=ev.question, answer=str(response))

    @step
    async def combine_answers(self, ctx: Context, ev: AnswerEvent) -> StopEvent | None:
        ready = ctx.collect_events(ev, [AnswerEvent] * ctx.data["sub_question_count"])
        if ready is None:
            return None

        answers = "\n\n".join(
            [
                f"Question: {event.question}: \n Answer: {event.answer}"
                for event in ready
            ]
        )

        prompt = f"""
You are given an overall question that has been split into sub-questions, each of which has been answered.
Combine the answers to all the sub-questions into a single answer to the original question.

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
        self.llm = llm
        self.tools = TavilyToolSpec(api_key=parse_env("TAVILY_API_KEY")).to_tool_list()
        self.workflow = DQAWorkflow(timeout=120, verbose=True)

    async def run(self, query: str):
        result = await self.workflow.run(
            llm=self.llm,
            tools=self.tools,
            query=query,
        )
        ic(result)
        return result
