[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=3776ab&labelColor=e4e4e4)](https://www.python.org/downloads/release/python-3120/)
[![Experimental status](https://img.shields.io/badge/Status-experimental-orange)](#)
[![Python linter and format checker with ruff](https://github.com/anirbanbasu/dqa/actions/workflows/python-linter-format-checker.yml/badge.svg)](https://github.com/anirbanbasu/dqa/actions/workflows/python-linter-format-checker.yml)

# DQA: Difficult Questions Attempted

<p align="right">
  <img width="400" height="200" src="https://raw.githubusercontent.com/anirbanbasu/dqa/master/assets/logo.svg" alt="dqa logo" style="filter: invert(0.5)">
</p>

## Overview

The DQA aka _difficult questions attempted_ project utilises one or more agents to facilitate large language models (LLMs) perform _multi-hop question answering_ (MHQA). This project is inspired by a tutorial [^1] from [Dean Sacoransky](https://www.linkedin.com/in/dean-sacoransky-6a671119a/). Unlike the tutorial's use of the [LangGraph framework from LangChain](https://langchain-ai.github.io/langgraph/) for building agents, this project makes use of [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/).

### An example of a difficult question and LLM responses
The tutorial uses the question _Which David Fincher film that stars Edward Norton does not star Brad Pitt?_ as a litmus test for assessing new AI systems. The answer is supposed to be _None_, but at the time of writing the tutorial (26 August 2024), the author states that ChatGPT's `gpt-4o` model generates the following response.
> The David Fincher film starring Edward Norton that does not star Brad Pitt is the "The Game" (1997). Edward Norton appears in an uncredited cameo role in this film.

The author further states that it is impossible "to answer this complex, multi-hop, logical question in one feed-forward pass of a neural network". At the end of the tutorial, the improved response to the question using agents that perform retrieval augmented generation (RAG) is seen to be the following.
> None, as there is only one mentioned David Fincher film starring Edward Norton, which is "Fight Club" and it stars Brad Pitt.

## Query decomposition, refinement and ReAct to the rescue

This project implements an agent-based framework akin to the one mentioned in the tutorial [^1].

### A simpler difficult question
Let's focus on a slightly simpler test question that nonetheless baffles ChatGPT `gpt-4o`. Let's ask _Tell me the result of multiplying the number of 'r's in the word 'strawberry' with the sum of 3 and 4. Explain the process._ ChatGPT hopelessly responds to this with the following.

> Let's break down the problem step by step:
>
> Find the number of 'r's in the word "strawberry": The word "strawberry" contains 2 'r's.
>
> Find the sum of 3 and 4: $3+4=7$.
>
> Multiply the number of 'r's by the sum: $2 \times 7=14$.
>
> So, the result of multiplying the number of 'r's in "strawberry" by the sum of 3 and 4 is 14.

While ChatGPT did not make mistakes with the basic arithmetic operations, it counted the number of 'r's in the word 'strawberry' wrong. In contrast, the DQA workflow gets it right with the following answer, as of September 6, 2024, surprisingly using the `gpt-4o-mini` model.

> Final Answer: The result of multiplying the number of 'r's in the word 'strawberry' (which is 3) with the sum of 3 and 4 (which is 7) is 21.
>
> Detailed Explanation: To arrive at the final answer, we first determined the number of 'r's in the word 'strawberry'. The analysis revealed that there are 3 'r's in the word. Next, we calculated the sum of 3 and 4, which is 7. Finally, we multiplied these two results together: 3 (the number of 'r's) multiplied by 7 (the sum of 3 and 4) equals 21. Therefore, the final result is 21.

The reason the `gpt-4o-mini` model is able to count the number of 'r's correctly is because DQA lets it use a function to calculate the occurrences of a specific character or a sequence of characters in a string.

### The agent workflow
The approximate current workflow for DQA can be summarised as follows.
![Workflow](./diagrams/workflow.svg)

The DQA workflow uses a [self-discover](https://arxiv.org/abs/2402.03620) "agent" to produce a reasoning structure but not answer the question. Similar to the tutorial [^1], the DQA workflow performs query decomposition with respect to the reasoning structure to ensure that complex queries are not directly sent to the LLM. Instead, sub-questions (i.e., decompositions of the complex query) that help answer the complex query are sent. The workflow further optimises the sub-questions through a query refinement step, which loops if necessary, for a maximum number of allowed iterations.

Once the refined sub-questions are satisfactory, the sub-questions are answered sequentially by separate instances of a [ReAct](https://arxiv.org/abs/2210.03629) "agent", also implemented as a workflow. ReAct workflows have the responses from previous ReAct workflows as contextual information.

When all ReAct workflows have finished, the final step for answer generation collects the responses from the ReAct workflows and asks the LLM to generate a consolidated answer citing sources where relevant, in accordance with the initially generated reasoning structure.

### Response to the initial difficult question
Recalling the litmus test question (i.e., _Which David Fincher film that stars Edward Norton does not star Brad Pitt?_), the response from DQA with `gpt-4o-mini` is correct, as in the answer is _none_, but the response is long-winded.

> The David Fincher film that stars Edward Norton but does not feature Brad Pitt is **none**. The only film directed by David Fincher that includes both Edward Norton and Brad Pitt is Fight Club (1999). In this film, Edward Norton plays the unnamed narrator, while Brad Pitt portrays Tyler Durden. Therefore, there are no David Fincher films starring Edward Norton that exclude Brad Pitt.
>
> To summarize:
>
> - Film featuring both Edward Norton and Brad Pitt: Fight Club (1999)
> - Other films directed by David Fincher include:
>   - Alien 3 (1992)
>   - Se7en (1995)
>   - The Game (1997)
>   - Panic Room (2002)
>   - Zodiac (2007)
>   - The Curious Case of Benjamin Button (2008)
>   - The Social Network (2010)
>   - The Girl with the Dragon Tattoo (2011)
>   - Gone Girl (2014)
>   - Mank (2020)

### Inconsistency and the need for improvement
The generated responses depend heavily on the LLM making them very inconsistent. In addition, while the workflow passes on the examples shown here, there remains room for improvement, with respect to wasteful LLM calls, wasteful tool calls, consistency of the answer from the same LLM, ability to generate reliable answers from low parameter quantised models (available on Ollama, for instance), amongst others.

[^1]: Sacoransky, D., 2024. Build a RAG agent to answer complex questions. IBM Developer Tutorial. [URL](https://developer.ibm.com/tutorials/awb-build-rag-llm-agents/).

## Project status

Following is a table of some updates regarding the project status. Note that these do not correspond to specific commits or milestones.

| Date     |  Status   |  Notes or observations   |
|----------|:-------------:|----------------------|
| September 17, 2024 |  active |  ReAct workflows handle questions sequentially instead of in parallel.  |
| September 15, 2024 |  active |  Vector storage is not used as of now. Qdrant may be removed in the future.  |
| September 13, 2024 |  active |  Low parameter LLMs perform badly in unnecessary self-discovery, query refinements and ReAct tool selections.  |
| September 12, 2024 |  active |  Self-discover may need to be conditionally bypassed to reduce the number of unnecessary LLM calls.  |
| September 10, 2024 |  active |  Query decomposition may generate unnecessary sub-workflows.  |
| September 7, 2024 |  active |  Cohere `command-r-plus` is _very_ slow.  |
| August 31, 2024 |  active |  Using built-in ReAct agent.  |
| August 29, 2024 |  active |  Project started.  |


## Installation

Install and activate a Python virtual environment in the directory where you have cloned this repository. Let us refer to this directory as the _working directory_ or _WD_ (interchangeably) hereonafter. You could do that using [pyenv](https://github.com/pyenv/pyenv), for example. Make sure you use Python 3.12.0 or later. Inside the activated virtual environment, run the following.

```bash
python -m pip install -U pip
python -m pip install -U -r requirements.txt
```
While calling `pip install` with `-U` on `requirements.txt` will install the latest packages, this may create an environment with unforeseen bugs and incompatibilities. To create a more stable environment, run `pip` on a list of packages that specifies package versions.

```bash
python -m pip install -r requirements-frozen.txt
```

If necessary, you can uninstall everything previously installed by `pip` (in the virtual environment) by running the following.

```bash
python -m pip freeze | cut -d "@" -f1 | xargs pip uninstall -y
```

In addition to Python dependencies, see the installation instructions of [Ollama](https://ollama.com/download). You can install it on a separate machine. Download the [tool calling model of Ollama](https://ollama.com/search?c=tools) that you want to use, e.g., `llama3.1` or `mistral-nemo`.

## Usage (local)

Make a copy of the file `.env.docker` in the _working directory_ as a `.env` file.

```bash
cp .env.docker .env
```

Change all occurrences of `host.docker.internal` to `localhost` or some other host or IP assuming that you have Ollama on port 11434 on your preferred host. Set the Ollama model to the tool calling model that you have downloaded on your Ollama installation. Set the value of the `LLM_PROVIDER` to the provider that you want to use. Supported names are `Anthropic`, `Cohere`, `Groq`, `Ollama` and `Open AI`.

You can use the environment variable `SUPPORTED_LLM_PROVIDERS` to further restrict the supported LLM providers to a subset of the aforementioned, such as, by setting the value to `Groq:Ollama` to allow only Groq and Ollama for some deployment of this app. Note that the only separating character between LLM provider names is a `:`. If you add a provider that is not in the aforementioned set, the app will throw an error and refuse to start.

Add the API keys for [Anthropic](https://console.anthropic.com/), [Cohere](https://dashboard.cohere.com/welcome/login), [Groq](https://console.groq.com/keys) or [Open AI](https://platform.openai.com/docs/overview) if you want to use any of these. In addition, add [an API key of Tavily](https://app.tavily.com/sign-in).
<!-- Qdrant API key is not necessary if you are not using [Qdrant cloud](https://qdrant.tech/documentation/qdrant-cloud-api/). -->

With all these setup done, run the following to start the web server. The web server will serve a web user interface as well as a REST API. It is not configured to use HTTPS.

```bash
python src/webapp.py
```

The web UI will be available at [http://localhost:7860](http://localhost:7860).

## Usage (Docker)

In the `.env.docker`, Ollama is expected to be available on port 11434 on your Docker host, i.e., `host.docker.internal`. Set that to some other host(s), if that is where your Ollama server is available. Set the Ollama model to the tool calling model that you have downloaded on your Ollama installation.

Set the value of the `LLM_PROVIDER` to the provider that you want to use and add the API keys for Anthropic, Cohere, Groq and Open AI LLM providers as well as that of Tavily as metioned above in the **Usage (local)** section.

With all these setup done, and assuming that you have Docker installed, you can build an image of the DQA app, create a container and start it as follows.

```bash
docker build -f local.dockerfile -t dqa .
docker create -p 7860:7860/tcp --name dqa-container dqa
docker container start dqa-container
```

You can replace the second line above to the following, in order to use a `.env` file on your Docker host that resides at the **absolute** path `PATH_TO_YOUR_.env_FILE`.

```bash
docker create -v /PATH_TO_YOUR_.env_FILE:/home/app_user/app/.env -p 7860:7860/tcp --name dqa-container dqa
```

The web server will serve a web user interface as well as a REST API at [http://localhost:7860](http://localhost:7860). It is not configured to use HTTPS.

## Contributing

Install [`pre-commit`](https://pre-commit.com/) for Git and [`ruff`](https://docs.astral.sh/ruff/installation/). Then enable `pre-commit` by running the following in the _WD_.

```bash
pre-commit install
```
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/).
