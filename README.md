[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=3776ab&labelColor=e4e4e4)](https://www.python.org/downloads/release/python-3120/)
[![Experimental status](https://img.shields.io/badge/Status-experimental-orange)](#)

# DQA: Difficult Questions Attempted

<p align="right">
  <img width="400" height="200" src="https://raw.githubusercontent.com/anirbanbasu/dqa/master/assets/logo.svg" alt="dqa logo" style="filter: invert(0.5)">
</p>

## Overview

The DQA aka _difficult questions attempted_ project utilises large language models (LLMs) to perform _multi-hop question answering_ (MHQA). This project has been inspired by the tutorial [^1] and the article [^2], both by [Dean Sacoransky](https://www.linkedin.com/in/dean-sacoransky-6a671119a/). Unlike both the tutorial and the article, which use the [LangGraph framework from LangChain](https://langchain-ai.github.io/langgraph/) for building agents, this project makes use of [DSPy](https://dspy.ai/).


[^1]: Sacoransky, D., 2024. Build a RAG agent to answer complex questions. IBM Developer Tutorial. [URL](https://developer.ibm.com/tutorials/awb-build-rag-llm-agents/).
[^2]: Sacoransky, D., 2025. Reasoning & Recursive Retrieval With Deepseek-r1, Tavily, and LangGraph. Medium article. [URL](https://medium.com/@deansaco/reasoning-recursive-retrieval-with-deepseek-r1-tavily-and-langgraph-79b3336731e2).

## Project status

Following is a table of some updates regarding the project status. Note that these do not correspond to specific commits or milestones.

| Date     |  Status   |  Notes or observations   |
|----------|:-------------:|----------------------|
| February 15, 2025 |  active |  Custom adapter added for Deepseek models.  |
| January 26, 2025 |  active |  LlamaIndex Workflows replaced by DSPy.  |
| September 21, 2024 |  active |  Workflows made selectable.  |
| September 13, 2024 |  active |  Low parameter LLMs perform badly in unnecessary self-discovery, query refinements and ReAct tool selections.  |
| September 10, 2024 |  active |  Query decomposition may generate unnecessary sub-workflows.  |
| August 31, 2024 |  active |  Using built-in ReAct agent.  |
| August 29, 2024 |  active |  Project started.  |


## Installation

Install and activate a Python virtual environment in the directory where you have cloned this repository. Let us refer to this directory as the _working directory_ or _WD_ (interchangeably) hereonafter. Install [poetry](https://python-poetry.org/docs/). Make sure you use Python 3.12.0 or later. To install the project with its dependencies in a virtual environment, run the following in the _WD_.

```bash
poetry install
```

In addition to Python dependencies, see the installation instructions of [Ollama](https://ollama.com/download). You can install it on a separate machine. Download the [tool calling model of Ollama](https://ollama.com/search?c=tools) that you want to use, e.g., `llama3.1` or `mistral-nemo`. Reinforcement learning based models such as `deepseek-r1:7b` will also work.

## Environment variables

Following is a list of environment variables that can be used to configure the DQA application. All environment variables should be supplied as quoted strings. They will be interpreted as the correct type as necessary.

For environment variables starting with `GRADIO_`, See [Gradio documentation for environment variables](https://www.gradio.app/guides/environment-variables).

| Variable |  [Default value] and description   |
|--------------|----------------|
<!-- | `ANTHROPIC_API_KEY` | [None] Check the [docs](https://docs.anthropic.com/en/api/getting-started) to get an API key. |
| `LLM__ANTHROPIC_MODEL` | [claude-3-opus-20240229] See the [available models](https://docs.anthropic.com/en/docs/about-claude/models). |
| `OPENAI_API_KEY` | [None] Check the [docs](https://platform.openai.com/) to get an API key. |
| `LLM__OPENAI_MODEL` | [gpt-4o-mini] See the [available models](https://platform.openai.com/docs/models). |
| `COHERE_API_KEY` | [None] Check the [docs](https://dashboard.cohere.com/) to get an API key. |
| `LLM__COHERE_MODEL` | [command-r-plus] See the [available models](https://docs.cohere.com/docs/models). |
| `GROQ_API_KEY` | [None] Check the [docs](https://console.groq.com/) to get an API key. |
| `LLM__GROQ_MODEL` | [llama-3.1-70b-versatile] See the [available models](https://console.groq.com/docs/models). | -->
| `OLLAMA_URL` | [http://localhost:11434] URL of your intended Ollama host. |
| `LLM__OLLAMA_MODEL` | [mistral-nemo] See the [available models](https://ollama.com/library). The model must be available on the selected Ollama server. The model must [support tool calling]((https://ollama.com/search?c=tools)). |
<!-- | LLM__PROVIDER | [Ollama] Select one from the following default list. |
| `SUPPORTED_LLM_PROVIDERS` | [Anthropic:Open AI:Cohere:Groq:Ollama] Separator character is ":". A subset of the default set of LLM providers may be used to restrict access in a particular deployment. |
| `LLM__TEMPERATURE` | [0.0] Inferred type: `float`. This is the temperature setting for the LLM, which is between $[0, 1]$ for all supported LLM providers except Open AI, for which it is $[0, 2]$. |
| `LLM__TOP_P` | [0.4] Inferred type: `float`. This is the nucleus sampling hyperparameter that controls the randomness of the LLM output. This parameter is only available when using the Ollama LLM provider. |
| `LLM__TOP_K` | [40] Inferred type: `int`. This is the top-k setting for the LLM, which controls token selection. This parameter is only available when using the Ollama LLM provider. |
| `LLM__REPEAT_PENALTY` | [1.1] Inferred type: `float`. This is a parameter to control repeated sequences in the output. This parameter is only available when using the Ollama LLM provider. |
| `LLM__SEED` | [1] Inferred type: `int`. This parameter is used to initialise the LLM's sampling process. Any fixed value will result in a deterministic initialisation of the sampling process. This parameter is only available when using the Ollama LLM provider. |
| `TAVILY_API_KEY` | [None] Check the [docs](https://docs.tavily.com/docs/gpt-researcher/getting-started) to get an API key. | -->

## Usage (local)

Create a `.env` file in the _working directory_, to set the environment variables as above. Then, run the following in the _WD_ to start the web server.

```bash
poetry run dqa-webapp
```

The web UI will be available at [http://localhost:7860](http://localhost:7860). To exit the server, use the Ctrl+C key combination.


## Contributing

Install [`pre-commit`](https://pre-commit.com/) for Git and [`ruff`](https://docs.astral.sh/ruff/installation/). Then enable `pre-commit` by running the following in the _WD_.

```bash
pre-commit install
```
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/).
