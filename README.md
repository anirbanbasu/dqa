[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=3776ab&labelColor=e4e4e4)](https://www.python.org/downloads/release/python-3120/)
[![Experimental status](https://img.shields.io/badge/Status-experimental-orange)](#)

# DQA: Difficult Questions Attempted

<p align="right">
  <img width="400" height="200" src="https://raw.githubusercontent.com/anirbanbasu/dqa/master/assets/logo.svg" alt="dqa logo" style="filter: invert(0.5)">
</p>

## Overview

The DQA aka _difficult questions attempted_ project utilises large language models (LLMs) to perform _multi-hop question answering_ (MHQA). This project has been inspired by the tutorial [^1] and the article [^2], both by [Dean Sacoransky](https://www.linkedin.com/in/dean-sacoransky-6a671119a/).

**Note that this README is rather out-of-date, pending a major revision.**


[^1]: Sacoransky, D., 2024. Build a RAG agent to answer complex questions. IBM Developer Tutorial. [URL](https://developer.ibm.com/tutorials/awb-build-rag-llm-agents/).
[^2]: Sacoransky, D., 2025. Reasoning & Recursive Retrieval With Deepseek-r1, Tavily, and LangGraph. Medium article. [URL](https://medium.com/@deansaco/reasoning-recursive-retrieval-with-deepseek-r1-tavily-and-langgraph-79b3336731e2).

## Project status

Following is a table of some updates regarding the project status. Note that these do not correspond to specific commits or milestones.

| Date     |  Status   |  Notes or observations   |
|----------|:-------------:|----------------------|
| June 7, 2025 |  active |  Changed package and project manager to `uv` from `poetry`. Changed LLM orchestration to LangChain/LangGraph from DSPy. Added supporting MCP server. |
| February 15, 2025 |  active |  Custom adapter added for Deepseek models.  |
| January 26, 2025 |  active |  LlamaIndex Workflows replaced by DSPy.  |
| September 21, 2024 |  active |  Workflows made selectable.  |
| September 13, 2024 |  active |  Low parameter LLMs perform badly in unnecessary self-discovery, query refinements and ReAct tool selections.  |
| September 10, 2024 |  active |  Query decomposition may generate unnecessary sub-workflows.  |
| August 31, 2024 |  active |  Using built-in ReAct agent.  |
| August 29, 2024 |  active |  Project started.  |


## Installation

The directory where you clone this repository will be referred to as the _working directory_ or _WD_ hereinafter.

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). To install the project with its essential dependencies in a virtual environment, run the following in the _WD_. To install all non-essential dependencies, add the `--all-extras` flag to the following command.

```bash
uv sync
```

In addition to project dependencies, see the installation instructions of [Ollama](https://ollama.com/download). You can also install it on a separate machine. Download a [tool calling model of Ollama](https://ollama.com/search?c=tools) that you want to use, e.g., `llama3.1` or `mistral-nemo`.

## Environment variables

Following is a list of environment variables that can be used to configure the DQA application. All environment variables should be supplied as quoted strings. They will be interpreted as the correct type as necessary.

For environment variables starting with `GRADIO_`, See [Gradio documentation for environment variables](https://www.gradio.app/guides/environment-variables).

When running the provided MCP server, the following environment variables can be specified, prefixed with `FASTMCP_SERVER_`: `HOST`, `PORT`, `DEBUG` and `LOG_LEVEL`. See [key configuration options](https://gofastmcp.com/servers/fastmcp#key-configuration-options) for FastMCP. Note that `on_duplicate_` prefixed options specified as environment variables _will be ignored_.

| Variable |  [Default value] and description   |
|--------------|----------------|
| `DQA_MCP_CLIENT_CONFIG` | [`config/mcp-client.json`] The path to the config file for providing a set of MCP servers. |
| `DQA_USE_MCP` | [`True`] If this is set to True, then make sure that the MCP client configuration points to available MCP server(s). You may want to run the provided MCP server. See below for instructions regarding that. |
| `DQA_LLM_CONFIG` | [`config/chat-ollama.json`] The path to the config file for the Ollama LLM provider. |
| `LLM__OLLAMA_MODEL` | [mistral-nemo] See the [available models](https://ollama.com/library). The model must be available on the selected Ollama server. The model must [support tool calling](https://ollama.com/search?c=tools). |
| `DQA_MCP_SERVER_TRANSPORT` | [sse] The acceptable options are either `sse` or `streamable-http`. |

The structure of the MCP configuration file is as follows. Note that while the configuration supports the `stdio` transport, the DQA agent is configured to support only `streamable_http` or `sse`, which are also the allowed modes for the DQA MCP server. For examples of the configuration, see [the LangChain MCP adapters](https://github.com/langchain-ai/langchain-mcp-adapters).

```json
{
  "<name>": {
    "transport": "<Either streamable_http or sse>",
    "url": "<url-of-the-remote-server>"
  }
}
```

The structure of the LLM config JSON file is as follows. The full list of acceptable configuration parameters for the Ollama LLM is available in [the LangChain documentation](https://python.langchain.com/api_reference/ollama/chat_models/langchain_ollama.chat_models.ChatOllama.html).

```json
{
    "baseUrl": "<model-provider-url>",
    "model": "<model-name>"
}
```

## Usage

Create a `.env` file in the _WD_, to set the environment variables as above, if you want to use anything other than the default settings.

### Optional DQA MCP server

Run the following in the _WD_ to start the MCP server.

```bash
uv run dqa-mcp
```

### DQA webapp

Run the following in the _WD_ to start the web server.

```bash
uv run dqa-webapp
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
